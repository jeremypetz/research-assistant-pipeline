"""
title: Research Assistant
author: Jeremy
date: 2026-02-28
version: 1.0
license: MIT
description: Multi-pass deep research pipeline using Tavily or SearXNG+Crawl4AI for web search, with LLM-driven gap analysis, controversy detection, source credibility scoring, and structured report generation.
requirements: requests
"""

import re
import json
import time
import requests
from datetime import datetime, timezone
from typing import List, Union, Generator, Iterator, Optional
from pydantic import BaseModel, Field


# =============================================================================
# === CONSTANTS ===============================================================
# =============================================================================

DEPTH_PRESETS = {
    "fast": {
        "maxPasses": 2,
        "resultsPerPass": 5,
        "evalSnippetLimit": 15,
        "controversySnippetLimit": 15,
        "reportSnippetLimit": 20,
        "maxCharsPerSnippet": 800,
        "reportMaxCharsPerSnippet": 1500,
        "enableSubtopicDive": False,
        "enableCounterPerspective": False,
        "reportWordTarget": "800-1200",
        "minKeyFindings": 4,
        "minDetailedParagraphs": 2,
    },
    "standard": {
        "maxPasses": 4,
        "resultsPerPass": 5,
        "evalSnippetLimit": 20,
        "controversySnippetLimit": 25,
        "reportSnippetLimit": 30,
        "maxCharsPerSnippet": 800,
        "reportMaxCharsPerSnippet": 2000,
        "enableSubtopicDive": False,
        "enableCounterPerspective": True,
        "reportWordTarget": "1500-2500",
        "minKeyFindings": 6,
        "minDetailedParagraphs": 4,
    },
    "thorough": {
        "maxPasses": 6,
        "resultsPerPass": 7,
        "evalSnippetLimit": 25,
        "controversySnippetLimit": 30,
        "reportSnippetLimit": 40,
        "maxCharsPerSnippet": 1000,
        "reportMaxCharsPerSnippet": 2500,
        "enableSubtopicDive": True,
        "enableCounterPerspective": True,
        "reportWordTarget": "2500-4000",
        "minKeyFindings": 8,
        "minDetailedParagraphs": 6,
    },
}

CREDIBILITY_LEVELS = ["off", "low", "medium", "high"]

SIMPLE_PRESET = {
    "resultsPerPass": 6,
    "maxCharsPerSnippet": 600,
    "reportMaxCharsPerSnippet": 1200,
    "reportWordTarget": "200-400",
}


# =============================================================================
# === HELPERS =================================================================
# =============================================================================

def detect_tavily_topic(query: str) -> str:
    """Auto-detect Tavily topic category from the query."""
    q = query.lower()
    finance_re = re.compile(
        r"\b(stock|stocks|market|markets|investment|investing|portfolio|trading|"
        r"cryptocurrency|crypto|bitcoin|ethereum|earnings|revenue|valuation|ipo|"
        r"nasdaq|s&p|dow jones|finance|financial|dividend|hedge fund|mutual fund|"
        r"bond|bonds|forex|etf)\b"
    )
    news_re = re.compile(
        r"\b(breaking|latest|current|recent|today|yesterday|2024|2025|2026|"
        r"election|scandal|announcement|news|update|developing|reported)\b"
    )
    if finance_re.search(q):
        return "finance"
    if news_re.search(q):
        return "news"
    return "general"


def truncate_to_sentence(text: str, max_chars: int) -> str:
    """Truncate text to the nearest sentence boundary."""
    if not text or len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_period = max(
        truncated.rfind(". "),
        truncated.rfind(".\n"),
        truncated.rfind("! "),
        truncated.rfind("? "),
    )
    if last_period > max_chars * 0.6:
        return truncated[: last_period + 1] + " [...]"
    return truncated + "..."


def detect_content_field(result: dict) -> str:
    """Detect which content field was used from a search result."""
    if result.get("raw_content"):
        return "raw_content"
    if result.get("content"):
        return "content"
    if result.get("snippet"):
        return "snippet"
    if result.get("description"):
        return "description"
    return "none"


def extract_sources(results: list, max_chars: int = 800) -> list:
    """Extract and normalise sources from search results."""
    if not results:
        return []

    sources = []
    for idx, r in enumerate(results):
        if not r or not r.get("url"):
            continue
        candidates = [
            r.get("raw_content"),
            r.get("content"),
            r.get("snippet"),
            r.get("description"),
        ]
        candidates = [c for c in candidates if isinstance(c, str) and c.strip()]
        richest = candidates[0] if candidates else "(no content available)"
        truncated = truncate_to_sentence(richest, max_chars)
        sources.append(
            {
                "index": idx + 1,
                "title": r.get("title", "Untitled Source"),
                "url": r["url"],
                "content": truncated,
                "fullContentLength": len(richest),
                "score": r.get("score"),
                "publishedDate": r.get("published_date") or r.get("publishedDate"),
                "contentField": detect_content_field(r),
            }
        )
    return sources


def deduplicate_sources(sources: list) -> list:
    """Deduplicate sources by URL and re-index."""
    seen = set()
    deduped = []
    for s in sources:
        if s["url"] in seen:
            continue
        seen.add(s["url"])
        deduped.append(s)
    for idx, s in enumerate(deduped):
        s["index"] = idx + 1
    return deduped


def assess_content_quality(results: list) -> dict:
    """Assess content quality across search results."""
    total = len(results)
    rich_count = 0
    total_chars = 0
    for r in results:
        content = r.get("raw_content") or r.get("content") or r.get("snippet") or r.get("description") or ""
        total_chars += len(content)
        if len(content) > 500:
            rich_count += 1
    avg_chars = round(total_chars / total) if total else 0
    rich_pct = round((rich_count / total) * 100) if total else 0
    return {
        "summary": f"{rich_count}/{total} rich results | avg {avg_chars} chars | {rich_pct}% rich",
        "avgChars": avg_chars,
        "richCount": rich_count,
    }


def build_snippet_block(sources: list, limit: int) -> str:
    """Build a text block of source snippets for inclusion in prompts."""
    parts = []
    for s in sources[:limit]:
        date = f" ({s['publishedDate']})" if s.get("publishedDate") else ""
        score = f" [relevance: {s['score']:.2f}]" if s.get("score") else ""
        parts.append(f"[{s['index']}] {s['title']}{date}{score}\nURL: {s['url']}\n{s['content']}")
    return "\n\n---\n\n".join(parts)


def parse_input(user_message: str, default_depth: str, default_credibility: str):
    """
    Parse user message for depth:/credibility: keywords (Option C).
    Returns (topic, depth_key, cred_level).
    """
    text = user_message.strip()
    depth_key = default_depth
    cred_level = default_credibility

    # Match depth:fast, depth:thorough, etc.
    depth_match = re.search(r"\bdepth:(\w+)\b", text, re.IGNORECASE)
    if depth_match:
        val = depth_match.group(1).lower()
        if val in DEPTH_PRESETS or val == "auto":
            depth_key = val
        text = text[: depth_match.start()] + text[depth_match.end() :]

    # Match credibility:high, credibility:off, etc.
    cred_match = re.search(r"\bcredibility:(\w+)\b", text, re.IGNORECASE)
    if cred_match:
        val = cred_match.group(1).lower()
        if val in CREDIBILITY_LEVELS:
            cred_level = val
        text = text[: cred_match.start()] + text[cred_match.end() :]

    topic = " ".join(text.split()).strip()
    return topic, depth_key, cred_level


# =============================================================================
# === SEARCH PROVIDERS ========================================================
# =============================================================================

class SearchProvider:
    """Base class for search providers.  Subclasses must implement `search`."""

    async def search(self, query: str, num_results: int, **kwargs) -> list:
        """
        Return a list of dicts with uniform keys:
            url, title, content (snippet), raw_content (full page or None), score
        """
        raise NotImplementedError


class TavilyProvider(SearchProvider):
    """Search via the Tavily API (search + content extraction in one call)."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def search(self, query: str, num_results: int, **kwargs) -> list:
        tavily_topic = kwargs.get("topic", "general")
        payload = {
            "query": query,
            "search_depth": "advanced",
            "include_raw_content": True,
            "include_answer": False,
            "include_images": False,
            "max_results": num_results,
            "topic": tavily_topic,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        resp = requests.post(
            "https://api.tavily.com/search",
            json=payload,
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            raise RuntimeError(f"Tavily API error: {data['error']}")
        return data.get("results", [])


class SearXNGProvider(SearchProvider):
    """
    Two-stage search: SearXNG for URLs/snippets, then Crawl4AI Docker API
    for full-page markdown content.

    SearXNG must have JSON format enabled in settings.yml:
        search:
          formats:
            - json

    Crawl4AI is expected to be running as a Docker container exposing its
    REST API (default port 11235).
    """

    def __init__(self, searxng_base_url: str, crawl4ai_base_url: str, max_crawl_urls: int = 5):
        self.searxng_url = searxng_base_url.rstrip("/")
        self.crawl4ai_url = crawl4ai_base_url.rstrip("/")
        self.max_crawl_urls = max_crawl_urls

    async def search(self, query: str, num_results: int, **kwargs) -> list:
        # ── Stage 1: SearXNG search ──
        params = {
            "q": query,
            "format": "json",
            "categories": "general",
            "pageno": 1,
        }
        resp = requests.get(
            f"{self.searxng_url}/search",
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        search_data = resp.json()
        results = search_data.get("results", [])[:num_results]
        if not results:
            return []

        # ── Stage 2: Crawl4AI content extraction ──
        urls_to_crawl = [r["url"] for r in results[: self.max_crawl_urls]]
        crawl_content = {}
        if urls_to_crawl:
            try:
                crawl_payload = {
                    "urls": urls_to_crawl,
                    "browser_config": {
                        "type": "BrowserConfig",
                        "params": {"headless": True},
                    },
                    "crawler_config": {
                        "type": "CrawlerRunConfig",
                        "params": {"stream": False, "cache_mode": "bypass"},
                    },
                }
                crawl_resp = requests.post(
                    f"{self.crawl4ai_url}/crawl",
                    json=crawl_payload,
                    timeout=60,
                )
                crawl_resp.raise_for_status()
                crawl_data = crawl_resp.json()
                for cr in crawl_data.get("results", []):
                    if cr.get("success") and cr.get("markdown"):
                        crawl_content[cr["url"]] = cr["markdown"]
            except Exception:
                # Graceful degradation — fall back to SearXNG snippets only
                pass

        # ── Stage 3: Combine into uniform format ──
        combined = []
        for r in results:
            url = r.get("url", "")
            combined.append(
                {
                    "url": url,
                    "title": r.get("title", ""),
                    "content": r.get("content", ""),
                    "raw_content": crawl_content.get(url),
                    "score": r.get("score", 0),
                    "published_date": r.get("publishedDate"),
                }
            )
        return combined


def get_search_provider(valves) -> SearchProvider:
    """Factory: return the configured search provider."""
    provider_name = (valves.SEARCH_PROVIDER or "tavily").lower().strip()
    if provider_name == "searxng":
        return SearXNGProvider(
            searxng_base_url=valves.SEARXNG_BASE_URL,
            crawl4ai_base_url=valves.CRAWL4AI_BASE_URL,
            max_crawl_urls=valves.MAX_CRAWL_URLS_PER_PASS,
        )
    else:
        return TavilyProvider(api_key=valves.TAVILY_API_KEY)


# =============================================================================
# === LLM HELPER ==============================================================
# =============================================================================

async def complete(prompt: str, valves) -> str:
    """
    Call an OpenAI-compatible chat completions endpoint.

    Currently uses a single model for all phases.  Structured so each call
    site could trivially accept a per-phase model config in the future
    (e.g. valves.CLASSIFIER_MODEL_ID, valves.SYNTHESIS_MODEL_ID, etc.).
    """
    payload = {
        "model": valves.MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {valves.OPENAI_API_KEY}",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "Open-WebUI Research Assistant",
    }
    base = valves.OPENAI_API_BASE_URL.rstrip("/")
    resp = requests.post(
        f"{base}/chat/completions",
        json=payload,
        headers=headers,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices", [])
    if choices:
        return choices[0].get("message", {}).get("content", "")
    return ""


# =============================================================================
# === PROMPT BUILDERS =========================================================
# =============================================================================

def format_override_header(task_description: str) -> str:
    return (
        f"[TASK-SPECIFIC FORMATTING INSTRUCTIONS]\n"
        f"The following instructions apply ONLY to the output format of this specific task: {task_description}\n"
        f"These formatting instructions take precedence over any general formatting or length preferences, "
        f"but do NOT override your role, persona, ethical considerations, tone guidance, or any behavioral "
        f"instructions you have been given. Continue to apply all role and behavioral context as normal.\n"
        f"[END FORMATTING INSTRUCTIONS]\n\n"
    )


def build_classification_prompt(topic: str) -> str:
    return (
        'You are a query router. Classify the following user query into one of these categories:\n\n'
        '**SIMPLE** — Quick factual lookups with a single definitive answer:\n'
        '- Definitions, specific facts, dates, numbers, names\n'
        '- Current status checks (e.g., "What is the current price of Bitcoin?")\n'
        '- Simple how-to with straightforward answers\n\n'
        '**RESEARCH-FAST** — Topics needing modest research (2 passes, short report):\n'
        '- Single-faceted questions that benefit from a few sources\n'
        '- Straightforward "what is X" where X is moderately complex\n'
        '- Questions with a generally accepted answer but worth verifying\n\n'
        '**RESEARCH-STANDARD** — Topics needing solid research (4 passes, detailed report):\n'
        '- Multi-faceted topics requiring analysis from multiple angles\n'
        '- Comparative analysis (e.g., "How do X and Y differ?")\n'
        '- Topics with some debate or nuance\n'
        '- Policy or strategy questions\n\n'
        '**RESEARCH-THOROUGH** — Topics needing deep, exhaustive research (6 passes, comprehensive report):\n'
        '- Highly contested or controversial topics with strong opposing views\n'
        '- Complex policy analysis requiring jurisdictional comparison\n'
        '- Topics where the user explicitly wants comprehensive/detailed/in-depth analysis\n'
        '- Questions requiring synthesis across many domains or disciplines\n'
        '- Queries containing words like "comprehensive", "thorough", "detailed", "in-depth", "analyze", "pros and cons"\n\n'
        'Examples:\n'
        '"What year was the Eiffel Tower built?" → SIMPLE\n'
        '"What is intermittent fasting?" → RESEARCH-FAST\n'
        '"What are the health effects of intermittent fasting?" → RESEARCH-STANDARD\n'
        '"Comprehensive analysis of intermittent fasting: health effects, risks, demographic differences, and current scientific consensus" → RESEARCH-THOROUGH\n'
        '"Who is the CEO of Apple?" → SIMPLE\n'
        '"How has climate change affected global agriculture?" → RESEARCH-STANDARD\n'
        '"Compare climate change policies across G20 nations with pros, cons, and effectiveness data" → RESEARCH-THOROUGH\n'
        '"What are the best strategies for reducing technical debt?" → RESEARCH-STANDARD\n'
        '"In-depth analysis of technical debt management strategies across enterprise vs startup contexts" → RESEARCH-THOROUGH\n\n'
        'When in doubt between two adjacent levels, choose the deeper one.\n\n'
        f'Query: "{topic}"\n\n'
        'Respond with ONLY one of: SIMPLE, RESEARCH-FAST, RESEARCH-STANDARD, RESEARCH-THOROUGH'
    )


def build_simple_answer_prompt(topic: str, sources: list) -> str:
    snippets = build_snippet_block(sources, len(sources))
    return (
        f"{format_override_header('quick search answer — provide a clear, concise, well-sourced answer using the format below')}"
        f"You are a knowledgeable research assistant providing a direct, factual answer.\n\n"
        f'Question: "{topic}"\n\n'
        f"Sources:\n{snippets}\n\n"
        f"INSTRUCTIONS:\n"
        f"- Provide a clear, direct answer to the question in {SIMPLE_PRESET['reportWordTarget']} words.\n"
        f"- Use inline citations [n] for every factual claim, where n is the source number.\n"
        f"- Start with the most important/direct answer, then add supporting context.\n"
        f"- If sources disagree, briefly note the discrepancy.\n"
        f'- End with a "💡 **Key Takeaway**:" one-sentence summary.\n'
        f"- Do NOT use section headings (##) — write flowing prose paragraphs.\n"
        f"- Do NOT add a references or sources section — it will be appended automatically.\n"
        f"- Keep the tone informative and neutral."
    )


def build_evaluation_prompt(
    topic: str, sources: list, pass_count: int, max_passes: int, snippet_limit: int
) -> str:
    snippets = build_snippet_block(sources, snippet_limit)
    remaining = max_passes - pass_count
    return (
        f"{format_override_header('research gap evaluation — you MUST respond using ONLY the exact labeled format shown below, with no additional text, preamble, or explanation outside of the defined fields')}"
        f"You are acting as an academic research analyst conducting an objective, scholarly evaluation "
        f"of gathered sources. Apply any role or behavioral context you have been given while completing "
        f"this evaluation task.\n\n"
        f'Research topic: "{topic}"\n'
        f"Search passes completed: {pass_count}\n"
        f"Remaining search passes allowed: {remaining}\n\n"
        f"Information gathered so far:\n{snippets}\n\n"
        f"Your tasks:\n"
        f"1. Assess how completely the gathered information covers the research topic from an academic standpoint.\n"
        f"2. Identify any critical gaps that would significantly improve the scholarly quality of the final report.\n"
        f"3. Decide whether another search pass is warranted given the remaining passes allowed.\n\n"
        f"Rules:\n"
        f"- Only recommend continuing if there are CRITICAL gaps, not minor ones.\n"
        f"- If remaining passes is 0, you MUST respond with CONTINUE: NO.\n"
        f"- Be conservative — a thorough report can be written with good partial information.\n\n"
        f"REQUIRED OUTPUT FORMAT — do not deviate from this structure:\n"
        f"CONTINUE: [YES or NO]\n"
        f"REASON: [one sentence explanation]\n"
        f"NEXT_QUERY: [search query if CONTINUE is YES, otherwise leave blank]"
    )


def build_subtopic_prompt(topic: str, sources: list, snippet_limit: int) -> str:
    snippets = build_snippet_block(sources, snippet_limit)
    return (
        f"{format_override_header('subtopic identification — respond with ONLY a numbered list of 2-3 search queries, one per line, with no preamble or explanation')}"
        f"You are a research analyst identifying gaps in the current coverage.\n\n"
        f'Research topic: "{topic}"\n\n'
        f"Sources gathered so far:\n{snippets}\n\n"
        f"Identify 2-3 specific SUBTOPICS that emerged from the research but are not yet well-covered. "
        f"For each, provide a focused search query that would find detailed information.\n\n"
        f"Rules:\n"
        f"- Each query should target a SPECIFIC aspect, not the broad topic\n"
        f"- Prefer queries about mechanisms, data, case studies, or expert opinions\n"
        f"- Do NOT repeat queries similar to the original topic\n\n"
        f"RESPOND WITH ONLY:\n"
        f"1. [search query]\n"
        f"2. [search query]\n"
        f"3. [search query]"
    )


def build_credibility_prompt(
    topic: str, sources: list, cred_level: str, snippet_limit: int
) -> str:
    parts = []
    for s in sources[:snippet_limit]:
        try:
            from urllib.parse import urlparse
            domain = urlparse(s["url"]).hostname or s["url"]
        except Exception:
            domain = s["url"]
        date = f" ({s['publishedDate']})" if s.get("publishedDate") else ""
        parts.append(
            f"[{s['index']}] {s['title']}{date}\n"
            f"    Domain: {domain}\n"
            f"    URL: {s['url']}\n"
            f"    Content preview: {s['content'][:300]}"
        )
    source_list = "\n\n".join(parts)

    level_instructions = {
        "low": (
            "For each source, provide a one-line credibility flag: HIGH, MEDIUM, or LOW with a brief reason (5-10 words).\n\n"
            "Example format:\n"
            "[1] HIGH — established academic publisher\n"
            "[2] LOW — commercial blog with no citations"
        ),
        "medium": (
            "For each source, rate credibility as HIGH, MEDIUM, or LOW and provide:\n"
            "- Authority: Is the publisher/domain a recognized authority?\n"
            "- Evidence quality: Does it cite data, studies, or expert sources?\n"
            "- Potential bias: Any commercial, political, or ideological lean?\n"
            "(2-3 sentences per source)"
        ),
        "high": (
            "For each source, provide a detailed credibility assessment:\n"
            "- CREDIBILITY RATING: HIGH / MEDIUM / LOW\n"
            "- AUTHORITY: Publisher reputation, domain expertise, author credentials if visible\n"
            "- METHODOLOGY: Does the source cite primary data, peer-reviewed studies, official statistics?\n"
            "- RECENCY: How current is the information relative to the topic?\n"
            "- BIAS INDICATORS: Commercial interests, political lean, advocacy position, funding sources\n"
            "- CROSS-REFERENCE: Do other sources in this set corroborate or contradict this source?\n"
            "(4-6 sentences per source)\n\n"
            "After rating all sources, add:\n"
            "OVERALL ASSESSMENT: Summary of source quality across the research set.\n"
            "RELIABILITY WARNINGS: Any specific claims that rest on LOW-credibility sources only."
        ),
    }

    return (
        f"{format_override_header('source credibility assessment — rate each source using the format specified below')}"
        f"You are an expert research librarian and fact-checker performing source credibility analysis.\n\n"
        f'Research topic: "{topic}"\n\n'
        f"Sources to evaluate:\n{source_list}\n\n"
        f"{level_instructions.get(cred_level, level_instructions['medium'])}"
    )


def build_controversy_prompt(topic: str, sources: list, snippet_limit: int) -> str:
    snippets = build_snippet_block(sources, snippet_limit)
    return (
        f"{format_override_header('controversy and conflict analysis — you MUST structure your response using the exact labeled sections defined below. Section labels must appear verbatim. Content within each section may be as detailed as your role and behavioral context requires.')}"
        f"You are acting as an academic research analyst performing an objective conflict analysis. "
        f"Apply any role, persona, or behavioral context you have been given while completing this analysis.\n\n"
        f'Research topic: "{topic}"\n\n'
        f"Sources gathered:\n{snippets}\n\n"
        f"Analyze the following dimensions and respond using these EXACT section labels:\n\n"
        f"CONFLICTING CLAIMS:\n"
        f"[Identify specific factual claims that directly contradict each other across sources.]\n\n"
        f"METHODOLOGICAL DISPUTES:\n"
        f"[Describe any disagreements about how research on this topic is conducted or measured.]\n\n"
        f"EXPERT DISAGREEMENT:\n"
        f"[Describe where subject matter experts or authoritative sources diverge in their conclusions.]\n\n"
        f"JURISDICTIONAL VARIANCE:\n"
        f"[Describe aspects of this topic treated differently across countries, states, or legal systems.]\n\n"
        f"BIAS INDICATORS:\n"
        f"[Note any sources that appear to have a strong ideological, commercial, or political bias.]\n\n"
        f"CONSENSUS AREAS:\n"
        f"[Describe what most sources agree on despite other controversies.]"
    )


def build_report_prompt(
    topic: str,
    sources: list,
    controversy_analysis: str,
    credibility_analysis: str,
    snippet_limit: int,
    preset: dict,
    depth_key: str,
    cred_level: str,
) -> str:
    snippets = build_snippet_block(sources, snippet_limit)
    source_list = "\n".join(
        f"[{s['index']}] {s['title']}{' (' + s['publishedDate'] + ')' if s.get('publishedDate') else ''} - {s['url']}"
        for s in sources
    )
    cred_section = ""
    if credibility_analysis:
        cred_section = (
            f"\nSource credibility analysis (pre-processed — use this to weight your claims; "
            f"prioritize HIGH-credibility sources and flag claims that rely solely on LOW-credibility sources):\n"
            f"{credibility_analysis}\n"
        )
    depth_requirements = (
        f"\nDEPTH & LENGTH REQUIREMENTS (research depth: {depth_key.upper()}):\n"
        f"- Target report length: {preset['reportWordTarget']} words.\n"
        f"- The Overview MUST be at least 2 substantive paragraphs.\n"
        f"- Key Findings MUST include at least {preset['minKeyFindings']} findings, each with 2-3 sentences of supporting context and citations.\n"
        f"- Detailed Analysis MUST be at least {preset['minDetailedParagraphs']} paragraphs with specific evidence.\n"
        f"- Each analytical section SHOULD include at least one markdown table comparing data, perspectives, or frameworks where appropriate.\n"
        f"- The Conclusion MUST include confidence ratings (HIGH / MEDIUM / LOW) for each major finding based on the strength and agreement of sources.\n"
        f"- Prioritize claims backed by HIGH-credibility sources. Flag claims that rely solely on LOW-credibility sources with ⚠️.\n"
        f"- The Key Data & Statistics section MUST contain at least one table summarizing quantitative claims from sources."
    )
    return (
        f"{format_override_header('final research report — you MUST use the exact Markdown section headings listed below, in the order listed, using the specified emoji. Inline citation format must be [n] where n is the source number. Do NOT add a citations, references, or methodology section — they will be appended automatically. All other aspects of tone, depth, and presentation should reflect your assigned role and behavioral context.')}"
        f"You are acting as an expert research writer producing a comprehensive, balanced report. "
        f"Apply your assigned role, persona, tone, and any behavioral guidance you have received. "
        f"Where your role provides relevant domain expertise or perspective, apply it fully.\n\n"
        f'Research topic: "{topic}"\n\n'
        f"Gathered research data:\n{snippets}\n\n"
        f"Controversy and conflict analysis (pre-processed — incorporate this into the relevant sections):\n"
        f"{controversy_analysis}\n"
        f"{cred_section}"
        f"Available sources for citation:\n{source_list}\n\n"
        f"REQUIRED REPORT STRUCTURE — use these exact headings in this order:\n"
        f"## 📖 Overview\n"
        f"## 📋 Executive Summary\n"
        f"## 🔑 Key Findings\n"
        f"## 📊 Key Data & Statistics\n"
        f"## 🔬 Detailed Analysis\n"
        f"## ⚖️ Controversies & Conflicting Perspectives\n"
        f"## 🌍 Jurisdictional & Contextual Variance\n"
        f"## ⚠️ Important Considerations\n"
        f"## 🎯 Conclusion\n"
        f"{depth_requirements}\n\n"
        f"REQUIRED FORMATTING RULES:\n"
        f"- Emoji usage: 🔑 Key points | ⚠️ Warnings | 📊 Data/stats | 💡 Insights | 🎯 Conclusions | ⚖️ Debate | 🌍 Jurisdictional | 📌 Notes\n"
        f'- CITATIONS: Cite sources inline using [n] where n is the source number. Example: "Studies show a 40% increase [3] while others disagree [7]."\n'
        f"- Every factual claim MUST have at least one [n] citation.\n"
        f"- Do NOT embed URLs inline — the bibliography with full URLs is appended automatically.\n"
        f"- Where sources conflict, present ALL perspectives fairly with explicit attribution.\n"
        f"- Do NOT editorialize or take sides on contested claims.\n"
        f"- Do NOT add a citations, references, or methodology section — they will be appended automatically.\n"
        f"- Include markdown tables where comparing data points, frameworks, or jurisdictional differences."
    )


# =============================================================================
# === PARSERS =================================================================
# =============================================================================

def parse_classification(response_text: str) -> str:
    if not response_text:
        return "RESEARCH-STANDARD"
    cleaned = response_text.strip().upper()

    # Check for the specific depth-tagged research categories first
    if "RESEARCH-THOROUGH" in cleaned:
        return "RESEARCH-THOROUGH"
    if "RESEARCH-STANDARD" in cleaned:
        return "RESEARCH-STANDARD"
    if "RESEARCH-FAST" in cleaned:
        return "RESEARCH-FAST"

    # Fall back to simple detection
    if cleaned == "SIMPLE" or cleaned.startswith("SIMPLE"):
        return "SIMPLE"

    # Legacy/ambiguous — check for plain RESEARCH
    if "RESEARCH" in cleaned:
        return "RESEARCH-STANDARD"
    if re.search(r"\bSIMPLE\b", cleaned):
        return "SIMPLE"

    return "RESEARCH-STANDARD"


def extract_fallback_query(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"""["']([^"']{10,100})["']""", text)
    return m.group(1).strip() if m else None


def parse_evaluation(response_text: str) -> dict:
    if not response_text:
        return {"continueSearch": False, "reason": "No response", "nextQuery": None}

    continue_match = re.search(r"CONTINUE:\s*(YES|NO)", response_text, re.IGNORECASE)
    reason_match = re.search(r"REASON:\s*(.+?)(?:\n|$)", response_text, re.IGNORECASE)
    query_match = re.search(r"NEXT_QUERY:\s*(.+?)(?:\n|$)", response_text, re.IGNORECASE)

    looks_yes = bool(re.search(r"\b(continue|additional|another|more|missing|gap|insufficient)\b", response_text, re.IGNORECASE))
    looks_no = bool(re.search(r"\b(sufficient|complete|enough|comprehensive|no further|ready)\b", response_text, re.IGNORECASE))

    if continue_match:
        continue_search = continue_match.group(1).upper() == "YES"
    else:
        continue_search = looks_yes and not looks_no

    reason = (
        reason_match.group(1).strip()
        if reason_match
        else (response_text[:120] + "..." if len(response_text) > 20 else "No reason provided")
    )

    next_query = None
    if query_match and query_match.group(1).strip():
        next_query = query_match.group(1).strip()
    else:
        next_query = extract_fallback_query(response_text)

    return {"continueSearch": continue_search, "reason": reason, "nextQuery": next_query}


def parse_subtopics(response_text: str) -> list:
    if not response_text:
        return []
    lines = [l.strip() for l in response_text.split("\n") if l.strip()]
    queries = []
    for line in lines:
        cleaned = re.sub(r"^\d+[.):\-]\s*", "", line).strip()
        if 10 <= len(cleaned) <= 150:
            queries.append(cleaned)
    return queries[:3]


# =============================================================================
# === REPORT ASSEMBLY =========================================================
# =============================================================================

def assemble_final_report(
    report_body: str,
    sources: list,
    topic: str,
    search_log: list,
    pass_count: int,
    depth_key: str,
    cred_level: str,
    credibility_analysis: str,
    search_provider_name: str,
) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")

    def _label(s):
        if s["pass"] == "counter":
            return "Counter"
        if s["pass"] == "subtopic":
            return "Subtopic"
        return f"Pass {s['pass']}"

    search_summary = " → ".join(f'{_label(s)}: "{s["query"]}" ({s["results"]} results)' for s in search_log)

    header = (
        f"---\n\n"
        f"> 🔬 **Research Assistant** | Open WebUI Pipeline\n"
        f"> 📅 *Generated: {timestamp}*\n"
        f"> 🔍 *Topic: {topic}*\n"
        f"> 📋 *Depth: {depth_key.upper()} | Credibility: {cred_level.upper()}*\n"
        f"> 📊 *{len(search_log)} search operations | {len(sources)} unique sources*\n"
        f"> 🗺️ *Search path: {search_summary}*\n"
        f"> 🔎 *Search provider: {search_provider_name}*\n\n"
        f"---\n\n"
    )

    main_passes = sum(1 for s in search_log if isinstance(s["pass"], int))
    counter_passes = sum(1 for s in search_log if s["pass"] == "counter")
    subtopic_passes = sum(1 for s in search_log if s["pass"] == "subtopic")

    method_parts = [
        "## 🔬 Methodology\n",
        "This report was produced using automated multi-pass research:\n",
        f"- **{main_passes}** primary search passes with LLM-driven gap analysis",
    ]
    if counter_passes:
        method_parts.append(f"- **{counter_passes}** counter-perspective search to capture opposing viewpoints")
    if subtopic_passes:
        method_parts.append(f"- **{subtopic_passes}** subtopic deep-dive searches for granular coverage")
    method_parts.append(f"- **{len(sources)}** unique sources deduplicated and analyzed")
    if cred_level != "off":
        method_parts.append(f"- **Source credibility scoring** at {cred_level.upper()} level")
    method_parts.append("- **Controversy & conflict analysis** across all sources")
    method_parts.append(f"- **Research depth:** {depth_key.upper()}")
    method_parts.extend(["", "---", ""])

    cred_block = ""
    if cred_level != "off" and credibility_analysis and credibility_analysis.strip():
        cred_block = (
            f"---\n\n"
            f"## 🏅 Source Credibility Assessment\n\n"
            f"{credibility_analysis.strip()}\n\n"
        )

    citation_block = ""
    if sources:
        citation_lines = ["---\n", "## 📚 Citations\n"]
        for s in sources:
            date = f" *({s['publishedDate']})*" if s.get("publishedDate") else ""
            chars = f" — {s['fullContentLength']:,} chars indexed" if s.get("fullContentLength") else ""
            citation_lines.append(f"- [{s['index']}] [{s['title']}]({s['url']}){date}{chars}")
        citation_lines.append("")
        citation_block = "\n".join(citation_lines)

    footer = (
        "\n---\n\n"
        "> 🔬 *This report was produced by the **Research Assistant** pipeline using multi-pass search,*\n"
        "> *LLM-driven gap analysis, automated controversy detection, and source credibility scoring.*\n"
        "> *It should be independently verified before use in critical decisions.*\n"
    )

    return f"{header}{chr(10).join(method_parts)}\n{report_body}\n\n{cred_block}{citation_block}{footer}"


def assemble_simple_report(answer_body: str, sources: list, topic: str, search_provider_name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")

    header = (
        f"---\n\n"
        f"> 🔍 **Research Assistant — Quick Search** | Open WebUI Pipeline\n"
        f"> 📅 *Generated: {timestamp}*\n"
        f"> 🔍 *Topic: {topic}*\n"
        f"> 📊 *{len(sources)} sources consulted*\n"
        f"> 🔎 *Search provider: {search_provider_name}*\n\n"
        f"---\n\n"
    )

    citation_block = ""
    if sources:
        citation_lines = ["\n---\n", "## 📚 Sources\n"]
        for s in sources:
            date = f" *({s['publishedDate']})*" if s.get("publishedDate") else ""
            citation_lines.append(f"- [{s['index']}] [{s['title']}]({s['url']}){date}")
        citation_lines.append("")
        citation_block = "\n".join(citation_lines)

    footer = (
        "\n---\n\n"
        "> 🔍 *Quick search by the **Research Assistant** pipeline.*\n"
        "> *For more thorough analysis, use depth:fast, depth:standard, or depth:thorough in your query.*\n"
    )

    return f"{header}{answer_body}\n{citation_block}{footer}"


# =============================================================================
# === PIPELINE ================================================================
# =============================================================================

class Pipeline:
    class Valves(BaseModel):
        # ── LLM Configuration ──
        # Currently a single model for all phases. Future: split into
        # CLASSIFIER_MODEL_ID, EVALUATOR_MODEL_ID, SYNTHESIS_MODEL_ID,
        # REPORT_MODEL_ID for per-phase model routing.
        OPENAI_API_BASE_URL: str = Field(
            default="http://localhost:11434/v1",
            description="Base URL for the OpenAI-compatible API (e.g. Ollama, OpenAI, vLLM)",
        )
        OPENAI_API_KEY: str = Field(
            default="",
            description="API key for the OpenAI-compatible endpoint (leave blank for Ollama)",
        )
        MODEL_ID: str = Field(
            default="llama3.1:8b",
            description="Model ID for all LLM calls (classification, evaluation, synthesis, report)",
        )

        # ── Search Provider ──
        SEARCH_PROVIDER: str = Field(
            default="tavily",
            description="Search provider: 'tavily' or 'searxng'",
        )
        TAVILY_API_KEY: str = Field(
            default="",
            description="Tavily API key (required if SEARCH_PROVIDER is 'tavily')",
        )
        SEARXNG_BASE_URL: str = Field(
            default="http://localhost:8888",
            description="SearXNG instance base URL (required if SEARCH_PROVIDER is 'searxng')",
        )
        CRAWL4AI_BASE_URL: str = Field(
            default="http://localhost:11235",
            description="Crawl4AI Docker API base URL (used with SearXNG for full-page extraction)",
        )
        MAX_CRAWL_URLS_PER_PASS: int = Field(
            default=5,
            description="Maximum URLs to crawl via Crawl4AI per search pass (controls latency)",
        )

        # ── Defaults ──
        DEFAULT_DEPTH: str = Field(
            default="auto",
            description="Default research depth: auto, fast, standard, or thorough",
        )
        DEFAULT_CREDIBILITY: str = Field(
            default="medium",
            description="Default source credibility level: off, low, medium, or high",
        )

    def __init__(self):
        self.name = "Research Assistant"
        self.valves = self.Valves()

    async def on_startup(self):
        pass

    async def on_shutdown(self):
        pass

    async def on_valves_updated(self):
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict,
        __event_emitter__=None,
    ) -> Union[str, Generator, Iterator]:
        """
        Main entry point. Status updates are sent through __event_emitter__ when
        available (renders as a collapsible panel in Open-WebUI). Falls back to
        yielding inline text so messages are always visible regardless of version.
        Only the final report is yielded as substantive chat content.
        """
        import logging
        import asyncio

        def status(msg: str, done: bool = False):
            """
            Generator: tries __event_emitter__ first (collapsible panel), falls
            back to yielding the message as inline text if the emitter is absent
            or raises. This guarantees status messages are always visible.
            """
            emitted = False
            if __event_emitter__:
                try:
                    loop = asyncio.new_event_loop()
                    try:
                        loop.run_until_complete(__event_emitter__({
                            "type": "status",
                            "data": {"description": msg, "done": done},
                        }))
                        emitted = True
                    finally:
                        loop.close()
                except Exception:
                    pass
            if not emitted:
                yield f"\n{msg}\n"

        # Skip Open-WebUI internal tasks (title generation, follow-ups, tags, etc.)
        user_lower = user_message.lower()
        if ("broad tags categorizing" in user_lower) \
                or ("create a concise" in user_lower) \
                or ("suggest 3-5 relevant follow-up" in user_lower) \
                or ("analyze the chat history to determine" in user_lower) \
                or ("respond to the user query using" in user_lower) \
                or user_message.strip().startswith("### Task:"):
            yield "(internal task skipped)"
            return

        yield from status("🔍 Parsing query...")

        # --- Parse depth/credibility from the RAW message BEFORE extraction ---
        # This ensures depth:thorough / credibility:high are captured even if
        # query extraction later truncates or reformats the message.
        _, raw_depth, raw_cred = parse_input(
            user_message,
            self.valves.DEFAULT_DEPTH,
            self.valves.DEFAULT_CREDIBILITY,
        )

        # Extract the actual user query
        # Open-WebUI formats messages with system prompts embedded, ending with "Query: <actual query>"
        actual_user_query = user_message
        extraction_method = "direct"

        # Try to extract from "Query:" marker (Open-WebUI format)
        if "Query:" in user_message:
            parts = user_message.rsplit("Query:", 1)
            if len(parts) > 1:
                extracted = parts[1].strip().strip('"').strip()
                if extracted:
                    actual_user_query = extracted
                    extraction_method = "Query: marker"
                    logging.info(f"EXTRACTED from Query: {actual_user_query[:100]}")
        # Also try "USER:" marker
        elif "USER:" in user_message and "SYSTEM:" in user_message:
            parts = user_message.rsplit("USER:", 1)
            if len(parts) > 1:
                extracted = parts[1].strip()
                # Clean up trailing markers
                for marker in ["</chat_history>", "ASSISTANT:", "\n\n"]:
                    if marker in extracted:
                        extracted = extracted.split(marker)[0].strip()
                if extracted:
                    actual_user_query = extracted
                    extraction_method = "USER: marker"
                    logging.info(f"EXTRACTED from USER: {actual_user_query[:100]}")
        # If message is short and doesn't look like a system prompt, use it directly
        elif len(user_message) < 1000 and not user_message.startswith("History:"):
            actual_user_query = user_message
            extraction_method = "short message"
            logging.info(f"USING SHORT MESSAGE: {actual_user_query[:100]}")
        else:
            extraction_method = "full message (no marker found)"
            logging.info(f"NO EXTRACTION - using full message len={len(user_message)}")

        # Also parse the extracted query (in case depth: keyword is in the extracted part)
        _, extracted_depth, extracted_cred = parse_input(
            actual_user_query,
            self.valves.DEFAULT_DEPTH,
            self.valves.DEFAULT_CREDIBILITY,
        )

        # Prefer raw-message parse (captures keywords before extraction strips them),
        # fall back to extracted-query parse
        default_depth = self.valves.DEFAULT_DEPTH
        default_cred = self.valves.DEFAULT_CREDIBILITY
        final_depth = raw_depth if raw_depth != default_depth else extracted_depth
        final_cred = raw_cred if raw_cred != default_cred else extracted_cred

        # Strip any depth:/credibility: from the extracted query to avoid duplication
        clean_query = re.sub(r"\bdepth:\w+\b", "", actual_user_query, flags=re.IGNORECASE)
        clean_query = re.sub(r"\bcredibility:\w+\b", "", clean_query, flags=re.IGNORECASE)
        clean_query = " ".join(clean_query.split()).strip()

        # Re-inject resolved depth/credibility so _run_research picks them up
        prefix = ""
        if final_depth != default_depth:
            prefix += f"depth:{final_depth} "
        if final_cred != default_cred:
            prefix += f"credibility:{final_cred} "

        final_message = (prefix + clean_query).strip()
        logging.info(f"PIPELINE INPUT: depth={final_depth}, cred={final_cred}, query={clean_query[:100]}")

        # Show the user what we parsed
        depth_display = final_depth.upper() if final_depth != default_depth else f"{default_depth.upper()} (default)"
        cred_display = final_cred.upper() if final_cred != default_cred else f"{default_cred.upper()} (default)"
        provider_name = (self.valves.SEARCH_PROVIDER or "tavily").lower().strip()
        yield from status(
            f"📋 Depth: {depth_display} | Credibility: {cred_display} | "
            f"Provider: {provider_name} | Extraction: {extraction_method}"
        )

        if not clean_query:
            yield from status("❌ No research topic found after parsing.", done=True)
            yield "\n❌ No research topic found after parsing. Please enter a topic to research.\n"
            return

        yield from status(f'🎯 Topic: "{clean_query}"')

        # Delegate to the research pipeline generator
        yield from self._run_research(final_message, body, status)

    def _run_research(self, user_message: str, body: dict, status_fn=None) -> Generator:
        """
        Synchronous generator that orchestrates the full research pipeline.
        Uses synchronous wrappers around the async helpers since Open WebUI
        consumes generators synchronously.

        status_fn(msg, done=False) — generator function provided by pipe() that
        tries __event_emitter__ and falls back to yielding inline text. Defaults
        to inline-text-only when called directly (e.g. in tests).
        """
        import asyncio

        # Default: yield inline text (used when called outside of pipe(), e.g. tests)
        if status_fn is None:
            def status_fn(msg: str, done: bool = False):
                yield f"\n{msg}\n"

        # ── Parse input ──
        topic, depth_key, cred_level = parse_input(
            user_message,
            self.valves.DEFAULT_DEPTH,
            self.valves.DEFAULT_CREDIBILITY,
        )
        if not topic:
            yield from status_fn("❌ No research topic provided.", done=True)
            yield "❌ No research topic provided. Please enter a topic to research."
            return

        # ── Validate search provider config ──
        provider_name = (self.valves.SEARCH_PROVIDER or "tavily").lower().strip()
        if provider_name == "tavily" and not self.valves.TAVILY_API_KEY:
            yield from status_fn("❌ Tavily API key not configured.", done=True)
            yield "❌ Tavily API key not configured. Set it in the pipeline Valves, or switch SEARCH_PROVIDER to 'searxng'."
            return

        search_provider = get_search_provider(self.valves)

        # Helper to run async functions from this sync generator
        def run_async(coro):
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        try:
            # ──────────────────────────────────────────
            # STAGE 1: Auto-routing classification
            # ──────────────────────────────────────────
            if depth_key == "auto":
                yield from status_fn("🔀 Auto-routing: classifying query complexity...")
                class_prompt = build_classification_prompt(topic)
                class_response = run_async(complete(class_prompt, self.valves))
                classification = parse_classification(class_response)

                if classification == "SIMPLE":
                    yield from status_fn("🔀 Auto-routing: SIMPLE — quick single-pass search")
                    tavily_topic = detect_tavily_topic(topic)
                    yield from status_fn(f'🔍 Quick search: "{topic}" (category: {tavily_topic})')

                    simple_results = run_async(
                        search_provider.search(topic, SIMPLE_PRESET["resultsPerPass"], topic=tavily_topic)
                    )
                    if not simple_results:
                        yield from status_fn("❌ Quick search returned no results.", done=True)
                        yield "\n❌ Quick search returned no results. Try rephrasing your query or using depth:fast.\n"
                        return

                    simple_sources = extract_sources(simple_results, SIMPLE_PRESET["maxCharsPerSnippet"])
                    yield from status_fn(f"✅ Found {len(simple_sources)} sources. Synthesizing answer...")

                    answer_prompt = build_simple_answer_prompt(topic, simple_sources)
                    answer_body = run_async(complete(answer_prompt, self.valves))
                    if not answer_body or not answer_body.strip():
                        yield from status_fn("❌ Answer synthesis returned empty.", done=True)
                        yield "\n❌ Answer synthesis returned empty. Try again or use depth:fast.\n"
                        return

                    simple_report = assemble_simple_report(answer_body, simple_sources, topic, provider_name)
                    yield from status_fn(f"✅ Quick search complete! ({len(simple_report)} chars)", done=True)
                    yield f"\n{simple_report}"
                    return

                # Map classification to depth preset
                depth_map = {
                    "RESEARCH-FAST": "fast",
                    "RESEARCH-STANDARD": "standard",
                    "RESEARCH-THOROUGH": "thorough",
                }
                depth_key = depth_map.get(classification, "standard")
                yield from status_fn(f"🔀 Auto-routing: {classification} → depth {depth_key.upper()} — proceeding with deep research")

            # ──────────────────────────────────────────
            # STAGE 2: Resolve preset
            # ──────────────────────────────────────────
            preset = DEPTH_PRESETS[depth_key]
            _max_passes = preset["maxPasses"]
            _results_per_pass = preset["resultsPerPass"]
            _eval_snippet_limit = preset["evalSnippetLimit"]
            _controversy_snippet_limit = preset["controversySnippetLimit"]
            _report_snippet_limit = preset["reportSnippetLimit"]
            _max_chars_per_snippet = preset["maxCharsPerSnippet"]
            _report_max_chars_per_snippet = preset["reportMaxCharsPerSnippet"]

            tavily_topic = detect_tavily_topic(topic)

            yield from status_fn(f'🔍 Starting deep research on: "{topic}"')
            yield from status_fn(
                f"📋 Depth: {depth_key.upper()} | Credibility: {cred_level.upper()} | "
                f"Search category: {tavily_topic} | Provider: {provider_name}"
            )
            yield from status_fn(
                f"⚙️ Settings: max {_max_passes} passes | {_results_per_pass} results/pass | "
                f"eval={_eval_snippet_limit} | controversy={_controversy_snippet_limit} | "
                f"report={_report_snippet_limit} snippets | "
                f"{_max_chars_per_snippet}/{_report_max_chars_per_snippet} chars/snippet"
            )

            est_tokens = _report_snippet_limit * _report_max_chars_per_snippet / 4
            if est_tokens > 20000:
                yield from status_fn(f"⚠️ Estimated report-prompt token load: ~{round(est_tokens):,}. If your model has a small context window, consider a lower depth.")

            all_sources: list = []
            current_query = topic
            pass_count = 0
            continue_research = True
            search_log: list = []

            # ──────────────────────────────────────────
            # STAGE 3: Dynamic search loop
            # ──────────────────────────────────────────
            while continue_research and pass_count < _max_passes:
                pass_count += 1
                yield from status_fn(f'📡 Search pass {pass_count}/{_max_passes}: "{current_query}"')

                search_results = run_async(
                    search_provider.search(current_query, _results_per_pass, topic=tavily_topic)
                )

                if not search_results:
                    yield from status_fn(f"⚠️ No results returned on pass {pass_count}. Stopping search loop.")
                    break

                quality = assess_content_quality(search_results)
                yield from status_fn(f"📊 Content quality: {quality['summary']}")

                sources = extract_sources(search_results, _max_chars_per_snippet)
                all_sources = deduplicate_sources(all_sources + sources)

                search_log.append({"pass": pass_count, "query": current_query, "results": len(search_results)})
                yield from status_fn(f"✅ Pass {pass_count} complete. Total unique sources: {len(all_sources)}")

                yield from status_fn(f"🧠 Evaluating research completeness after pass {pass_count}...")
                eval_prompt = build_evaluation_prompt(
                    topic, all_sources, pass_count, _max_passes, _eval_snippet_limit
                )
                eval_response = run_async(complete(eval_prompt, self.valves))
                decision = parse_evaluation(eval_response)

                if decision["continueSearch"]:
                    yield from status_fn(f'🤔 LLM Decision: Continue — "{decision["nextQuery"]}"')
                else:
                    yield from status_fn("🤔 LLM Decision: Sufficient — ready to write report")
                yield from status_fn(f'📋 Reason: {decision["reason"]}')

                if decision["continueSearch"] and decision["nextQuery"]:
                    current_query = decision["nextQuery"]
                else:
                    continue_research = False

            if pass_count >= _max_passes:
                yield from status_fn(f"⚠️ Reached maximum search passes ({_max_passes}). Proceeding to analysis.")

            # ──────────────────────────────────────────
            # STAGE 4: Counter-perspective search
            # ──────────────────────────────────────────
            if preset["enableCounterPerspective"]:
                yield from status_fn("🔄 Running counter-perspective search...")
                counter_query = f"{topic} criticism OR controversy OR problems OR limitations"
                counter_results = run_async(
                    search_provider.search(counter_query, _results_per_pass, topic=tavily_topic)
                )
                if counter_results:
                    counter_sources = extract_sources(counter_results, _max_chars_per_snippet)
                    all_sources = deduplicate_sources(all_sources + counter_sources)
                    search_log.append({"pass": "counter", "query": counter_query, "results": len(counter_results)})
                    yield from status_fn(f"✅ Counter-perspective: +{len(counter_results)} results. Total unique: {len(all_sources)}")
                else:
                    yield from status_fn("⚠️ Counter-perspective search returned no results.")

            # ──────────────────────────────────────────
            # STAGE 5: Subtopic deep-dive
            # ──────────────────────────────────────────
            if preset["enableSubtopicDive"]:
                yield from status_fn("🔬 Identifying key subtopics for deep-dive...")
                subtopic_prompt = build_subtopic_prompt(topic, all_sources, _eval_snippet_limit)
                subtopic_response = run_async(complete(subtopic_prompt, self.valves))
                subtopics = parse_subtopics(subtopic_response)

                if subtopics:
                    yield from status_fn(f"🔬 Deep-diving into {len(subtopics)} subtopics: {', '.join(subtopics)}")
                    for sub in subtopics:
                        yield from status_fn(f'  📡 Subtopic search: "{sub}"')
                        sub_results = run_async(
                            search_provider.search(sub, _results_per_pass, topic=tavily_topic)
                        )
                        if sub_results:
                            sub_sources = extract_sources(sub_results, _max_chars_per_snippet)
                            all_sources = deduplicate_sources(all_sources + sub_sources)
                            search_log.append({"pass": "subtopic", "query": sub, "results": len(sub_results)})
                            yield from status_fn(f"  ✅ +{len(sub_results)} results. Total unique: {len(all_sources)}")
                else:
                    yield from status_fn("⚠️ No additional subtopics identified.")

            # ──────────────────────────────────────────
            # STAGE 6: Source credibility scoring
            # ──────────────────────────────────────────
            credibility_analysis = ""
            if cred_level != "off":
                yield from status_fn(f"🏅 Scoring source credibility (level: {cred_level})...")
                cred_prompt = build_credibility_prompt(topic, all_sources, cred_level, _report_snippet_limit)
                credibility_analysis = run_async(complete(cred_prompt, self.valves))
                yield from status_fn("✅ Source credibility analysis complete.")

            # ──────────────────────────────────────────
            # STAGE 7: Controversy analysis
            # ──────────────────────────────────────────
            yield from status_fn("⚖️ Analyzing sources for conflicting information and controversy...")
            controversy_prompt = build_controversy_prompt(topic, all_sources, _controversy_snippet_limit)
            controversy_analysis = run_async(complete(controversy_prompt, self.valves))
            yield from status_fn("✅ Controversy analysis complete.")

            # ──────────────────────────────────────────
            # STAGE 8: Report synthesis
            # ──────────────────────────────────────────
            yield from status_fn(f"📝 Synthesizing final report from {len(all_sources)} unique sources across {len(search_log)} search operations...")

            report_prompt = build_report_prompt(
                topic, all_sources, controversy_analysis, credibility_analysis,
                _report_snippet_limit, preset, depth_key, cred_level,
            )
            report_body = run_async(complete(report_prompt, self.valves))
            if not report_body or not report_body.strip():
                yield from status_fn("❌ Report generation returned empty output.", done=True)
                yield "\n❌ Deep Research: report generation returned empty output. Try reducing depth or using a model with a larger context window.\n"
                return

            yield from status_fn(f"✅ Report body received ({len(report_body)} chars). Assembling final output...")

            final_report = assemble_final_report(
                report_body, all_sources, topic, search_log, pass_count,
                depth_key, cred_level, credibility_analysis, provider_name,
            )

            yield from status_fn(f"✅ Deep research complete! {len(all_sources)} sources | {len(search_log)} searches | {len(final_report):,} chars", done=True)

            # Yield the full report as chat output
            yield f"\n{final_report}"

        except Exception as e:
            yield from status_fn(f"❌ Error: {e}", done=True)
            yield f"\n❌ Deep Research encountered an error: {e}\n"
