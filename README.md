# Research Assistant Pipeline v1.0

A multi-pass deep research pipeline for [Open WebUI](https://github.com/open-webui/open-webui) that performs automated web research with LLM-driven gap analysis, controversy detection, source credibility scoring, and structured report generation.

## Features

### Dual Search Provider Support

| Provider | How it works | Best for |
|---|---|---|
| **Tavily** | Single API call returns URLs + full-page content | Simplicity, no self-hosting |
| **SearXNG + Crawl4AI** | Two-stage: SearXNG for URLs/snippets, then Crawl4AI for full-page markdown extraction | Privacy, self-hosted setups |

Switch between providers via the `SEARCH_PROVIDER` valve -- no code changes needed.

### Intelligent Auto-Routing

When depth is set to `auto` (the default), the pipeline classifies each query before deciding how to handle it:

- **SIMPLE** queries (e.g. "What is the capital of France?") get a single-pass quick search with a concise 200-400 word answer
- **RESEARCH** queries (e.g. "What are the health effects of intermittent fasting?") get multi-pass deep research with a full report

### Interactive Depth Selection (Two-Turn Prompt)

When a query is classified as RESEARCH and `ASK_DEPTH` is enabled (default), the pipeline pauses and presents a settings menu:

```
This looks like a research query! Choose your settings:

Depth:
1. Fast -- 2 passes, concise report (800-1,200 words)
2. Standard -- 4 passes, balanced report (1,500-2,500 words)
3. Thorough -- 6 passes, comprehensive report (2,500-4,000 words)

Credibility (optional -- append to your choice):
- credibility:off, credibility:low, credibility:medium (default), credibility:high

Reply with a number (e.g. 2) or name (e.g. standard credibility:high)
```

The user replies with their choice and research proceeds with those settings. Bypassed when:
- The user specifies `depth:X` explicitly in their query
- The query is classified as SIMPLE
- `ASK_DEPTH` is set to `False` in valves (defaults to fast silently)

### Three Research Depth Presets

| Preset | Passes | Results/Pass | Report Length | Counter-Perspective | Subtopic Dive |
|---|---|---|---|---|---|
| Fast | 2 | 5 | 800-1,200 words | No | No |
| Standard | 4 | 5 | 1,500-2,500 words | Yes | No |
| Thorough | 6 | 7 | 2,500-4,000 words | Yes | Yes |

### Source Credibility Scoring

Four levels of source credibility analysis, configurable per-query:

| Level | Detail |
|---|---|
| off | No credibility analysis |
| low | One-line HIGH/MEDIUM/LOW flag per source |
| medium | Authority, evidence quality, and bias assessment per source (default) |
| high | Full assessment including methodology, recency, cross-referencing, and overall reliability warnings |

### 8-Stage Research Pipeline

1. **Query Classification** -- SIMPLE vs RESEARCH routing
2. **Interactive Settings** -- depth/credibility selection (RESEARCH only)
3. **Dynamic Search Loop** -- multi-pass search with LLM-driven gap analysis deciding when to continue
4. **Counter-Perspective Search** -- dedicated search for criticism, controversy, and limitations (Standard+)
5. **Subtopic Deep-Dive** -- LLM identifies under-covered subtopics for additional searches (Thorough only)
6. **Source Credibility Scoring** -- per-source authority, bias, and reliability assessment
7. **Controversy Analysis** -- conflicting claims, expert disagreement, jurisdictional variance, bias indicators
8. **Report Synthesis** -- structured report with inline citations, data tables, and confidence ratings

### Live Status Messages

The pipeline streams progress updates throughout all stages via Python generators:

```
Parsing query...
Topic: "health effects of intermittent fasting"
Auto-routing: classifying query...
Search pass 1/4: "health effects of intermittent fasting"
Content quality: 4/5 rich results | avg 2340 chars | 80% rich
Evaluating research completeness after pass 1...
LLM Decision: Continue -- "intermittent fasting long-term studies"
...
Synthesizing final report from 23 unique sources...
Deep research complete!
```

### Structured Report Output

Every research report includes:

- **Header** with metadata (timestamp, topic, depth, provider, source count, search path)
- **Methodology** section documenting the research process
- **9 required sections**: Overview, Executive Summary, Key Findings, Key Data and Statistics, Detailed Analysis, Controversies and Conflicting Perspectives, Jurisdictional Variance, Important Considerations, Conclusion
- **Inline citations** `[n]` linking to numbered sources
- **Source Credibility Assessment** section (when enabled)
- **Full bibliography** with clickable links, dates, and indexed content lengths
- **Confidence ratings** (HIGH / MEDIUM / LOW) for major findings
- **Markdown tables** for data comparisons

## Installation

1. Copy `research_assistant_pipeline.py` into your Open WebUI pipelines directory
2. The pipeline will auto-load on next restart (requires `requests` -- installed automatically)
3. Configure valves in the Open WebUI admin panel

## Configuration (Valves)

| Valve | Default | Description |
|---|---|---|
| `OPENAI_API_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible API base URL (Ollama, OpenAI, vLLM, etc.) |
| `OPENAI_API_KEY` | *(empty)* | API key (leave blank for Ollama) |
| `MODEL_ID` | `llama3.1:8b` | Model for all LLM calls |
| `SEARCH_PROVIDER` | `tavily` | `tavily` or `searxng` |
| `TAVILY_API_KEY` | *(empty)* | Required when using Tavily |
| `SEARXNG_BASE_URL` | `http://localhost:8888` | SearXNG instance URL |
| `CRAWL4AI_BASE_URL` | `http://localhost:11235` | Crawl4AI Docker API URL |
| `MAX_CRAWL_URLS_PER_PASS` | `5` | Max URLs to crawl per search pass |
| `DEFAULT_DEPTH` | `auto` | Default depth: `auto`, `fast`, `standard`, `thorough` |
| `DEFAULT_CREDIBILITY` | `medium` | Default credibility: `off`, `low`, `medium`, `high` |
| `ASK_DEPTH` | `true` | Ask user for depth/credibility on RESEARCH queries |

## Usage

### Basic (auto-routing with interactive prompt)

```
What are the pros and cons of remote work?
```

Classified as RESEARCH, shows depth menu, user picks, full report generated.

### Quick factual lookup

```
What is the capital of France?
```

Classified as SIMPLE, concise sourced answer returned.

### Explicit depth override (skips the prompt)

```
depth:thorough What are the health effects of intermittent fasting?
```

### Explicit depth + credibility

```
depth:standard credibility:high How has climate change affected agriculture?
```

### Interactive prompt reply examples

```
2                          -> Standard depth, default credibility
3 credibility:high         -> Thorough depth, high credibility
fast                       -> Fast depth, default credibility
thorough credibility:off   -> Thorough depth, no credibility scoring
```

## SearXNG Setup

If using the SearXNG provider, ensure JSON format is enabled in your SearXNG `settings.yml`:

```yaml
search:
  formats:
    - json
```

Crawl4AI should be running as a Docker container:

```bash
docker run -d -p 11235:11235 unclecode/crawl4ai
```

## Architecture

```
User Query
    |
    v
+-------------+
|  pipe()     |--- Skip internal tasks (title gen, tags, etc.)
|  (generator)|--- Extract actual query from Open-WebUI formatting
|             |--- Detect two-turn settings reply
+------+------+
       |
       v
+------------------+
| _run_research()  |
|  (generator)     |
+------------------+
| 1. Classification|---> SIMPLE: quick search -> concise answer -> done
| 2. Settings menu |---> ASK_DEPTH? show menu, wait for reply
| 3. Search loop   |---> Multi-pass + LLM gap evaluation
| 4. Counter-persp |---> Criticism/controversy search (Standard+)
| 5. Subtopic dive |---> Deep-dive on emergent subtopics (Thorough)
| 6. Credibility   |---> Per-source scoring
| 7. Controversy   |---> Conflict analysis across all sources
| 8. Report        |---> Structured synthesis with citations
+------------------+
       |
       v
  Assembled Report (Markdown)
```

## Known Issues

- Open-WebUI's built-in Web Search feature may interfere -- disable it for this model
- Internal task detection (title generation, follow-up suggestions, tags) may need tuning for your Open-WebUI configuration
- `__event_emitter__` is not injected into pipeline `pipe()` calls by Open-WebUI, so collapsible status panels are not available; status is delivered via inline text yields instead

## License

MIT
