# Research Assistant Pipeline

A multi-pass web research pipeline for Open-WebUI that performs deep research using web searches and LLM synthesis.

## Version 0.1

Initial release with:
- Multi-pass web search using Tavily API
- LLM-powered query optimization and report generation
- OpenRouter API integration
- Auto-routing between quick answers and deep research
- Citation management and source tracking

## Configuration

Required environment variables/valves:
- `OPENAI_API_KEY`: OpenRouter API key
- `TAVILY_API_KEY`: Tavily API key
- `OPENAI_API_BASE_URL`: OpenRouter endpoint (default: https://openrouter.ai/api/v1)
- `MODEL_ID`: LLM model to use (e.g., anthropic/claude-3.5-sonnet)

## Installation

1. Place `research_assistant_pipeline.py` in your Open-WebUI pipelines directory
2. Restart the pipelines container
3. Configure the pipeline valves with your API keys
4. Select "Research Assistant" from the model picker

## Usage

Simply ask questions in Open-WebUI. The pipeline will:
1. Classify the query (quick vs deep research)
2. Perform web searches via Tavily
3. Generate comprehensive reports with citations

## Known Issues

- Open-WebUI's Web Search feature may interfere - disable it for this model
- Internal task detection may need tuning for your use case
