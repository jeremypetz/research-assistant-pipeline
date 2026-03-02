"""
title: Research Assistant
author: Jeremy
date: 2026-03-02
version: 1.2
license: MIT
description: Default Research Assistant wrapper. Configure valves in the Open-WebUI admin panel.
"""
import sys
import os

# Ensure the pipelines directory is on sys.path so the
# research_assistant_base package can be imported.
_pipelines_dir = os.path.dirname(os.path.abspath(__file__))
if _pipelines_dir not in sys.path:
    sys.path.insert(0, _pipelines_dir)

from research_assistant_base import Pipeline as _Base


class Pipeline(_Base):
    """
    Default Research Assistant instance.

    This is a thin wrapper around the base pipeline that registers as a
    separate model in Open-WebUI with its own independent valve configuration.
    Edit the defaults below or change them at runtime via the Valves panel.
    """

    def __init__(self):
        super().__init__()
        self.name = "Research Assistant"
        # Uses all base defaults — override any here if desired:
        # self.valves = self.Valves(
        #     OPENAI_API_BASE_URL="https://api.openai.com/v1",
        #     OPENAI_API_KEY="sk-...",
        #     MODEL_ID="gpt-4o",
        #     SEARCH_PROVIDER="tavily",
        #     TAVILY_API_KEY="tvly-...",
        # )
