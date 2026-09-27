"""ContentBlitz package — AI Content Marketing multi-agent system.

Interview Kickstart capstone, agent 2. Six specialized agents behind an
intelligent LangGraph Query Handler:

- query_handler   intelligent LLM router (initial + post-research routing)
- deep_research   comprehensive web research with citations
- seo_blog        search-optimized long-form writer
- linkedin        professional social writer
- image           DALL-E 3 -> DALL-E 2 -> labelled placeholder
- strategist      content strategy & repurposing plans
"""

__version__ = "0.2.0"

from contentblitz import agents, brand, config, exports, graph, memory, quality, tools, workflows  # noqa: E402,F401
