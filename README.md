# Multi-Agent AI Researcher

An evidence-grounded research system built around LangGraph.

This repository is being initialized from the existing evidence-research portfolio. The first implementation milestone will preserve the tested research primitives (Tavily search, BM25/hybrid ranking, citation screening, optional NLI verification and SQLite run storage) while introducing explicit LangGraph orchestration.

## Development approach

- Keep agent orchestration separate from search, retrieval and verification services.
- Use structured state and Pydantic contracts between graph stages.
- Dispatch independent research tasks concurrently with bounded fan-out.
- Preserve source metadata and make evidence traceable.
- Keep deterministic tests independent of external API credentials.

See the development branch and pull requests for incremental implementation.
