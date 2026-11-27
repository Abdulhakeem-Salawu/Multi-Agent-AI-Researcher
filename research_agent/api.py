from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from research_agent.dependencies import ResearchSettings
from research_agent.graph import build_default_graph
from research_core.storage import ResearchRunStore


class ResearchRequest(BaseModel):
    question: str = Field(min_length=8, max_length=2000)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Multi-Agent AI Researcher",
        version="0.1.0",
        description="Evidence-grounded multi-agent research orchestrated with LangGraph.",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "multi-agent-ai-researcher"}

    @app.post("/research")
    def research(request: ResearchRequest) -> dict:
        thread_id = str(uuid4())
        try:
            graph = build_default_graph()
            result = graph.invoke(
                {"question": request.question},
                config={"configurable": {"thread_id": thread_id}},
            )
            return result["report"]
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Research workflow failed: {type(exc).__name__}: {exc}",
            ) from exc

    @app.get("/runs/{run_id}")
    def get_run(run_id: int) -> dict:
        settings = ResearchSettings.from_environment()
        report = ResearchRunStore(settings.database_path).get(run_id)
        if report is None:
            raise HTTPException(status_code=404, detail="Research run not found.")
        return report

    return app


app = create_app()
