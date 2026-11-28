from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from research_agent.dependencies import ResearchSettings
from research_agent.graph import build_default_graph
from research_core.storage import ResearchRunStore
from research_observability.metrics import prometheus_payload, track_research_run
from research_observability.tracing import flush_telemetry, observation


class ResearchRequest(BaseModel):
    question: str = Field(min_length=8, max_length=2000)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        flush_telemetry()

    app = FastAPI(
        title="Multi-Agent AI Researcher",
        version="0.1.0",
        description="Evidence-grounded multi-agent research orchestrated with LangGraph.",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "multi-agent-ai-researcher"}

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        payload = prometheus_payload()
        if payload is None:
            raise HTTPException(
                status_code=503,
                detail='Prometheus metrics are not installed. Run: pip install -e ".[observability]"',
            )
        content, content_type = payload
        return Response(content=content, media_type=content_type)

    @app.post("/research")
    def research(request: ResearchRequest) -> dict:
        thread_id = str(uuid4())
        try:
            with track_research_run("api"):
                with observation(
                    "research.run",
                    metadata={
                        "research.entrypoint": "api",
                        "research.run_id": thread_id,
                    },
                ) as span:
                    graph = build_default_graph()
                    result = graph.invoke(
                        {"question": request.question},
                        config={"configurable": {"thread_id": thread_id}},
                    )
                    report = result["report"]
                    if hasattr(span, "update"):
                        span.update(
                            output={
                                "run_id": report.get("run_id"),
                                "source_count": len(report.get("sources", [])),
                                "claim_count": len(report.get("claim_verifications", [])),
                                "research_rounds": report.get("research_rounds", 0),
                            }
                        )
                    return report
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
