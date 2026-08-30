from __future__ import annotations

import httpx
from fastapi import FastAPI, HTTPException, Response

from agent.plugin_composition import DashboardContext


def register(app: FastAPI, context: DashboardContext) -> httpx.Client:
    """Expose the exact generation's private Computer endpoint to its web tab."""

    gateway = context.workload_url("computer", "gateway")
    client = httpx.Client(base_url=gateway, timeout=125.0)

    def forward(method: str, path: str, payload: object | None = None) -> httpx.Response:
        try:
            response = client.request(method, path, json=payload)
        except httpx.HTTPError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        if response.status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"Computer returned {response.status_code}",
            )
        return response

    @app.get("/api/dashboard/computer/activity")
    def activity() -> Response:
        result = forward("GET", "/activity")
        return Response(result.content, media_type="application/json")

    @app.get("/api/dashboard/computer/screenshot")
    def screenshot() -> Response:
        result = forward("GET", "/screenshot?quiet=1")
        return Response(
            result.content,
            media_type="image/png",
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/api/dashboard/computer/input")
    def computer_input(payload: dict[str, object]) -> Response:
        result = forward("POST", "/input", payload)
        return Response(result.content, media_type="application/json")

    return client
