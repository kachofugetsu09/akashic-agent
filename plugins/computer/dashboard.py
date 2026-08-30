from __future__ import annotations

from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field

from agent.plugin_composition import DashboardContext


class ComputerInput(BaseModel):
    """Bound one user input before it crosses into the Computer workload."""

    model_config = ConfigDict(extra="forbid")

    action: Literal[
        "click", "double_click", "move", "drag", "type", "key", "scroll", "wait"
    ]
    x: int | None = Field(default=None, ge=0, le=1279)
    y: int | None = Field(default=None, ge=0, le=799)
    to_x: int | None = Field(default=None, ge=0, le=1279)
    to_y: int | None = Field(default=None, ge=0, le=799)
    text: str | None = Field(default=None, max_length=16_384)
    key: str | None = Field(default=None, max_length=80)
    amount: int | None = Field(default=None, ge=-100, le=100)
    ms: int | None = Field(default=None, ge=0, le=30_000)


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
    def computer_input(payload: ComputerInput) -> Response:
        result = forward("POST", "/input", payload.model_dump(exclude_none=True))
        return Response(result.content, media_type="application/json")

    return client
