from fastapi import FastAPI, HTTPException
from agent.plugin_composition import DashboardContext
from plugins.onboarding.contract import ONBOARDING

inject = (ONBOARDING,)


def register(app: FastAPI, context: DashboardContext) -> None:
    @app.get("/api/dashboard/onboarding/catalog")
    async def catalog():
        return await context.require(ONBOARDING).catalog()

    @app.get("/api/dashboard/onboarding/status/{key:path}")
    async def status(key: str):
        try:
            return await context.require(ONBOARDING).status(key)
        except KeyError as error:
            raise HTTPException(404, "此配置项已移除，请刷新目录") from error
