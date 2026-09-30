"""把固定 Cua Agent 的 provider 步骤和异步工具接到现有驱动。"""

import base64
import asyncio
import hashlib
import io
import json
import os
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp
import litellm
from openai import APIError
from cua_agent import ComputerAgent
from cua_agent.decorators import register_agent
from cua_agent.responses import (
    convert_completion_messages_to_responses_items,
    convert_responses_items_to_completion_messages,
    make_input_image_item,
)
from cua_agent.types import ToolError
from PIL import Image
from pydantic import BaseModel


def json_model(value):
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    raise TypeError(f"Unsupported evidence type: {type(value).__name__}")


class AgentTools:
    """只暴露模型动作；环境准备和原判分器由 benchmark host 持有。"""

    def __init__(self, session):
        self.session = session
        self.images = []

    async def screenshot(self):
        """规范成 PNG，保持原桌面尺寸，不引入模型坐标缩放。"""
        raw = await self.session.screenshot()
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            if image.size != (1280, 800):
                raise ValueError(f"Unexpected benchmark screen size: {image.size}")
            output = io.BytesIO()
            image.save(output, format="PNG")
        return base64.b64encode(output.getvalue()).decode("ascii")

    async def run(self, code):
        """执行正式 driver 调用；真实错误交给 Cua 的工具错误路径。"""
        try:
            result = await self.session.run_code(code)
        except aiohttp.ClientResponseError as error:
            # SourceSession 已记录这次真实 HTTP 响应，不能只留下通用的 500 文案。
            raise ToolError(
                json.dumps(self.session.actions[-1]["output"], ensure_ascii=False)
            ) from error
        if "error" in result:
            raise ToolError(result["error"])
        content = []
        for item in result["content"]:
            if item["type"] == "image":
                self.images.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:{item['mimeType']};base64,{item['data']}",
                    }
                )
                content.append(
                    {"type": "text", "text": "Driver image attached to next request"}
                )
            else:
                content.append(item)
        return json.dumps(content, ensure_ascii=False)

    async def computer_action(self, action: str, arguments: dict):
        """通过原生桌面动作操作界面。

        Parameters
        ----------
        action : str
            click、move、drag、scroll、press_key、type_text 或 screenshot。
        arguments : dict
            Sky 参数：click/move 使用 x/y；click 可用 mouse_button、click_count；
            drag 使用 path（x/y 数组）；scroll 使用 x/y、direction、pixels；
            press_key 使用 key；type_text 使用 text。桌面为 1280×800。
        """
        if action == "screenshot":
            if arguments:
                raise ToolError("screenshot takes no arguments")
            png = base64.b64decode(await self.screenshot())
            return json.dumps(
                {
                    "screenshot_sha256": hashlib.sha256(png).hexdigest(),
                    "screen": [1280, 800],
                    "image_in_next_request": True,
                }
            )
        if action not in {"click", "move", "drag", "scroll", "press_key", "type_text"}:
            raise ToolError(f"Unsupported desktop action: {action}")
        return await self.run(
            f"nodeRepl.write(await sky[{json.dumps(action)}]({json.dumps(arguments)}));"
        )

    async def browser_run(self, code: str):
        """使用已安装的 Browser API 观察或操作界面。

        Parameters
        ----------
        code : str
            JavaScript；先读取 browser.documentation()，再使用其列出的接口。
        """
        return await self.run(code)


@register_agent(models=r"^benchmark-native/", priority=100)
class NativeCompletion:
    """只转换 provider 协议；Agent 循环和工具历史继续归 Cua。"""

    def configure(self, tools, browser):
        self.tools = tools
        self.expected_names = (
            {"computer_action", "browser_run"} if browser else {"computer_action"}
        )
        self.requests = []
        self.responses = []

    def get_capabilities(self):
        return ["step"]

    async def predict_click(self, **kwargs):
        raise NotImplementedError(
            "This benchmark uses native tool calls, not click prediction"
        )

    async def predict_step(
        self,
        messages,
        model,
        tools,
        stream=False,
        max_retries=0,
        computer_handler=None,
        use_prompt_caching=False,
        _on_api_start=None,
        _on_api_end=None,
        _on_usage=None,
        _on_screenshot=None,
        **generation,
    ):
        """把实际 SDK 请求转换为原生 tools/tool_calls，保留 call ID。"""
        if stream or use_prompt_caching or max_retries:
            raise ValueError(
                "This benchmark requires non-streaming requests without hidden caching or retries"
            )
        # 1. 先检查实际 schema，不能把未注册的 Browser 当作降级成功。
        schemas = []
        for tool in tools:
            if tool["type"] != "function":
                raise ValueError("This harness requires explicit callable tool schemas")
            schemas.append({"type": "function", "function": tool["function"]})
        names = {item["function"]["name"] for item in schemas}
        if names != self.expected_names or len(schemas) != len(names):
            raise ValueError(
                f"Actual tools differ: {sorted(names)}; expected {sorted(self.expected_names)}"
            )
        # 2. 初始图像保留在 SDK 输入；后续步骤增加当前 UI，不把 grader 送入请求。
        view = list(messages)
        if self.requests:
            view.append(make_input_image_item(await self.tools.screenshot()))
        if self.tools.images:
            view.append({"role": "user", "content": self.tools.images.copy()})
            self.tools.images.clear()
        completion = convert_responses_items_to_completion_messages(view)
        kwargs: dict[str, Any] = dict(
            model=model.removeprefix("benchmark-native/"),
            messages=completion,
            tools=schemas,
            stream=False,
            max_retries=0,
            parallel_tool_calls=False,
            **generation,
        )
        evidence = {key: kwargs[key] for key in ("model", "messages", "tools")}
        evidence["settings"] = {
            key: kwargs[key] for key in ("max_tokens", "temperature") if key in kwargs
        }
        self.requests.append(evidence)
        if _on_api_start is not None:
            await _on_api_start(kwargs)
        response = await litellm.acompletion(**kwargs)
        if _on_api_end is not None:
            await _on_api_end(kwargs, response)
        # 3. 仅保存响应事实，不导出客户端 headers、凭据或 SDK 私有配置。
        raw = response.model_dump()
        self.responses.append(
            {key: raw[key] for key in ("id", "model", "choices", "usage")}
        )
        usage = raw["usage"]
        if _on_usage is not None:
            await _on_usage(usage)
        message = raw["choices"][0]["message"]
        # 固定 SDK 不捕获工具参数的 JSON 解码错误；只在模型响应边界分类。
        for call in message.get("tool_calls") or []:
            function = call["function"]
            try:
                arguments = json.loads(function["arguments"])
            except json.JSONDecodeError as error:
                raise ToolError(
                    f"Invalid arguments for {function['name']}: {error}"
                ) from error
            if not isinstance(arguments, dict):
                raise ToolError(
                    f"Arguments for {function['name']} must be a JSON object"
                )
        return {
            "output": convert_completion_messages_to_responses_items([message]),
            "usage": usage,
        }


class StepLimit:
    """用上游 continue hook 限制真实 provider 步数。"""

    def __init__(self, loop, limit):
        self.loop, self.limit = loop, limit
        self.exhausted = False

    async def on_run_continue(self, kwargs, old_items, new_items):
        self.exhausted = len(self.loop.requests) >= self.limit
        return not self.exhausted


async def run_agent(args, env, description, case_dir, guidance):
    """只运行原版 Agent；原题准备、判分和证据落盘仍归 benchmark host。"""
    # 1. 注册上游正式 loop 接口；custom_loop 在固定 SDK 的 run 中缺少配置记录。
    tools = AgentTools(env.session)
    functions: list[Callable[..., Awaitable[str]]] = [tools.computer_action]
    if args.agent_browser:
        functions.append(tools.browser_run)
    instruction = (
        guidance + "\n\nThis is an isolated benchmark with an empty Chromium profile. "
    )
    instruction += "Use only the tools actually listed in the request. "
    instruction += "Do not inspect task source, grader code or private page variables. "
    instruction += "Describe completion only after observing the visible goal."
    agent = ComputerAgent(
        model="benchmark-native/" + args.model,
        tools=functions,
        instructions=instruction,
        max_retries=0,
        telemetry_enabled=False,
        api_base=args.api_base,
        api_key=os.environ.get(args.api_key_env),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.request_timeout,
    )
    loop = agent.agent_loop
    loop.configure(tools, args.agent_browser)
    budget = StepLimit(loop, args.max_steps)
    agent.callbacks.append(budget)
    evidence = {"requests": loop.requests, "responses": loop.responses, "yields": []}
    status = "execution_error"
    try:
        # 2. 初始 PNG 和描述是模型输入；原参考解法从不执行。
        image = await tools.screenshot()
        (case_dir / "before.png").write_bytes(base64.b64decode(image))
        async with asyncio.timeout(args.episode_timeout):
            async for item in agent.run(
                [{"role": "user", "content": description}, make_input_image_item(image)]
            ):
                evidence["yields"].append(item)
        status = "step_limit" if budget.exhausted else "evaluated"
    except TimeoutError:
        status = "timed_out"
        await env.session.cancel_call()
    except (APIError, ToolError) as error:
        evidence["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        # 3. 保存实际 wire/历史；取消后的 driver 排空由外层 episode owner 负责。
        evidence["status"] = status
        (case_dir / "agent.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2, default=json_model)
            + "\n"
        )
    return {
        "status": status,
        "provider_steps": len(loop.requests),
        "yields": len(evidence["yields"]),
        "observation_validity": (
            "requires_manual_review" if args.agent_browser else "desktop_only"
        ),
        "provider_kind": args.provider_kind,
    }
