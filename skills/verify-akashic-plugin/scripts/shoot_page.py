"""打开实际 Web 页面，保存亮色、暗色和窄屏截图及交互、请求证据。"""
import argparse
import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from urllib.parse import urlsplit


# 插件按 resources.json 取回的 Chromium（visuals、design 声明）：Core 镜像不带浏览器，生产里由这里提供。
# 只认带完成标记的资源，最近发布的在前；历史版本可能残缺，所以由调用方逐个试启动。
def _plugin_resource_browsers() -> list[str]:
    workspace = os.environ.get("AKASHIC_WORKSPACE")
    if not workspace:
        return []
    entry = "content/chrome-headless-shell-linux64/chrome-headless-shell"
    published = [marker.parent for marker in Path(workspace).glob("runtime/plugin-resources/*/COMPLETE.json")]
    published.sort(key=lambda folder: (folder / "COMPLETE.json").stat().st_mtime, reverse=True)
    return [str(folder / entry) for folder in published if (folder / entry).is_file()]


async def _launch(playwright, executable: str | None):
    """显式路径只试它；否则按已安装 Playwright 浏览器、插件资源里的浏览器、系统浏览器顺序逐个试启动。"""
    if executable is not None:
        if not Path(executable).is_file():
            print(f"--browser 指定的浏览器不存在: {executable}", file=sys.stderr)
            raise SystemExit(2)
        # 启动失败保留真实错误，不关闭浏览器沙箱或掩盖依赖问题。
        return await playwright.chromium.launch(executable_path=executable)
    bundled = playwright.chromium.executable_path
    candidates = ([bundled] if Path(bundled).is_file() else []) + _plugin_resource_browsers() + [
        path for name in ("chromium", "chromium-browser", "google-chrome") if (path := shutil.which(name))]
    failures: list[str] = []
    for candidate in candidates:
        try:
            return await playwright.chromium.launch(executable_path=candidate)
        except Exception as error:  # 逐个候选尝试；全部失败时把每个的真实错误都打印出来
            failures.append(f"{candidate}: {str(error).strip().splitlines()[0] if str(error).strip() else type(error).__name__}")
    print("没有可用浏览器；界面未亲眼验证，可用 --browser 指定已安装路径", file=sys.stderr)
    for line in failures:
        print(f"  启动失败 {line}", file=sys.stderr)
    raise SystemExit(2)


async def _set_theme(page, theme: str) -> None:
    """等主题的 CSS 过渡结束，避免把中间颜色当作暗色验收。"""
    await page.evaluate("""async theme => {
        document.documentElement.setAttribute('data-theme', theme);
        await new Promise(requestAnimationFrame);
        await Promise.allSettled(document.getAnimations()
            .filter(animation => animation instanceof CSSTransition)
            .map(animation => animation.finished));
    }""", theme)


async def _shoot(label: str, out: Path, clicks: list[str], url: str,
                 executable: str | None) -> dict[str, object]:
    """在独立页面记录操作和三种视口，失败也由本次调用关闭浏览器。"""
    from playwright.async_api import async_playwright

    out.mkdir(parents=True, exist_ok=False)
    errors: list[str] = []
    cancelled_requests: list[str] = []
    browser_notes: list[str] = []
    async with async_playwright() as playwright:
        browser = await _launch(playwright, executable)
        try:
            page = await browser.new_page(viewport={"width": 1280, "height": 800})
            def record_console(message):
                """保留控制台故障；浏览器自动请求的缺省图标单列。"""
                if message.type != "error":
                    return
                location = message.location.get("url", "")
                detail = f"{message.text} {location}"
                target = browser_notes if (urlsplit(location).path == "/favicon.ico"
                                           and "404" in message.text) else errors
                target.append(detail)

            def record_failure(request):
                """路由切换取消的请求保留证据，但不当作连接故障。"""
                target = cancelled_requests if request.failure == "net::ERR_ABORTED" else errors
                target.append(f"{request.failure} {request.url}")

            page.on("console", record_console)
            page.on("pageerror", lambda error: errors.append(f"pageerror {error}"))
            page.on("response", lambda response: errors.append(f"{response.status} {response.url}")
                    if response.status >= 400 else None)
            page.on("requestfailed", record_failure)
            # 1. 以实际导航元素就绪为准，不等待整个应用停止网络活动。
            await page.goto(url, wait_until="domcontentloaded")
            skip = page.get_by_text("我已经配好了，直接聊天", exact=True)
            if await skip.is_visible():
                await skip.click()
            await page.get_by_role("button", name=label, exact=True).or_(
                page.get_by_role("link", name=label, exact=True),
            ).first.click()
            # 2. 保存每次点击后的页面文字；异步业务结果仍需按需求核对。
            texts: list[str] = []
            for name in clicks:
                await page.get_by_role("button", name=name, exact=True).first.click()
                await page.wait_for_timeout(1500)
                texts.append(f"{name} -> {(await page.locator('body').inner_text())[:2000]}")
            # 3. 主题与窄屏截图不覆盖上次证据。
            shots: list[str] = []
            for theme in ("light", "dark"):
                await _set_theme(page, theme)
                await page.screenshot(path=out / f"{theme}.png")
                shots.append(str(out / f"{theme}.png"))
            await _set_theme(page, "light")
            await page.set_viewport_size({"width": 390, "height": 780})
            await page.screenshot(path=out / "narrow.png")
            shots.append(str(out / "narrow.png"))
            overflow = await page.evaluate("document.documentElement.scrollWidth > window.innerWidth")
        finally:
            await browser.close()
    return {"shots": shots, "clicks": texts, "horizontal_overflow": overflow, "errors": errors,
            "cancelled_requests": cancelled_requests, "browser_notes": browser_notes}


def main() -> None:
    """明确验证网页地址；缺少浏览器依赖与实际验收失败使用不同退出码。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label")
    parser.add_argument("out", type=Path)
    parser.add_argument("--url", required=True, help="当前实例可访问的完整 Web URL")
    parser.add_argument("--browser", help="已安装 Chromium 的路径")
    parser.add_argument("--click", action="append", default=[])
    args = parser.parse_args()
    url = urlsplit(args.url)
    if url.scheme not in {"http", "https"} or not url.hostname:
        parser.error("--url 必须是完整的 http(s) 网页地址")
    try:
        import playwright  # noqa: F401
    except ImportError:
        print("没有 playwright；界面未亲眼验证", file=sys.stderr)
        raise SystemExit(2)
    result = asyncio.run(_shoot(args.label, args.out, args.click, args.url, args.browser))
    print(json.dumps(result, ensure_ascii=False, indent=1))
    if result["horizontal_overflow"] or result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
