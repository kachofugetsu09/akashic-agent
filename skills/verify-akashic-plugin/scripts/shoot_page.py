# 打开 WebUI 的一个导航页，亮色、暗色、390px 窄屏各截一张，并打印控制台错误、失败请求和横向溢出。
# 用法：python shoot_page.py <导航文字> <输出目录> [--click 按钮文字 ...]
# 需要能 import playwright；没有就直接退出码 2，验收写"界面未亲眼验证"。
import argparse
import asyncio
import json
import os
import shutil
from pathlib import Path


# 1. 先用 playwright 自带浏览器，失败再找系统 chromium
async def _launch(p):
    try:
        return await p.chromium.launch()
    except Exception:
        for name in ("chromium", "chromium-browser", "google-chrome"):
            path = shutil.which(name)
            if path:
                return await p.chromium.launch(executable_path=path, args=["--no-sandbox"])
        raise


async def _shoot(label: str, out: Path, clicks: list[str]) -> dict[str, object]:
    from playwright.async_api import async_playwright

    base = f"http://{os.environ.get('AKASHIC_WEB_HOST', '127.0.0.1')}:{os.environ['AKASHIC_WEB_PORT']}/"
    out.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    async with async_playwright() as p:
        browser = await _launch(p)
        page = await browser.new_page(viewport={"width": 1280, "height": 800})
        # 资源失败由 response 带 URL 记录；浏览器自取 favicon 的 404 不经过 response，忽略它的无 URL 控制台行
        page.on("console", lambda m: errors.append(m.text[:200])
                if m.type == "error" and not m.text.startswith("Failed to load resource") else None)
        page.on("pageerror", lambda e: errors.append(f"pageerror {e}"[:200]))
        page.on("response", lambda r: errors.append(f"{r.status} {r.url}") if r.status >= 400 else None)
        # 2. 打开首页，跳过首跑引导，点导航进入页面
        await page.goto(base, wait_until="networkidle")
        skip = page.get_by_text("我已经配好了，直接聊天")
        if await skip.count():
            await skip.click()
        await page.get_by_role("button", name=label).or_(page.get_by_role("link", name=label)).first.click()
        await page.wait_for_timeout(1000)
        # 3. 依次点用户给的按钮，记录每次点击后的页面文字
        texts: list[str] = []
        for name in clicks:
            await page.get_by_role("button", name=name).first.click()
            await page.wait_for_timeout(1500)
            texts.append(f"{name} -> {(await page.locator('body').inner_text())[:200]}")
        # 4. 亮色、暗色、窄屏各一张
        shots: list[str] = []
        for theme in ("light", "dark"):
            await page.evaluate("t => document.documentElement.setAttribute('data-theme', t)", theme)
            await page.wait_for_timeout(500)
            await page.screenshot(path=out / f"{theme}.png")
            shots.append(str(out / f"{theme}.png"))
        await page.evaluate("document.documentElement.setAttribute('data-theme', 'light')")
        await page.set_viewport_size({"width": 390, "height": 780})
        await page.wait_for_timeout(500)
        await page.screenshot(path=out / "narrow.png")
        shots.append(str(out / "narrow.png"))
        overflow = await page.evaluate("document.documentElement.scrollWidth > window.innerWidth")
        await browser.close()
    return {"shots": shots, "clicks": texts, "horizontal_overflow": overflow, "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("out")
    parser.add_argument("--click", action="append", default=[])
    args = parser.parse_args()
    try:
        import playwright  # noqa: F401
    except ImportError:
        print("没有 playwright，无法截图")
        raise SystemExit(2)
    print(json.dumps(asyncio.run(_shoot(args.label, Path(args.out), args.click)), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
