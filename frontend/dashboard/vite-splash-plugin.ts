import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { Plugin } from "vite";

const here = dirname(fileURLToPath(import.meta.url));
const catalogPath = resolve(here, "..", "theme", "src", "theme-catalog.json");

interface SplashTheme {
  id: string;
  colorScheme: "light" | "dark";
  background: string;
  ink: string;
}

const THEME_ID_PATTERN = /^[a-z0-9][a-z0-9-]{0,63}$/;
const HEX_COLOR_PATTERN = /^#[0-9a-f]{6}(?:[0-9a-f]{2})?$/i;

/** 构建期从主题目录取 splash 需要的两色；目录结构破损时让构建直接失败。 */
function loadSplashThemes(): { defaultThemeId: string; themes: SplashTheme[] } {
  const catalog = JSON.parse(readFileSync(catalogPath, "utf8")) as {
    version?: unknown;
    defaultThemeId?: unknown;
    themes?: unknown;
  };
  if (catalog.version !== 2 || typeof catalog.defaultThemeId !== "string" || !Array.isArray(catalog.themes)) {
    throw new Error("splash 注入: theme catalog 结构无效");
  }
  const themes = catalog.themes.map((raw, index) => {
    const theme = raw as {
      id?: unknown;
      colorScheme?: unknown;
      material?: { background?: unknown; onSurfaceVariant?: unknown };
    };
    if (typeof theme.id !== "string" || !THEME_ID_PATTERN.test(theme.id)) {
      throw new Error(`splash 注入: themes[${index}].id 无效`);
    }
    if (theme.colorScheme !== "light" && theme.colorScheme !== "dark") {
      throw new Error(`splash 注入: themes[${index}].colorScheme 无效`);
    }
    const background = theme.material?.background;
    const ink = theme.material?.onSurfaceVariant;
    if (typeof background !== "string" || !HEX_COLOR_PATTERN.test(background)
      || typeof ink !== "string" || !HEX_COLOR_PATTERN.test(ink)) {
      throw new Error(`splash 注入: themes[${index}] 缺少 background/onSurfaceVariant`);
    }
    return { id: theme.id, colorScheme: theme.colorScheme, background, ink };
  });
  if (!themes.some((theme) => theme.id === catalog.defaultThemeId)) {
    throw new Error("splash 注入: 默认主题不在目录中");
  }
  return { defaultThemeId: catalog.defaultThemeId, themes };
}

/**
 * 外壳首帧注入：主题变量由运行时 JS（theme-runtime installThemeCss）安装，JS 就绪前
 * 任何 var(--ak-*) 都无效，#root 空白意味着冷启动首帧要等整个模块图求值。这里在构建期把
 * 目录里的 canvas/secondary 两色内联进 HTML，并内联一段阻塞脚本按 cookie 预解析主题
 * （镜像 theme-runtime 的 readCookieTheme + resolveSelection，含 warm-paper 迁移），
 * 让 index.html 里的 .akashic-entry-splash 在 HTML 到达时即以正确主题着色——
 * 深色用户也顺带不再先看到一帧浏览器默认白底。观感镜像
 * frontend/chat/src/styles.css 的 .webui-entry-loading（改一边时两边都要看）。
 */
export function injectEntrySplash(): Plugin {
  return {
    name: "akashic-entry-splash",
    transformIndexHtml() {
      const { defaultThemeId, themes } = loadSplashThemes();
      const themeRules = themes.map((theme) =>
        `:root[data-theme="${theme.id}"]{--ak-splash-bg:${theme.background};` +
        `--ak-splash-ink:${theme.ink};color-scheme:${theme.colorScheme}}`,
      ).join("\n");
      const style = [
        themeRules,
        `html{background:var(--ak-splash-bg,${themes.find((t) => t.id === defaultThemeId)?.background})}`,
        ".akashic-entry-splash{display:grid;min-height:100vh;place-items:center;margin:0;" +
        "color:var(--ak-splash-ink);font-size:0.875rem}",
      ].join("\n");
      const script = `(function(){var d=document.documentElement,` +
        `known=${JSON.stringify(themes.map((t) => t.id))},` +
        `dark=${JSON.stringify(themes.filter((t) => t.colorScheme === "dark").map((t) => t.id))},` +
        `fallback=${JSON.stringify(defaultThemeId)};` +
        `var apply=function(id){d.dataset.theme=id;var isDark=dark.indexOf(id)>=0;` +
        `d.classList.toggle("dark",isDark);d.style.colorScheme=isDark?"dark":"light";` +
        `var meta=document.querySelector('meta[name="color-scheme"]');` +
        `if(meta)meta.setAttribute("content",isDark?"dark":"light")};` +
        `try{var entry=document.cookie.split(";").map(function(p){return p.trim()})` +
        `.find(function(p){return p.indexOf("akashic_theme=")===0});` +
        `var id=entry?decodeURIComponent(entry.slice(14)):"";` +
        `if(!/^[a-z0-9][a-z0-9-]{0,63}$/.test(id))id="";` +
        `if(id==="warm-paper")id="light";` +
        `apply(known.indexOf(id)>=0?id:fallback)}catch(e){apply(fallback)}})();`;
      // 追加到 head 末尾而不是 prepend:charset 必须留在文档前 1024 字节内,
      // 内联脚本是同步经典脚本,head 内任何位置都在首帧绘制前执行。
      return [
        { tag: "script", children: script, injectTo: "head" },
        { tag: "style", children: style, injectTo: "head" },
      ];
    },
  };
}
