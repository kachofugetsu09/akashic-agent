import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { Plugin } from "vite";
import { migrateThemePreference } from "../theme/src/theme-preference";

const here = dirname(fileURLToPath(import.meta.url));
const catalogPath = resolve(here, "..", "theme", "src", "theme-catalog.json");

interface SplashTheme {
  id: string;
  family: string;
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
  if (catalog.version !== 3 || typeof catalog.defaultThemeId !== "string" || !Array.isArray(catalog.themes)) {
    throw new Error("splash 注入: theme catalog 结构无效");
  }
  const themes = catalog.themes.map((raw, index) => {
    const theme = raw as {
      id?: unknown;
      family?: unknown;
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
    if (typeof theme.family !== "string") throw new Error(`splash 注入: themes[${index}].family 无效`);
    return { id: theme.id, family: theme.family, colorScheme: theme.colorScheme, background, ink };
  });
  if (!themes.some((theme) => theme.id === catalog.defaultThemeId)) {
    throw new Error("splash 注入: 默认主题不在目录中");
  }
  return { defaultThemeId: catalog.defaultThemeId, themes };
}

// 跟随系统的偏好按配色查 [亮, 暗] 两个主题 id。
function splashFamilies(themes: SplashTheme[]): Record<string, [string, string]> {
  const families: Record<string, [string, string]> = {};
  for (const theme of themes) {
    const pair = families[theme.family] ??= ["", ""];
    pair[theme.colorScheme === "dark" ? 1 : 0] = theme.id;
  }
  return families;
}

/**
 * 外壳首帧注入：主题变量由运行时 JS（theme-runtime installThemeCss）安装，JS 就绪前
 * 任何 var(--ak-*) 都无效，#root 空白意味着冷启动首帧要等整个模块图求值。这里在构建期把
 * 目录里的 canvas/secondary 两色内联进 HTML，并内联一段阻塞脚本按 cookie 预解析主题
 * （镜像 theme-runtime 的偏好解析，共用退役身份迁移，支持 system-<配色>），
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
        `fallback=${JSON.stringify(defaultThemeId)},` +
        `families=${JSON.stringify(splashFamilies(themes))};` +
        `var apply=function(id){d.dataset.theme=id;var isDark=dark.indexOf(id)>=0;` +
        `d.classList.toggle("dark",isDark);d.style.colorScheme=isDark?"dark":"light";` +
        `var meta=document.querySelector('meta[name="color-scheme"]');` +
        `if(meta)meta.setAttribute("content",isDark?"dark":"light")};` +
        `try{var entry=document.cookie.split(";").map(function(p){return p.trim()})` +
        `.find(function(p){return p.indexOf("akashic_theme=")===0});` +
        `var id=entry?decodeURIComponent(entry.slice(14)):"";` +
        `if(!/^[a-z0-9][a-z0-9-]{0,63}$/.test(id))id="";` +
        `id=(${migrateThemePreference.toString()})(id);` +
        `var pair=id.indexOf("system-")===0&&families[id.slice(7)];` +
        `if(pair)id=pair[matchMedia("(prefers-color-scheme: dark)").matches?1:0];` +
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
