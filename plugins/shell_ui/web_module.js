import { jsx as s, jsxs as w } from "react/jsx-runtime";
import { useContext as B, createContext as U, forwardRef as I, createElement as S, useSyncExternalStore as q, useMemo as P, useState as L, useRef as D, useCallback as W, useLayoutEffect as j, useEffect as A } from "react";
import { createRoot as F } from "react-dom/client";
/**
 * @license lucide-react v1.28.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const H = (...e) => e.filter((t, r, o) => !!t && t.trim() !== "" && o.indexOf(t) === r).join(" ").trim();
/**
 * @license lucide-react v1.28.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const O = (e) => e.replace(/([a-z0-9])([A-Z])/g, "$1-$2").toLowerCase();
/**
 * @license lucide-react v1.28.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const J = (e) => e.replace(
  /^([A-Z])|[\s-_]+(\w)/g,
  (t, r, o) => o ? o.toUpperCase() : r.toLowerCase()
);
/**
 * @license lucide-react v1.28.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const M = (e) => {
  const t = J(e);
  return t.charAt(0).toUpperCase() + t.slice(1);
};
/**
 * @license lucide-react v1.28.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
var T = {
  xmlns: "http://www.w3.org/2000/svg",
  width: 24,
  height: 24,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 2,
  strokeLinecap: "round",
  strokeLinejoin: "round"
};
/**
 * @license lucide-react v1.28.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const R = (e) => {
  for (const t in e)
    if (t.startsWith("aria-") || t === "role" || t === "title")
      return !0;
  return !1;
}, V = U({}), K = () => B(V), Y = I(
  ({ color: e, size: t, strokeWidth: r, absoluteStrokeWidth: o, className: i = "", children: a, iconNode: h, ...l }, v) => {
    const {
      size: d = 24,
      strokeWidth: g = 2,
      absoluteStrokeWidth: C = !1,
      color: n = "currentColor",
      className: c = ""
    } = K() ?? {}, f = o ?? C ? Number(r ?? g) * 24 / Number(t ?? d) : r ?? g;
    return S(
      "svg",
      {
        ref: v,
        ...T,
        width: t ?? d ?? T.width,
        height: t ?? d ?? T.height,
        stroke: e ?? n,
        strokeWidth: f,
        className: H("lucide", c, i),
        ...!a && !R(l) && { "aria-hidden": "true" },
        ...l
      },
      [
        ...h.map(([m, p]) => S(m, p)),
        ...Array.isArray(a) ? a : [a]
      ]
    );
  }
);
/**
 * @license lucide-react v1.28.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const G = (e, t) => {
  const r = I(
    ({ className: o, ...i }, a) => S(Y, {
      ref: a,
      iconNode: t,
      className: H(
        `lucide-${O(M(e))}`,
        `lucide-${e}`,
        o
      ),
      ...i
    })
  );
  return r.displayName = M(e), r;
};
/**
 * @license lucide-react v1.28.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const Z = [
  ["path", { d: "M12 2v2", key: "tus03m" }],
  [
    "path",
    {
      d: "M14.837 16.385a6 6 0 1 1-7.223-7.222c.624-.147.97.66.715 1.248a4 4 0 0 0 5.26 5.259c.589-.255 1.396.09 1.248.715",
      key: "xlf6rm"
    }
  ],
  ["path", { d: "M16 12a4 4 0 0 0-4-4", key: "6vsxu" }],
  ["path", { d: "m19 5-1.256 1.256", key: "1yg6a6" }],
  ["path", { d: "M20 12h2", key: "1q8mjw" }]
], Q = G("sun-moon", Z), $ = "data:image/png;base64,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", _ = "light", ee = [{ id: "light", label: "纸感", status: "stable", colorScheme: "light", material: { primary: "#1b365d", onPrimary: "#fafaf9", primaryContainer: "#eef2f7", onPrimaryContainer: "#0d1f38", secondary: "#53524e", onSecondary: "#fafaf9", secondaryContainer: "#e8e7e2", onSecondaryContainer: "#1c1c19", tertiary: "#4a5560", onTertiary: "#fafaf9", tertiaryContainer: "#e4e8ec", onTertiaryContainer: "#151a1f", error: "#9b2c2c", onError: "#ffffff", errorContainer: "#f5d5d2", onErrorContainer: "#3b0a0a", background: "#f5f4ed", onBackground: "#141413", surface: "#f5f4ed", onSurface: "#141413", surfaceVariant: "#e8e6dc", onSurfaceVariant: "#504e49", outline: "#6b6a64", outlineVariant: "#e5e3d8", shadow: "#141413", scrim: "#141413", inverseSurface: "#2a2a28", inverseOnSurface: "#f0efeb", inversePrimary: "#a8bdd9", surfaceDim: "#e8e6dc", surfaceBright: "#faf9f5", surfaceContainerLowest: "#faf9f5", surfaceContainerLow: "#f7f6f0", surfaceContainer: "#efeee6", surfaceContainerHigh: "#e8e6dc", surfaceContainerHighest: "#e0ded4", surfaceTint: "#1b365d" }, domain: { success: "#2f5d3a", onSuccess: "#ffffff", successContainer: "#d7e8db", onSuccessContainer: "#0f2414", warning: "#6b5420", onWarning: "#ffffff", warningContainer: "#efe2c4", onWarningContainer: "#2a1f08", trace: "#5a5366", onTrace: "#ffffff", traceContainer: "#e7e3ef", onTraceContainer: "#1f1a28", info: "#2a4f66", onInfo: "#ffffff", infoContainer: "#d9e8f1", onInfoContainer: "#0c1f2a" } }, { id: "dark", label: "墨纸", status: "stable", colorScheme: "dark", material: { primary: "#a8bdd9", onPrimary: "#0d1f38", primaryContainer: "#2a4568", onPrimaryContainer: "#eef2f7", secondary: "#c8c6bf", onSecondary: "#1c1c19", secondaryContainer: "#3d3d3a", onSecondaryContainer: "#e8e7e2", tertiary: "#c2c8cf", onTertiary: "#151a1f", tertiaryContainer: "#3a424a", onTertiaryContainer: "#e4e8ec", error: "#e8b4b0", onError: "#3b0a0a", errorContainer: "#6e1f1f", onErrorContainer: "#f5d5d2", background: "#141413", onBackground: "#e8e7e3", surface: "#141413", onSurface: "#e8e7e3", surfaceVariant: "#3d3d3a", onSurfaceVariant: "#b0aea7", outline: "#8f8d86", outlineVariant: "#30302e", shadow: "#000000", scrim: "#000000", inverseSurface: "#e8e7e3", inverseOnSurface: "#2a2a28", inversePrimary: "#1b365d", surfaceDim: "#141413", surfaceBright: "#3d3d3a", surfaceContainerLowest: "#0f0f0e", surfaceContainerLow: "#1c1c1a", surfaceContainer: "#222220", surfaceContainerHigh: "#30302e", surfaceContainerHighest: "#3d3d3a", surfaceTint: "#a8bdd9" }, domain: { success: "#a5c9ae", onSuccess: "#0f2414", successContainer: "#2f5d3a", onSuccessContainer: "#d7e8db", warning: "#d6c08a", onWarning: "#2a1f08", warningContainer: "#6b5420", onWarningContainer: "#efe2c4", trace: "#c8c0d6", onTrace: "#1f1a28", traceContainer: "#5a5366", onTraceContainer: "#e7e3ef", info: "#9ec5d8", onInfo: "#0c1f2a", infoContainer: "#2a4f66", onInfoContainer: "#d9e8f1" } }], te = {
  defaultThemeId: _,
  themes: ee
}, re = [
  "primary",
  "onPrimary",
  "primaryContainer",
  "onPrimaryContainer",
  "secondary",
  "onSecondary",
  "secondaryContainer",
  "onSecondaryContainer",
  "tertiary",
  "onTertiary",
  "tertiaryContainer",
  "onTertiaryContainer",
  "error",
  "onError",
  "errorContainer",
  "onErrorContainer",
  "background",
  "onBackground",
  "surface",
  "onSurface",
  "surfaceVariant",
  "onSurfaceVariant",
  "outline",
  "outlineVariant",
  "shadow",
  "scrim",
  "inverseSurface",
  "inverseOnSurface",
  "inversePrimary",
  "surfaceDim",
  "surfaceBright",
  "surfaceContainerLowest",
  "surfaceContainerLow",
  "surfaceContainer",
  "surfaceContainerHigh",
  "surfaceContainerHighest",
  "surfaceTint"
], ne = [
  "success",
  "onSuccess",
  "successContainer",
  "onSuccessContainer",
  "warning",
  "onWarning",
  "warningContainer",
  "onWarningContainer",
  "trace",
  "onTrace",
  "traceContainer",
  "onTraceContainer",
  "info",
  "onInfo",
  "infoContainer",
  "onInfoContainer"
], ae = "akashic_theme", N = /^[a-z0-9][a-z0-9-]{0,63}$/, oe = /^#[0-9a-f]{6}(?:[0-9a-f]{2})?$/i, E = "akashic-theme-change";
function ie(e) {
  if (!e || typeof e != "object") throw new Error("Theme catalog 不是对象");
  const t = e;
  if (typeof t.defaultThemeId != "string" || !Array.isArray(t.themes))
    throw new Error("Theme catalog 结构无效");
  const r = t.themes.map((o, i) => {
    if (!o || typeof o != "object") throw new Error(`Theme catalog themes[${i}] 不是对象`);
    const a = o;
    if (typeof a.id != "string" || !N.test(a.id))
      throw new Error(`Theme catalog themes[${i}].id 无效`);
    if (typeof a.label != "string" || !a.label.trim())
      throw new Error(`Theme catalog themes[${i}].label 无效`);
    if (a.status !== "stable" && a.status !== "experimental")
      throw new Error(`Theme catalog themes[${i}].status 无效`);
    if (a.colorScheme !== "light" && a.colorScheme !== "dark")
      throw new Error(`Theme catalog themes[${i}].colorScheme 无效`);
    const h = k(a, "material", re), l = k(a, "domain", ne);
    return {
      id: a.id,
      label: a.label,
      status: a.status,
      colorScheme: a.colorScheme,
      material: h,
      domain: l
    };
  });
  if (new Set(r.map((o) => o.id)).size !== r.length)
    throw new Error("Theme catalog 存在重复 theme id");
  if (!r.some((o) => o.id === t.defaultThemeId))
    throw new Error("Theme catalog 默认主题不存在");
  return { defaultThemeId: t.defaultThemeId, themes: r };
}
function k(e, t, r) {
  const o = e[t];
  if (!o || typeof o != "object")
    throw new Error(`Theme catalog ${String(e.id)}.${t} 无效`);
  const i = o;
  for (const a of r)
    if (typeof i[a] != "string" || !oe.test(i[a]))
      throw new Error(`Theme catalog ${String(e.id)}.${t}.${a} 无效`);
  return Object.fromEntries(r.map((a) => [a, i[a]]));
}
const u = ie(te), y = new Map(u.themes.map((e) => [e.id, e]));
let b = {
  effectiveThemeId: u.defaultThemeId
};
function ce(e) {
  const t = e === "warm-paper" ? "light" : e, r = y.has(t) ? t : u.defaultThemeId;
  return {
    requestedThemeId: e,
    effectiveThemeId: r,
    unavailable: r !== e && e !== "warm-paper"
  };
}
function se(e) {
  var r;
  b = e;
  const t = y.get(e.effectiveThemeId);
  if (!t) throw new Error(`Theme catalog 缺少有效主题: ${e.effectiveThemeId}`);
  document.documentElement.dataset.theme = t.id, document.documentElement.style.colorScheme = t.colorScheme, (r = document.querySelector('meta[name="color-scheme"]')) == null || r.setAttribute("content", t.colorScheme), window.dispatchEvent(new CustomEvent(E));
}
function fe(e, t = !0) {
  if (!N.test(e)) throw new Error(`Theme id 无效: ${e}`);
  const r = ce(e);
  return se(r), t && (document.cookie = `${ae}=${encodeURIComponent(e)}; Path=/; Max-Age=31536000; SameSite=Lax`), r;
}
function le() {
  const e = u.themes.findIndex((t) => t.id === b.effectiveThemeId);
  return fe(u.themes[(e + 1) % u.themes.length].id);
}
function X() {
  const e = y.get(b.effectiveThemeId);
  if (!e) throw new Error(`Theme catalog 缺少有效主题: ${b.effectiveThemeId}`);
  return e;
}
function de() {
  return u.themes;
}
function ue(e) {
  return window.addEventListener(E, e), () => window.removeEventListener(E, e);
}
function x() {
  return q(ue, X, X);
}
function ve(e) {
  return e.ui.inject("web.root.v1", (t) => t.register({
    id: "shell",
    children: [{ id: "shell.pages.v1", cardinality: "list" }],
    render(r, o) {
      const i = F(r);
      return i.render(/* @__PURE__ */ s(he, { pages: o.child("shell.pages.v1") })), () => i.unmount();
    }
  }));
}
function he({ pages: e }) {
  const t = x(), r = P(() => me(e.entries), [e.entries]), o = r.find((n) => n.route === "") ?? r[0], i = r.find((n) => n.setup), [a, h] = L(() => {
    var n;
    return ((n = z(r, o)) == null ? void 0 : n.id) ?? "";
  }), [l, v] = L("starting"), d = D(/* @__PURE__ */ new Map()), g = window.location.origin, C = W((n) => {
    h(n.id);
    const c = `${window.location.pathname}${window.location.search}`;
    window.history.replaceState(null, "", n.route ? `${c}#${n.route}` : c);
  }, []);
  return j(() => {
    const n = [];
    for (const c of r) {
      const f = d.current.get(c.id);
      f && n.push(e.render(c.id, f));
    }
    return () => {
      for (const c of n.reverse()) c();
    };
  }, [r, e]), A(() => {
    const n = () => {
      const c = z(r, o);
      c && h(c.id);
    };
    return window.addEventListener("hashchange", n), window.addEventListener("popstate", n), () => {
      window.removeEventListener("hashchange", n), window.removeEventListener("popstate", n);
    };
  }, [o, r]), A(() => {
    let n = !0;
    const c = async () => {
      try {
        const m = await fetch("/api/shell/state", { cache: "no-store" }), p = await m.json();
        if (!m.ok || typeof p.status != "string" || typeof p.chatReady != "boolean")
          throw new Error("/api/shell/state 返回了无效状态");
        n && v(p.chatReady ? "ready" : p.status);
      } catch (m) {
        console.error("[shell-ui] shell readiness failed", m), n && v("starting");
      }
    };
    c();
    const f = window.setInterval(c, 1500);
    return () => {
      n = !1, window.clearInterval(f);
    };
  }, []), A(() => {
    l === "needs_setup" && i && a !== i.id && C(i);
  }, [a, C, i, l]), A(() => {
    for (const n of d.current.values())
      n.querySelectorAll("iframe").forEach((c) => {
        var f;
        return (f = c.contentWindow) == null ? void 0 : f.postMessage(
          { type: "akashic.theme", themeId: t.id },
          g
        );
      });
  }, [g, t.id]), /* @__PURE__ */ w("div", { className: "unified-shell", children: [
    /* @__PURE__ */ w("header", { className: "primary-band", "aria-label": "Akashic 主导航", children: [
      /* @__PURE__ */ w("div", { className: "primary-band-brand", title: "Akashic", children: [
        /* @__PURE__ */ s("img", { src: $, alt: "" }),
        /* @__PURE__ */ s("strong", { children: "Akashic" })
      ] }),
      /* @__PURE__ */ s("nav", { className: "primary-band-nav", "aria-label": "主要功能", children: r.map((n) => /* @__PURE__ */ w(
        "button",
        {
          type: "button",
          className: `primary-rail-button ${a === n.id ? "is-active" : ""}`,
          "aria-label": n.label,
          title: n.label,
          "aria-current": a === n.id ? "page" : void 0,
          onClick: () => C(l === "needs_setup" && i ? i : n),
          children: [
            /* @__PURE__ */ s("span", { className: "shell-page-icon", "aria-hidden": "true", dangerouslySetInnerHTML: { __html: n.iconSvg } }),
            /* @__PURE__ */ s("span", { children: n.label })
          ]
        },
        n.id
      )) }),
      /* @__PURE__ */ s("div", { className: "primary-band-footer", children: /* @__PURE__ */ s(pe, {}) })
    ] }),
    /* @__PURE__ */ s("div", { className: "shell-view-stack", children: r.map((n) => /* @__PURE__ */ s(
      "section",
      {
        ref: (c) => {
          c ? d.current.set(n.id, c) : d.current.delete(n.id);
        },
        className: `shell-view ${a === n.id ? "is-active" : ""}`,
        "aria-hidden": a !== n.id
      },
      n.id
    )) })
  ] });
}
function me(e) {
  const t = e.map((r) => {
    if (typeof r.label != "string" || typeof r.route != "string" || typeof r.iconSvg != "string" || !r.iconSvg.startsWith("<svg"))
      throw new Error(`Shell 页面合同无效: ${r.id}`);
    return r;
  });
  if (new Set(t.map((r) => r.route)).size !== t.length)
    throw new Error("Shell 页面 route 不能重复");
  if (t.filter((r) => r.setup).length > 1)
    throw new Error("Shell 只能注册一个 setup 页面");
  return t;
}
function z(e, t) {
  const r = window.location.hash.slice(1);
  return e.find((o) => o.route === r) ?? t;
}
function pe() {
  const e = x(), t = de(), r = t.findIndex((i) => i.id === e.id), o = t[(r + 1) % t.length];
  return /* @__PURE__ */ w(
    "button",
    {
      type: "button",
      onClick: () => le(),
      title: `当前主题：${e.label}；切换到${o.label}`,
      "aria-label": `切换主题，当前为${e.label}，下一主题为${o.label}`,
      className: "theme-cycle-button",
      children: [
        /* @__PURE__ */ s(Q, { size: 20, strokeWidth: 2, "aria-hidden": "true" }),
        /* @__PURE__ */ w("span", { children: [
          "主题 · ",
          e.label
        ] })
      ]
    }
  );
}
export {
  ve as activate
};
