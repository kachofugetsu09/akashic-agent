var Yr = Object.defineProperty;
var Gr = (e, t, n) => t in e ? Yr(e, t, { enumerable: !0, configurable: !0, writable: !0, value: n }) : e[t] = n;
var Ge = (e, t, n) => Gr(e, typeof t != "symbol" ? t + "" : t, n);
import { jsx as a, Fragment as fe, jsxs as y } from "react/jsx-runtime";
import * as m from "react";
import Tn, { useState as T, useContext as Jr, createContext as Zr, forwardRef as xn, createElement as kt, useRef as H, useLayoutEffect as lt, useEffect as z, Component as Qr, useCallback as j, useMemo as xe, useEffectEvent as Ce } from "react";
import { createRoot as ei } from "react-dom/client";
import { Btn as St, MaterialIconButton as ct, Markdown as Ln, JsonView as ti } from "@akashic/dashboard-ui";
var ni = Object.defineProperty, Re = (e, t) => ni(e, "name", { value: t, configurable: !0 }), In = !!(typeof window < "u" && window.document && window.document.createElement);
function me(e, t, { checkForDefaultPrevented: n = !0 } = {}) {
  return /* @__PURE__ */ Re(function(i) {
    if (e == null || e(i), n === !1 || !i || !i.defaultPrevented)
      return t == null ? void 0 : t(i);
  }, "handleEvent");
}
Re(me, "composeEventHandlers");
function ri(e) {
  var t;
  if (!In)
    throw new Error("Cannot access window outside of the DOM");
  return ((t = e == null ? void 0 : e.ownerDocument) == null ? void 0 : t.defaultView) ?? window;
}
Re(ri, "getOwnerWindow");
function Ot(e) {
  if (!In)
    throw new Error("Cannot access document outside of the DOM");
  return (e == null ? void 0 : e.ownerDocument) ?? document;
}
Re(Ot, "getOwnerDocument");
function Fn(e, t = !1) {
  const { activeElement: n } = Ot(e);
  if (!(n != null && n.nodeName))
    return null;
  if (Bn(n) && n.contentDocument)
    return Fn(n.contentDocument.body, t);
  if (t) {
    const r = n.getAttribute("aria-activedescendant");
    if (r) {
      const i = Ot(n).getElementById(r);
      if (i)
        return i;
    }
  }
  return n;
}
Re(Fn, "getActiveElement");
function Bn(e) {
  return e.tagName === "IFRAME";
}
Re(Bn, "isFrame");
var ii = Object.defineProperty, jt = (e, t) => ii(e, "name", { value: t, configurable: !0 });
function Mt(e, t) {
  if (typeof e == "function")
    return e(t);
  e != null && (e.current = t);
}
jt(Mt, "setRef");
function Un(...e) {
  return (t) => {
    let n = !1;
    const r = e.map((i) => {
      const o = Mt(i, t);
      return !n && typeof o == "function" && (n = !0), o;
    });
    if (n)
      return () => {
        for (let i = 0; i < r.length; i++) {
          const o = r[i];
          typeof o == "function" ? o() : Mt(e[i], null);
        }
      };
  };
}
jt(Un, "composeRefs");
function Ae(...e) {
  return m.useCallback(Un(...e), e);
}
jt(Ae, "useComposedRefs");
var ai = Object.defineProperty, V = (e, t) => ai(e, "name", { value: t, configurable: !0 });
// @__NO_SIDE_EFFECTS__
function oi(e, t) {
  const n = m.createContext(t);
  n.displayName = e + "Context";
  const r = /* @__PURE__ */ V((o) => {
    const { children: l, ...u } = o, s = m.useMemo(() => u, Object.values(u));
    return /* @__PURE__ */ a(n.Provider, { value: s, children: l });
  }, "Provider");
  r.displayName = e + "Provider";
  function i(o, l = {}) {
    const { optional: u = !1 } = l, s = m.useContext(n);
    if (s) return s;
    if (t !== void 0) return t;
    if (!u)
      throw new Error(`\`${o}\` must be used within \`${e}\``);
  }
  return V(i, "useContext"), [r, i];
}
V(oi, "createContext");
// @__NO_SIDE_EFFECTS__
function Wn(e, t = []) {
  let n = [];
  function r(o, l) {
    const u = m.createContext(l);
    u.displayName = o + "Context";
    const s = n.length;
    n = [...n, l];
    const c = /* @__PURE__ */ V((h) => {
      var N;
      const { scope: v, children: E, ...C } = h, b = ((N = v == null ? void 0 : v[e]) == null ? void 0 : N[s]) || u, p = m.useMemo(() => C, Object.values(C));
      return /* @__PURE__ */ a(b.Provider, { value: p, children: E });
    }, "Provider");
    c.displayName = o + "Provider";
    function f(h, v, E = {}) {
      var N;
      const { optional: C = !1 } = E, b = ((N = v == null ? void 0 : v[e]) == null ? void 0 : N[s]) || u, p = m.useContext(b);
      if (p) return p;
      if (l !== void 0) return l;
      if (!C)
        throw new Error(`\`${h}\` must be used within \`${o}\``);
    }
    return V(f, "useContext"), [c, f];
  }
  V(r, "createContext");
  const i = /* @__PURE__ */ V(() => {
    const o = n.map((l) => m.createContext(l));
    return /* @__PURE__ */ V(function(u) {
      const s = (u == null ? void 0 : u[e]) || o;
      return m.useMemo(
        () => ({ [`__scope${e}`]: { ...u, [e]: s } }),
        [u, s]
      );
    }, "useScope");
  }, "createScope");
  return i.scopeName = e, [r, $n(i, ...t)];
}
V(Wn, "createContextScope");
function $n(...e) {
  const t = e[0];
  if (e.length === 1) return t;
  const n = /* @__PURE__ */ V(() => {
    const r = e.map((i) => ({
      useScope: i(),
      scopeName: i.scopeName
    }));
    return /* @__PURE__ */ V(function(o) {
      const l = r.reduce((u, { useScope: s, scopeName: c }) => {
        const h = s(o)[`__scope${c}`];
        return { ...u, ...h };
      }, {});
      return m.useMemo(() => ({ [`__scope${t.scopeName}`]: l }), [l]);
    }, "useComposedScopes");
  }, "createScope");
  return n.scopeName = t.scopeName, n;
}
V($n, "composeContextScopes");
var ue = globalThis != null && globalThis.document ? m.useLayoutEffect : () => {
}, si = Object.defineProperty, ci = (e, t) => si(e, "name", { value: t, configurable: !0 }), li = m[" useId ".trim().toString()] || (() => {
}), ui = 0;
function rt(e) {
  const [t, n] = m.useState(li());
  return ue(() => {
    e || n((r) => r ?? String(ui++));
  }, [e]), e || (t ? `radix-${t}` : "");
}
ci(rt, "useId");
var di = Object.defineProperty, fi = (e, t) => di(e, "name", { value: t, configurable: !0 }), mn = m[" useEffectEvent ".trim().toString()], hn = m[" useInsertionEffect ".trim().toString()];
function qn(e) {
  if (typeof mn == "function")
    return mn(e);
  const t = m.useRef(() => {
    throw new Error("Cannot call an event handler while rendering.");
  });
  return typeof hn == "function" ? hn(() => {
    t.current = e;
  }) : ue(() => {
    t.current = e;
  }), m.useMemo(() => ((...n) => {
    var r;
    return (r = t.current) == null ? void 0 : r.call(t, ...n);
  }), []);
}
fi(qn, "useEffectEvent");
var mi = Object.defineProperty, We = (e, t) => mi(e, "name", { value: t, configurable: !0 }), hi = m[" useInsertionEffect ".trim().toString()] || ue;
function jn({
  prop: e,
  defaultProp: t,
  onChange: n = /* @__PURE__ */ We(() => {
  }, "onChange"),
  caller: r
}) {
  const [i, o, l] = zn({
    defaultProp: t,
    onChange: n
  }), u = e !== void 0, s = u ? e : i, c = m.useCallback(
    (f) => {
      var h;
      if (u) {
        const v = Kn(f) ? f(e) : f;
        v !== e && ((h = l.current) == null || h.call(l, v));
      } else
        o(f);
    },
    [u, e, o, l]
  );
  return [s, c];
}
We(jn, "useControllableState");
function zn({
  defaultProp: e,
  onChange: t
}) {
  const [n, r] = m.useState(e), i = m.useRef(n), o = m.useRef(t);
  return hi(() => {
    o.current = t;
  }, [t]), m.useEffect(() => {
    var l;
    i.current !== n && ((l = o.current) == null || l.call(o, n), i.current = n);
  }, [n, i]), [n, r, o];
}
We(zn, "useUncontrolledState");
function Kn(e) {
  return typeof e == "function";
}
We(Kn, "isFunction");
var vn = Symbol("RADIX:SYNC_STATE");
function vi(e, t, n, r) {
  const { prop: i, defaultProp: o, onChange: l, caller: u } = t, s = i !== void 0, c = qn(l), f = [{ ...n, state: o }];
  r && f.push(r);
  const [h, v] = m.useReducer(
    (p, N) => {
      if (N.type === vn)
        return { ...p, state: N.state };
      const A = e(p, N);
      return s && !Object.is(A.state, p.state) && c(A.state), A;
    },
    ...f
  ), E = h.state, C = m.useRef(E);
  m.useEffect(() => {
    C.current !== E && (C.current = E, s || c(E));
  }, [E, C, s]);
  const b = m.useMemo(() => i !== void 0 ? { ...h, state: i } : h, [h, i]);
  return m.useEffect(() => {
    s && !Object.is(i, h.state) && v({ type: vn, state: i });
  }, [i, h.state, s]), [b, v];
}
We(vi, "useControllableStateReducer");
var wt = { exports: {} }, F = {};
/**
 * @license React
 * react-dom.production.js
 *
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var gn;
function gi() {
  if (gn) return F;
  gn = 1;
  var e = Tn;
  function t(s) {
    var c = "https://react.dev/errors/" + s;
    if (1 < arguments.length) {
      c += "?args[]=" + encodeURIComponent(arguments[1]);
      for (var f = 2; f < arguments.length; f++)
        c += "&args[]=" + encodeURIComponent(arguments[f]);
    }
    return "Minified React error #" + s + "; visit " + c + " for the full message or use the non-minified dev environment for full errors and additional helpful warnings.";
  }
  function n() {
  }
  var r = {
    d: {
      f: n,
      r: function() {
        throw Error(t(522));
      },
      D: n,
      C: n,
      L: n,
      m: n,
      X: n,
      S: n,
      M: n
    },
    p: 0,
    findDOMNode: null
  }, i = Symbol.for("react.portal");
  function o(s, c, f) {
    var h = 3 < arguments.length && arguments[3] !== void 0 ? arguments[3] : null;
    return {
      $$typeof: i,
      key: h == null ? null : "" + h,
      children: s,
      containerInfo: c,
      implementation: f
    };
  }
  var l = e.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE;
  function u(s, c) {
    if (s === "font") return "";
    if (typeof c == "string")
      return c === "use-credentials" ? c : "";
  }
  return F.__DOM_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE = r, F.createPortal = function(s, c) {
    var f = 2 < arguments.length && arguments[2] !== void 0 ? arguments[2] : null;
    if (!c || c.nodeType !== 1 && c.nodeType !== 9 && c.nodeType !== 11)
      throw Error(t(299));
    return o(s, c, null, f);
  }, F.flushSync = function(s) {
    var c = l.T, f = r.p;
    try {
      if (l.T = null, r.p = 2, s) return s();
    } finally {
      l.T = c, r.p = f, r.d.f();
    }
  }, F.preconnect = function(s, c) {
    typeof s == "string" && (c ? (c = c.crossOrigin, c = typeof c == "string" ? c === "use-credentials" ? c : "" : void 0) : c = null, r.d.C(s, c));
  }, F.prefetchDNS = function(s) {
    typeof s == "string" && r.d.D(s);
  }, F.preinit = function(s, c) {
    if (typeof s == "string" && c && typeof c.as == "string") {
      var f = c.as, h = u(f, c.crossOrigin), v = typeof c.integrity == "string" ? c.integrity : void 0, E = typeof c.fetchPriority == "string" ? c.fetchPriority : void 0;
      f === "style" ? r.d.S(
        s,
        typeof c.precedence == "string" ? c.precedence : void 0,
        {
          crossOrigin: h,
          integrity: v,
          fetchPriority: E
        }
      ) : f === "script" && r.d.X(s, {
        crossOrigin: h,
        integrity: v,
        fetchPriority: E,
        nonce: typeof c.nonce == "string" ? c.nonce : void 0
      });
    }
  }, F.preinitModule = function(s, c) {
    if (typeof s == "string")
      if (typeof c == "object" && c !== null) {
        if (c.as == null || c.as === "script") {
          var f = u(
            c.as,
            c.crossOrigin
          );
          r.d.M(s, {
            crossOrigin: f,
            integrity: typeof c.integrity == "string" ? c.integrity : void 0,
            nonce: typeof c.nonce == "string" ? c.nonce : void 0
          });
        }
      } else c == null && r.d.M(s);
  }, F.preload = function(s, c) {
    if (typeof s == "string" && typeof c == "object" && c !== null && typeof c.as == "string") {
      var f = c.as, h = u(f, c.crossOrigin);
      r.d.L(s, f, {
        crossOrigin: h,
        integrity: typeof c.integrity == "string" ? c.integrity : void 0,
        nonce: typeof c.nonce == "string" ? c.nonce : void 0,
        type: typeof c.type == "string" ? c.type : void 0,
        fetchPriority: typeof c.fetchPriority == "string" ? c.fetchPriority : void 0,
        referrerPolicy: typeof c.referrerPolicy == "string" ? c.referrerPolicy : void 0,
        imageSrcSet: typeof c.imageSrcSet == "string" ? c.imageSrcSet : void 0,
        imageSizes: typeof c.imageSizes == "string" ? c.imageSizes : void 0,
        media: typeof c.media == "string" ? c.media : void 0
      });
    }
  }, F.preloadModule = function(s, c) {
    if (typeof s == "string")
      if (c) {
        var f = u(c.as, c.crossOrigin);
        r.d.m(s, {
          as: typeof c.as == "string" && c.as !== "script" ? c.as : void 0,
          crossOrigin: f,
          integrity: typeof c.integrity == "string" ? c.integrity : void 0
        });
      } else r.d.m(s);
  }, F.requestFormReset = function(s) {
    r.d.r(s);
  }, F.unstable_batchedUpdates = function(s, c) {
    return s(c);
  }, F.useFormState = function(s, c, f) {
    return l.H.useFormState(s, c, f);
  }, F.useFormStatus = function() {
    return l.H.useHostTransitionStatus();
  }, F.version = "19.2.8", F;
}
var pn;
function pi() {
  if (pn) return wt.exports;
  pn = 1;
  function e() {
    if (!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ > "u" || typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE != "function"))
      try {
        __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(e);
      } catch (t) {
        console.error(t);
      }
  }
  return e(), wt.exports = gi(), wt.exports;
}
var Xn = pi(), yi = Object.defineProperty, Y = (e, t) => yi(e, "name", { value: t, configurable: !0 });
// @__NO_SIDE_EFFECTS__
function zt(e) {
  const t = m.forwardRef((n, r) => {
    let { children: i, ...o } = n, l = null, u = !1;
    const s = [];
    _t(i) && typeof Je == "function" && (i = Je(i._payload)), m.Children.forEach(i, (v) => {
      var E;
      if (Gn(v)) {
        u = !0;
        const C = v;
        let b = "child" in C.props ? C.props.child : C.props.children;
        _t(b) && typeof Je == "function" && (b = Je(b._payload)), l = Si(C, b), s.push((E = l == null ? void 0 : l.props) == null ? void 0 : E.children);
      } else
        s.push(v);
    }), l ? l = m.cloneElement(l, void 0, s) : (
      // A `Slottable` was found but it didn't resolve to a single element (e.g.
      // it wrapped multiple elements, text, or a render-prop `child` that
      // wasn't an element). Don't fall back to treating the `Slottable` wrapper
      // itself as the slot target — throw a descriptive error below instead.
      !u && m.Children.count(i) === 1 && m.isValidElement(i) && (l = i)
    );
    const c = l ? Yn(l) : void 0, f = Ae(r, c);
    if (!l) {
      if (i || i === 0)
        throw new Error(
          u ? Ei(e) : Ci(e)
        );
      return i;
    }
    const h = Vn(o, l.props ?? {});
    return l.type !== m.Fragment && (h.ref = r ? f : c), m.cloneElement(l, h);
  });
  return t.displayName = `${e}.Slot`, t;
}
Y(zt, "createSlot");
var Hn = Symbol.for("radix.slottable");
// @__NO_SIDE_EFFECTS__
function bi(e) {
  const t = /* @__PURE__ */ Y((n) => "child" in n ? n.children(n.child) : n.children, "Slottable");
  return t.displayName = `${e}.Slottable`, t.__radixId = Hn, t;
}
Y(bi, "createSlottable");
var Si = /* @__PURE__ */ Y((e, t) => {
  if ("child" in e.props) {
    const n = e.props.child;
    return m.isValidElement(n) ? m.cloneElement(n, void 0, e.props.children(n.props.children)) : null;
  }
  return m.isValidElement(t) ? t : null;
}, "getSlottableElementFromSlottable");
function Vn(e, t) {
  const n = { ...t };
  for (const r in t) {
    const i = e[r], o = t[r];
    /^on[A-Z]/.test(r) ? i && o ? n[r] = (...u) => {
      const s = o(...u);
      return i(...u), s;
    } : i && (n[r] = i) : r === "style" ? n[r] = { ...i, ...o } : r === "className" && (n[r] = [i, o].filter(Boolean).join(" "));
  }
  return { ...e, ...n };
}
Y(Vn, "mergeProps");
function Yn(e) {
  var r, i;
  let t = (r = Object.getOwnPropertyDescriptor(e.props, "ref")) == null ? void 0 : r.get, n = t && "isReactWarning" in t && t.isReactWarning;
  return n ? e.ref : (t = (i = Object.getOwnPropertyDescriptor(e, "ref")) == null ? void 0 : i.get, n = t && "isReactWarning" in t && t.isReactWarning, n ? e.props.ref : e.props.ref || e.ref);
}
Y(Yn, "getElementRef");
function Gn(e) {
  return m.isValidElement(e) && typeof e.type == "function" && "__radixId" in e.type && e.type.__radixId === Hn;
}
Y(Gn, "isSlottable");
var wi = Symbol.for("react.lazy");
function _t(e) {
  return e != null && typeof e == "object" && "$$typeof" in e && e.$$typeof === wi && "_payload" in e && Jn(e._payload);
}
Y(_t, "isLazyComponent");
function Jn(e) {
  return typeof e == "object" && e !== null && "then" in e;
}
Y(Jn, "isPromiseLike");
var Ci = /* @__PURE__ */ Y((e) => `${e} failed to slot onto its children. Expected a single React element child or \`Slottable\`.`, "createSlotError"), Ei = /* @__PURE__ */ Y((e) => `${e} failed to slot onto its \`Slottable\`. Expected \`Slottable\` to receive a single React element child.`, "createSlottableError"), Je = m[" use ".trim().toString()], Ni = Object.defineProperty, Pi = (e, t) => Ni(e, "name", { value: t, configurable: !0 }), Ri = [
  "a",
  "button",
  "div",
  "form",
  "h2",
  "h3",
  "img",
  "input",
  "label",
  "li",
  "nav",
  "ol",
  "p",
  "select",
  "span",
  "svg",
  "ul"
], De = Ri.reduce((e, t) => {
  const n = /* @__PURE__ */ zt(`Primitive.${t}`), r = m.forwardRef((i, o) => {
    const { asChild: l, ...u } = i, s = l ? n : t;
    return typeof window < "u" && (window[Symbol.for("radix-ui")] = !0), /* @__PURE__ */ a(s, { ...u, ref: o });
  });
  return r.displayName = `Primitive.${t}`, { ...e, [t]: r };
}, {});
function Zn(e, t) {
  e && Xn.flushSync(() => e.dispatchEvent(t));
}
Pi(Zn, "dispatchDiscreteCustomEvent");
var Ai = Object.defineProperty, Di = (e, t) => Ai(e, "name", { value: t, configurable: !0 });
function Pe(e) {
  const t = m.useRef(e);
  return m.useEffect(() => {
    t.current = e;
  }), m.useMemo(() => ((...n) => {
    var r;
    return (r = t.current) == null ? void 0 : r.call(t, ...n);
  }), []);
}
Di(Pe, "useCallbackRef");
var ki = Object.defineProperty, I = (e, t) => ki(e, "name", { value: t, configurable: !0 }), Tt = "dismissableLayer.update", Oi = "dismissableLayer.pointerDownOutside", Mi = "dismissableLayer.focusOutside", yn, Qn = m.createContext({
  layers: /* @__PURE__ */ new Set(),
  layersWithOutsidePointerEventsDisabled: /* @__PURE__ */ new Set(),
  branches: /* @__PURE__ */ new Set(),
  // Outside elements that belong to a layer's own dismiss affordance (eg, a
  // dialog overlay). Pressing them should dismiss the layer regardless of
  // whether or not they stop propagation.
  //
  // See https://github.com/radix-ui/primitives/issues/3346
  dismissableSurfaces: /* @__PURE__ */ new Set()
}), _i = /* @__PURE__ */ m.forwardRef(
  // blank line to reduce diff noise
  /* @__PURE__ */ I(function(t, n) {
    const {
      disableOutsidePointerEvents: r = !1,
      deferPointerDownOutside: i = !1,
      onEscapeKeyDown: o,
      onPointerDownOutside: l,
      onFocusOutside: u,
      onInteractOutside: s,
      onDismiss: c,
      ...f
    } = t, h = m.useContext(Qn), [v, E] = m.useState(null), C = (v == null ? void 0 : v.ownerDocument) ?? (globalThis == null ? void 0 : globalThis.document), [, b] = m.useState({}), p = Ae(n, E), N = Array.from(h.layers), [A] = [
      ...h.layersWithOutsidePointerEventsDisabled
    ].slice(-1), P = A ? N.indexOf(A) : -1, O = v ? N.indexOf(v) : -1, D = h.layersWithOutsidePointerEventsDisabled.size > 0, x = O >= P, $ = m.useRef(!1), he = tr(
      (M) => {
        l == null || l(M), s == null || s(M), M.defaultPrevented || c == null || c();
      },
      {
        ownerDocument: C,
        deferPointerDownOutside: i,
        isDeferredPointerDownOutsideRef: $,
        dismissableSurfaces: h.dismissableSurfaces,
        shouldHandlePointerDownOutside: m.useCallback(
          (M) => {
            if (!(M instanceof Node))
              return !1;
            const ke = [...h.branches].some(
              (ve) => ve.contains(M)
            );
            return x && !ke;
          },
          [h.branches, x]
        )
      }
    ), se = nr((M) => {
      if (i && $.current)
        return;
      const ke = M.target;
      [...h.branches].some((K) => K.contains(ke)) || (u == null || u(M), s == null || s(M), M.defaultPrevented || c == null || c());
    }, C), ne = v ? O === N.length - 1 : !1, J = Pe((M) => {
      M.key === "Escape" && (o == null || o(M), !M.defaultPrevented && c && (M.preventDefault(), c()));
    });
    return m.useEffect(() => {
      if (ne)
        return C.addEventListener("keydown", J, { capture: !0 }), () => C.removeEventListener("keydown", J, { capture: !0 });
    }, [C, ne, J]), m.useEffect(() => {
      if (v)
        return r && (h.layersWithOutsidePointerEventsDisabled.size === 0 && (yn = C.body.style.pointerEvents, C.body.style.pointerEvents = "none"), h.layersWithOutsidePointerEventsDisabled.add(v)), h.layers.add(v), xt(), () => {
          r && (h.layersWithOutsidePointerEventsDisabled.delete(v), h.layersWithOutsidePointerEventsDisabled.size === 0 && (C.body.style.pointerEvents = yn));
        };
    }, [v, C, r, h]), m.useEffect(() => () => {
      v && (h.layers.delete(v), h.layersWithOutsidePointerEventsDisabled.delete(v), xt());
    }, [v, h]), m.useEffect(() => {
      const M = /* @__PURE__ */ I(() => b({}), "handleUpdate");
      return document.addEventListener(Tt, M), () => document.removeEventListener(Tt, M);
    }, []), /* @__PURE__ */ a(
      De.div,
      {
        ...f,
        ref: p,
        style: {
          pointerEvents: D ? x ? "auto" : "none" : void 0,
          ...t.style
        },
        onFocusCapture: me(t.onFocusCapture, se.onFocusCapture),
        onBlurCapture: me(t.onBlurCapture, se.onBlurCapture),
        onPointerDownCapture: me(
          t.onPointerDownCapture,
          he.onPointerDownCapture
        )
      }
    );
  }, "DismissableLayer")
);
function er() {
  const e = m.useContext(Qn), [t, n] = m.useState(null);
  return m.useEffect(() => {
    if (t)
      return e.dismissableSurfaces.add(t), () => {
        e.dismissableSurfaces.delete(t);
      };
  }, [t, e.dismissableSurfaces]), n;
}
I(er, "useDismissableLayerSurface");
var Ti = /* @__PURE__ */ I(() => !0, "IS_TRUE");
function tr(e, t) {
  const {
    ownerDocument: n = globalThis == null ? void 0 : globalThis.document,
    deferPointerDownOutside: r = !1,
    isDeferredPointerDownOutsideRef: i,
    dismissableSurfaces: o,
    shouldHandlePointerDownOutside: l = Ti
  } = t, u = Pe(e), s = m.useRef(!1), c = m.useRef(!1), f = m.useRef(/* @__PURE__ */ new Map()), h = m.useRef(() => {
  });
  return m.useEffect(() => {
    function v() {
      c.current = !1, i.current = !1, f.current.clear();
    }
    I(v, "resetOutsideInteraction");
    function E() {
      return Array.from(f.current.values()).some(Boolean);
    }
    I(E, "isOutsideInteractionIntercepted");
    function C(P) {
      if (!c.current)
        return;
      const O = P.target;
      O instanceof Node && [...o].some((x) => x.contains(O)) || f.current.set(P.type, !0), P.type === "click" && window.setTimeout(() => {
        c.current && h.current();
      }, 0);
    }
    I(C, "handleInteractionCapture");
    function b(P) {
      c.current && f.current.set(P.type, !1);
    }
    I(b, "handleInteractionBubble");
    const p = /* @__PURE__ */ I((P) => {
      if (P.target && !s.current) {
        let O = function() {
          n.removeEventListener("click", h.current);
          const x = E();
          v(), x || Kt(
            Oi,
            u,
            D,
            { discrete: !0 }
          );
        };
        if (I(O, "handleAndDispatchPointerDownOutsideEvent"), !l(P.target)) {
          n.removeEventListener("click", h.current), v(), s.current = !1;
          return;
        }
        const D = { originalEvent: P };
        c.current = !0, i.current = r && P.button === 0, f.current.clear(), !r || P.button !== 0 ? O() : (n.removeEventListener("click", h.current), h.current = O, n.addEventListener("click", h.current, { once: !0 }));
      } else
        n.removeEventListener("click", h.current), v();
      s.current = !1;
    }, "handlePointerDown"), N = [
      "pointerup",
      "mousedown",
      "mouseup",
      "touchstart",
      "touchend",
      "click"
    ];
    for (const P of N)
      n.addEventListener(P, C, !0), n.addEventListener(P, b);
    const A = window.setTimeout(() => {
      n.addEventListener("pointerdown", p);
    }, 0);
    return () => {
      window.clearTimeout(A), n.removeEventListener("pointerdown", p), n.removeEventListener("click", h.current);
      for (const P of N)
        n.removeEventListener(P, C, !0), n.removeEventListener(P, b);
    };
  }, [
    n,
    u,
    r,
    i,
    o,
    l
  ]), {
    // ensures we check React component tree (not just DOM tree)
    onPointerDownCapture: /* @__PURE__ */ I(() => s.current = !0, "onPointerDownCapture")
  };
}
I(tr, "usePointerDownOutside");
function nr(e, t = globalThis == null ? void 0 : globalThis.document) {
  const n = Pe(e), r = m.useRef(!1);
  return m.useEffect(() => {
    const i = /* @__PURE__ */ I((o) => {
      o.target && !r.current && Kt(Mi, n, { originalEvent: o }, {
        discrete: !1
      });
    }, "handleFocus");
    return t.addEventListener("focusin", i), () => t.removeEventListener("focusin", i);
  }, [t, n]), {
    onFocusCapture: /* @__PURE__ */ I(() => r.current = !0, "onFocusCapture"),
    onBlurCapture: /* @__PURE__ */ I(() => r.current = !1, "onBlurCapture")
  };
}
I(nr, "useFocusOutside");
function xt() {
  const e = new CustomEvent(Tt);
  document.dispatchEvent(e);
}
I(xt, "dispatchUpdate");
function Kt(e, t, n, { discrete: r }) {
  const i = n.originalEvent.target, o = new CustomEvent(e, { bubbles: !1, cancelable: !0, detail: n });
  t && i.addEventListener(e, t, { once: !0 }), r ? Zn(i, o) : i.dispatchEvent(o);
}
I(Kt, "handleAndDispatchCustomEvent");
var xi = Object.defineProperty, W = (e, t) => xi(e, "name", { value: t, configurable: !0 }), Ct = "focusScope.autoFocusOnMount", Et = "focusScope.autoFocusOnUnmount", bn = { bubbles: !1, cancelable: !0 }, Li = /* @__PURE__ */ m.forwardRef(
  /* @__PURE__ */ W(function(t, n) {
    const {
      loop: r = !1,
      trapped: i = !1,
      onMountAutoFocus: o,
      onUnmountAutoFocus: l,
      ...u
    } = t, [s, c] = m.useState(null), f = Pe(o), h = Pe(l), v = m.useRef(null), E = Ae(n, c), C = m.useRef({
      paused: !1,
      pause() {
        this.paused = !0;
      },
      resume() {
        this.paused = !1;
      }
    }).current;
    m.useEffect(() => {
      if (i) {
        let p = function(O) {
          if (C.paused || !s) return;
          const D = O.target;
          s.contains(D) ? v.current = D : ie(v.current, { select: !0 });
        }, N = function(O) {
          if (C.paused || !s) return;
          const D = O.relatedTarget;
          D !== null && (s.contains(D) || ie(v.current, { select: !0 }));
        }, A = function(O) {
          if (document.activeElement === document.body)
            for (const x of O)
              x.removedNodes.length > 0 && ie(s);
        };
        W(p, "handleFocusIn"), W(N, "handleFocusOut"), W(A, "handleMutations"), document.addEventListener("focusin", p), document.addEventListener("focusout", N);
        const P = new MutationObserver(A);
        return s && P.observe(s, { childList: !0, subtree: !0 }), () => {
          document.removeEventListener("focusin", p), document.removeEventListener("focusout", N), P.disconnect();
        };
      }
    }, [i, s, C.paused]), m.useEffect(() => {
      if (s) {
        Sn.add(C);
        const p = document.activeElement;
        if (!s.contains(p)) {
          const A = new CustomEvent(Ct, bn);
          s.addEventListener(Ct, f), s.dispatchEvent(A), A.defaultPrevented || (rr(cr(Xt(s)), { select: !0 }), document.activeElement === p && ie(s));
        }
        return () => {
          s.removeEventListener(Ct, f), setTimeout(() => {
            const A = new CustomEvent(Et, bn);
            s.addEventListener(Et, h), s.dispatchEvent(A), A.defaultPrevented || ie(p ?? document.body, { select: !0 }), s.removeEventListener(Et, h), Sn.remove(C);
          }, 0);
        };
      }
    }, [s, f, h, C]);
    const b = m.useCallback(
      (p) => {
        if (!r && !i || C.paused) return;
        const N = p.key === "Tab" && !p.altKey && !p.ctrlKey && !p.metaKey, A = document.activeElement;
        if (N && A) {
          const P = p.currentTarget, [O, D] = ir(P);
          O && D ? !p.shiftKey && A === D ? (p.preventDefault(), r && ie(O, { select: !0 })) : p.shiftKey && A === O && (p.preventDefault(), r && ie(D, { select: !0 })) : A === P && p.preventDefault();
        }
      },
      [r, i, C.paused]
    );
    return /* @__PURE__ */ a(De.div, { tabIndex: -1, ...u, ref: E, onKeyDown: b });
  }, "FocusScope")
);
function rr(e, { select: t = !1 } = {}) {
  const n = document.activeElement;
  for (const r of e)
    if (ie(r, { select: t }), document.activeElement !== n) return;
}
W(rr, "focusFirst");
function ir(e) {
  const t = Xt(e), n = Lt(t, e), r = Lt(t.reverse(), e);
  return [n, r];
}
W(ir, "getTabbableEdges");
function Xt(e) {
  const t = [], n = document.createTreeWalker(e, NodeFilter.SHOW_ELEMENT, {
    acceptNode: /* @__PURE__ */ W((r) => {
      const i = r.tagName === "INPUT" && r.type === "hidden";
      return r.disabled || r.hidden || i ? NodeFilter.FILTER_SKIP : r.tabIndex >= 0 ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_SKIP;
    }, "acceptNode")
  });
  for (; n.nextNode(); ) t.push(n.currentNode);
  return t;
}
W(Xt, "getTabbableCandidates");
function Lt(e, t) {
  const n = typeof t.checkVisibility == "function" && t.checkVisibility({ checkVisibilityCSS: !0 });
  for (const r of e)
    if (!(n ? !r.checkVisibility({ checkVisibilityCSS: !0 }) : ar(r, { upTo: t })))
      return r;
}
W(Lt, "findVisible");
function ar(e, { upTo: t }) {
  if (getComputedStyle(e).visibility === "hidden") return !0;
  for (; e; ) {
    if (t !== void 0 && e === t) return !1;
    if (getComputedStyle(e).display === "none") return !0;
    e = e.parentElement;
  }
  return !1;
}
W(ar, "isHidden");
function or(e) {
  return e instanceof HTMLInputElement && "select" in e;
}
W(or, "isSelectableInput");
function ie(e, { select: t = !1 } = {}) {
  if (e && e.focus) {
    const n = document.activeElement;
    e.focus({ preventScroll: !0 }), e !== n && or(e) && t && e.select();
  }
}
W(ie, "focus");
var Sn = sr();
function sr() {
  let e = [];
  return {
    add(t) {
      const n = e[0];
      t !== n && (n == null || n.pause()), e = It(e, t), e.unshift(t);
    },
    remove(t) {
      var n;
      e = It(e, t), (n = e[0]) == null || n.resume();
    }
  };
}
W(sr, "createFocusScopesStack");
function It(e, t) {
  const n = [...e], r = n.indexOf(t);
  return r !== -1 && n.splice(r, 1), n;
}
W(It, "arrayRemove");
function cr(e) {
  return e.filter((t) => t.tagName !== "A");
}
W(cr, "removeLinks");
var Ii = Object.defineProperty, Fi = (e, t) => Ii(e, "name", { value: t, configurable: !0 }), Bi = /* @__PURE__ */ m.forwardRef(
  /* @__PURE__ */ Fi(function(t, n) {
    var s;
    const { container: r, ...i } = t, [o, l] = m.useState(!1);
    ue(() => l(!0), []);
    const u = r || o && ((s = globalThis == null ? void 0 : globalThis.document) == null ? void 0 : s.body);
    return u ? Xn.createPortal(/* @__PURE__ */ a(De.div, { ...i, ref: n }), u) : null;
  }, "Portal")
), Ui = Object.defineProperty, ae = (e, t) => Ui(e, "name", { value: t, configurable: !0 });
function lr(e, t) {
  return m.useReducer((n, r) => t[n][r] ?? n, e);
}
ae(lr, "useStateMachine");
var Ht = /* @__PURE__ */ ae((e) => {
  const { present: t, children: n } = e, r = ur(t), i = typeof n == "function" ? n({ present: r.isPresent }) : m.Children.only(n), o = dr(r.ref, fr(i));
  return typeof n == "function" || r.isPresent ? m.cloneElement(i, { ref: o }) : null;
}, "Presence");
function ur(e) {
  const [t, n] = m.useState(), r = m.useRef(null), i = m.useRef(e), o = m.useRef("none"), l = m.useRef(void 0), u = e ? "mounted" : "unmounted", [s, c] = lr(u, {
    mounted: {
      UNMOUNT: "unmounted",
      ANIMATION_OUT: "unmountSuspended"
    },
    unmountSuspended: {
      MOUNT: "mounted",
      ANIMATION_END: "unmounted"
    },
    unmounted: {
      MOUNT: "mounted"
    }
  });
  return m.useEffect(() => {
    s === "mounted" ? (o.current = l.current ?? Ee(r.current), l.current = void 0) : o.current = "none";
  }, [s]), ue(() => {
    const f = r.current, h = i.current;
    if (h !== e) {
      const E = o.current, C = Ee(f);
      e ? (l.current = C, c("MOUNT")) : C === "none" || (f == null ? void 0 : f.display) === "none" ? c("UNMOUNT") : c(h && E !== C ? "ANIMATION_OUT" : "UNMOUNT"), i.current = e;
    }
  }, [e, c]), ue(() => {
    if (t) {
      let f;
      const h = t.ownerDocument.defaultView ?? window, v = /* @__PURE__ */ ae((C) => {
        const p = Ee(r.current).includes(CSS.escape(C.animationName));
        if (C.target === t && p && (c("ANIMATION_END"), !i.current)) {
          const N = t.style.animationFillMode;
          t.style.animationFillMode = "forwards", f = h.setTimeout(() => {
            t.style.animationFillMode === "forwards" && (t.style.animationFillMode = N);
          });
        }
      }, "handleAnimationEnd"), E = /* @__PURE__ */ ae((C) => {
        C.target === t && (o.current = Ee(r.current));
      }, "handleAnimationStart");
      return t.addEventListener("animationstart", E), t.addEventListener("animationcancel", v), t.addEventListener("animationend", v), () => {
        h.clearTimeout(f), t.removeEventListener("animationstart", E), t.removeEventListener("animationcancel", v), t.removeEventListener("animationend", v);
      };
    } else
      c("ANIMATION_END");
  }, [t, c]), {
    isPresent: ["mounted", "unmountSuspended"].includes(s),
    ref: m.useCallback((f) => {
      if (f) {
        const h = getComputedStyle(f);
        r.current = h, l.current = Ee(h);
      } else
        r.current = null;
      n(f);
    }, [])
  };
}
ae(ur, "usePresence");
function Ft(e, t) {
  if (typeof e == "function")
    return e(t);
  e != null && (e.current = t);
}
ae(Ft, "setRef");
function dr(...e) {
  const t = m.useRef(e);
  return t.current = e, m.useCallback((n) => {
    const r = t.current;
    let i = !1;
    const o = r.map((l) => {
      const u = Ft(l, n);
      return !i && typeof u == "function" && (i = !0), u;
    });
    if (i)
      return () => {
        for (let l = 0; l < o.length; l++) {
          const u = o[l];
          typeof u == "function" ? u() : Ft(r[l], null);
        }
      };
  }, []);
}
ae(dr, "useStableComposedRefs");
function Ee(e) {
  return (e == null ? void 0 : e.animationName) || "none";
}
ae(Ee, "getAnimationName");
function fr(e) {
  var r, i;
  let t = (r = Object.getOwnPropertyDescriptor(e.props, "ref")) == null ? void 0 : r.get, n = t && "isReactWarning" in t && t.isReactWarning;
  return n ? e.ref : (t = (i = Object.getOwnPropertyDescriptor(e, "ref")) == null ? void 0 : i.get, n = t && "isReactWarning" in t && t.isReactWarning, n ? e.props.ref : e.props.ref || e.ref);
}
ae(fr, "getElementRef");
var Wi = Object.defineProperty, Vt = (e, t) => Wi(e, "name", { value: t, configurable: !0 }), Ze = 0, ee = null;
function $i(e) {
  return Yt(), e.children;
}
Vt($i, "FocusGuards");
function Yt() {
  m.useEffect(() => {
    ee || (ee = { start: Bt(), end: Bt() });
    const { start: e, end: t } = ee;
    return document.body.firstElementChild !== e && document.body.insertAdjacentElement("afterbegin", e), document.body.lastElementChild !== t && document.body.insertAdjacentElement("beforeend", t), Ze++, () => {
      Ze === 1 && (ee == null || ee.start.remove(), ee == null || ee.end.remove(), ee = null), Ze = Math.max(0, Ze - 1);
    };
  }, []);
}
Vt(Yt, "useFocusGuards");
function Bt() {
  const e = document.createElement("span");
  return e.setAttribute("data-radix-focus-guard", ""), e.tabIndex = 0, e.style.outline = "none", e.style.opacity = "0", e.style.position = "fixed", e.style.pointerEvents = "none", e;
}
Vt(Bt, "createFocusGuard");
var te = function() {
  return te = Object.assign || function(t) {
    for (var n, r = 1, i = arguments.length; r < i; r++) {
      n = arguments[r];
      for (var o in n) Object.prototype.hasOwnProperty.call(n, o) && (t[o] = n[o]);
    }
    return t;
  }, te.apply(this, arguments);
};
function mr(e, t) {
  var n = {};
  for (var r in e) Object.prototype.hasOwnProperty.call(e, r) && t.indexOf(r) < 0 && (n[r] = e[r]);
  if (e != null && typeof Object.getOwnPropertySymbols == "function")
    for (var i = 0, r = Object.getOwnPropertySymbols(e); i < r.length; i++)
      t.indexOf(r[i]) < 0 && Object.prototype.propertyIsEnumerable.call(e, r[i]) && (n[r[i]] = e[r[i]]);
  return n;
}
function qi(e, t, n) {
  if (n || arguments.length === 2) for (var r = 0, i = t.length, o; r < i; r++)
    (o || !(r in t)) && (o || (o = Array.prototype.slice.call(t, 0, r)), o[r] = t[r]);
  return e.concat(o || Array.prototype.slice.call(t));
}
var it = "right-scroll-bar-position", at = "width-before-scroll-bar", ji = "with-scroll-bars-hidden", zi = "--removed-body-scroll-bar-size";
function Nt(e, t) {
  return typeof e == "function" ? e(t) : e && (e.current = t), e;
}
function Ki(e, t) {
  var n = T(function() {
    return {
      // value
      value: e,
      // last callback
      callback: t,
      // "memoized" public interface
      facade: {
        get current() {
          return n.value;
        },
        set current(r) {
          var i = n.value;
          i !== r && (n.value = r, n.callback(r, i));
        }
      }
    };
  })[0];
  return n.callback = t, n.facade;
}
var Xi = typeof window < "u" ? m.useLayoutEffect : m.useEffect, wn = /* @__PURE__ */ new WeakMap();
function Hi(e, t) {
  var n = Ki(null, function(r) {
    return e.forEach(function(i) {
      return Nt(i, r);
    });
  });
  return Xi(function() {
    var r = wn.get(n);
    if (r) {
      var i = new Set(r), o = new Set(e), l = n.current;
      i.forEach(function(u) {
        o.has(u) || Nt(u, null);
      }), o.forEach(function(u) {
        i.has(u) || Nt(u, l);
      });
    }
    wn.set(n, e);
  }, [e]), n;
}
function Vi(e) {
  return e;
}
function Yi(e, t) {
  t === void 0 && (t = Vi);
  var n = [], r = !1, i = {
    read: function() {
      if (r)
        throw new Error("Sidecar: could not `read` from an `assigned` medium. `read` could be used only with `useMedium`.");
      return n.length ? n[n.length - 1] : e;
    },
    useMedium: function(o) {
      var l = t(o, r);
      return n.push(l), function() {
        n = n.filter(function(u) {
          return u !== l;
        });
      };
    },
    assignSyncMedium: function(o) {
      for (r = !0; n.length; ) {
        var l = n;
        n = [], l.forEach(o);
      }
      n = {
        push: function(u) {
          return o(u);
        },
        filter: function() {
          return n;
        }
      };
    },
    assignMedium: function(o) {
      r = !0;
      var l = [];
      if (n.length) {
        var u = n;
        n = [], u.forEach(o), l = n;
      }
      var s = function() {
        var f = l;
        l = [], f.forEach(o);
      }, c = function() {
        return Promise.resolve().then(s);
      };
      c(), n = {
        push: function(f) {
          l.push(f), c();
        },
        filter: function(f) {
          return l = l.filter(f), n;
        }
      };
    }
  };
  return i;
}
function Gi(e) {
  e === void 0 && (e = {});
  var t = Yi(null);
  return t.options = te({ async: !0, ssr: !1 }, e), t;
}
var hr = function(e) {
  var t = e.sideCar, n = mr(e, ["sideCar"]);
  if (!t)
    throw new Error("Sidecar: please provide `sideCar` property to import the right car");
  var r = t.read();
  if (!r)
    throw new Error("Sidecar medium not found");
  return m.createElement(r, te({}, n));
};
hr.isSideCarExport = !0;
function Ji(e, t) {
  return e.useMedium(t), hr;
}
var vr = Gi(), Pt = function() {
}, ut = m.forwardRef(function(e, t) {
  var n = m.useRef(null), r = m.useState({
    onScrollCapture: Pt,
    onWheelCapture: Pt,
    onTouchMoveCapture: Pt
  }), i = r[0], o = r[1], l = e.forwardProps, u = e.children, s = e.className, c = e.removeScrollBar, f = e.enabled, h = e.shards, v = e.sideCar, E = e.noRelative, C = e.noIsolation, b = e.inert, p = e.allowPinchZoom, N = e.as, A = N === void 0 ? "div" : N, P = e.gapMode, O = mr(e, ["forwardProps", "children", "className", "removeScrollBar", "enabled", "shards", "sideCar", "noRelative", "noIsolation", "inert", "allowPinchZoom", "as", "gapMode"]), D = v, x = Hi([n, t]), $ = te(te({}, O), i);
  return m.createElement(
    m.Fragment,
    null,
    f && m.createElement(D, { sideCar: vr, removeScrollBar: c, shards: h, noRelative: E, noIsolation: C, inert: b, setCallbacks: o, allowPinchZoom: !!p, lockRef: n, gapMode: P }),
    l ? m.cloneElement(m.Children.only(u), te(te({}, $), { ref: x })) : m.createElement(A, te({}, $, { className: s, ref: x }), u)
  );
});
ut.defaultProps = {
  enabled: !0,
  removeScrollBar: !0,
  inert: !1
};
ut.classNames = {
  fullWidth: at,
  zeroRight: it
};
var Zi = function() {
  if (typeof __webpack_nonce__ < "u")
    return __webpack_nonce__;
};
function Qi() {
  if (!document)
    return null;
  var e = document.createElement("style");
  e.type = "text/css";
  var t = Zi();
  return t && e.setAttribute("nonce", t), e;
}
function ea(e, t) {
  e.styleSheet ? e.styleSheet.cssText = t : e.appendChild(document.createTextNode(t));
}
function ta(e) {
  var t = document.head || document.getElementsByTagName("head")[0];
  t.appendChild(e);
}
var na = function() {
  var e = 0, t = null;
  return {
    add: function(n) {
      e == 0 && (t = Qi()) && (ea(t, n), ta(t)), e++;
    },
    remove: function() {
      e--, !e && t && (t.parentNode && t.parentNode.removeChild(t), t = null);
    }
  };
}, ra = function() {
  var e = na();
  return function(t, n) {
    m.useEffect(function() {
      return e.add(t), function() {
        e.remove();
      };
    }, [t && n]);
  };
}, gr = function() {
  var e = ra(), t = function(n) {
    var r = n.styles, i = n.dynamic;
    return e(r, i), null;
  };
  return t;
}, ia = {
  left: 0,
  top: 0,
  right: 0,
  gap: 0
}, Rt = function(e) {
  return parseInt(e || "", 10) || 0;
}, aa = function(e) {
  var t = window.getComputedStyle(document.body), n = t[e === "padding" ? "paddingLeft" : "marginLeft"], r = t[e === "padding" ? "paddingTop" : "marginTop"], i = t[e === "padding" ? "paddingRight" : "marginRight"];
  return [Rt(n), Rt(r), Rt(i)];
}, oa = function(e) {
  if (e === void 0 && (e = "margin"), typeof window > "u")
    return ia;
  var t = aa(e), n = document.documentElement.clientWidth, r = window.innerWidth;
  return {
    left: t[0],
    top: t[1],
    right: t[2],
    gap: Math.max(0, r - n + t[2] - t[0])
  };
}, sa = gr(), Ne = "data-scroll-locked", ca = function(e, t, n, r) {
  var i = e.left, o = e.top, l = e.right, u = e.gap;
  return n === void 0 && (n = "margin"), `
  .`.concat(ji, ` {
   overflow: hidden `).concat(r, `;
   padding-right: `).concat(u, "px ").concat(r, `;
  }
  body[`).concat(Ne, `] {
    overflow: hidden `).concat(r, `;
    overscroll-behavior: contain;
    `).concat([
    t && "position: relative ".concat(r, ";"),
    n === "margin" && `
    padding-left: `.concat(i, `px;
    padding-top: `).concat(o, `px;
    padding-right: `).concat(l, `px;
    margin-left:0;
    margin-top:0;
    margin-right: `).concat(u, "px ").concat(r, `;
    `),
    n === "padding" && "padding-right: ".concat(u, "px ").concat(r, ";")
  ].filter(Boolean).join(""), `
  }

  .`).concat(it, ` {
    right: `).concat(u, "px ").concat(r, `;
  }

  .`).concat(at, ` {
    margin-right: `).concat(u, "px ").concat(r, `;
  }

  .`).concat(it, " .").concat(it, ` {
    right: 0 `).concat(r, `;
  }

  .`).concat(at, " .").concat(at, ` {
    margin-right: 0 `).concat(r, `;
  }

  body[`).concat(Ne, `] {
    `).concat(zi, ": ").concat(u, `px;
  }
`);
}, Cn = function() {
  var e = parseInt(document.body.getAttribute(Ne) || "0", 10);
  return isFinite(e) ? e : 0;
}, la = function() {
  m.useEffect(function() {
    return document.body.setAttribute(Ne, (Cn() + 1).toString()), function() {
      var e = Cn() - 1;
      e <= 0 ? document.body.removeAttribute(Ne) : document.body.setAttribute(Ne, e.toString());
    };
  }, []);
}, ua = function(e) {
  var t = e.noRelative, n = e.noImportant, r = e.gapMode, i = r === void 0 ? "margin" : r;
  la();
  var o = m.useMemo(function() {
    return oa(i);
  }, [i]);
  return m.createElement(sa, { styles: ca(o, !t, i, n ? "" : "!important") });
}, Ut = !1;
if (typeof window < "u")
  try {
    var Qe = Object.defineProperty({}, "passive", {
      get: function() {
        return Ut = !0, !0;
      }
    });
    window.addEventListener("test", Qe, Qe), window.removeEventListener("test", Qe, Qe);
  } catch {
    Ut = !1;
  }
var be = Ut ? { passive: !1 } : !1, da = function(e) {
  return e.tagName === "TEXTAREA";
}, pr = function(e, t) {
  if (!(e instanceof Element))
    return !1;
  var n = window.getComputedStyle(e);
  return (
    // not-not-scrollable
    n[t] !== "hidden" && // contains scroll inside self
    !(n.overflowY === n.overflowX && !da(e) && n[t] === "visible")
  );
}, fa = function(e) {
  return pr(e, "overflowY");
}, ma = function(e) {
  return pr(e, "overflowX");
}, En = function(e, t) {
  var n = t.ownerDocument, r = t;
  do {
    typeof ShadowRoot < "u" && r instanceof ShadowRoot && (r = r.host);
    var i = yr(e, r);
    if (i) {
      var o = br(e, r), l = o[1], u = o[2];
      if (l > u)
        return !0;
    }
    r = r.parentNode;
  } while (r && r !== n.body);
  return !1;
}, ha = function(e) {
  var t = e.scrollTop, n = e.scrollHeight, r = e.clientHeight;
  return [
    t,
    n,
    r
  ];
}, va = function(e) {
  var t = e.scrollLeft, n = e.scrollWidth, r = e.clientWidth;
  return [
    t,
    n,
    r
  ];
}, yr = function(e, t) {
  return e === "v" ? fa(t) : ma(t);
}, br = function(e, t) {
  return e === "v" ? ha(t) : va(t);
}, ga = function(e, t) {
  return e === "h" && t === "rtl" ? -1 : 1;
}, pa = function(e, t, n, r, i) {
  var o = ga(e, window.getComputedStyle(t).direction), l = o * r, u = n.target, s = t.contains(u), c = !1, f = l > 0, h = 0, v = 0;
  do {
    if (!u)
      break;
    var E = br(e, u), C = E[0], b = E[1], p = E[2], N = b - p - o * C;
    (C || N) && yr(e, u) && (h += N, v += C);
    var A = u.parentNode;
    u = A && A.nodeType === Node.DOCUMENT_FRAGMENT_NODE ? A.host : A;
  } while (
    // portaled content
    !s && u !== document.body || // self content
    s && (t.contains(u) || t === u)
  );
  return (f && Math.abs(h) < 1 || !f && Math.abs(v) < 1) && (c = !0), c;
}, et = function(e) {
  return "changedTouches" in e ? [e.changedTouches[0].clientX, e.changedTouches[0].clientY] : [0, 0];
}, Nn = function(e) {
  return [e.deltaX, e.deltaY];
}, Pn = function(e) {
  return e && "current" in e ? e.current : e;
}, ya = function(e, t) {
  return e[0] === t[0] && e[1] === t[1];
}, ba = function(e) {
  return `
  .block-interactivity-`.concat(e, ` {pointer-events: none;}
  .allow-interactivity-`).concat(e, ` {pointer-events: all;}
`);
}, Sa = 0, Se = [];
function wa(e) {
  var t = m.useRef([]), n = m.useRef([0, 0]), r = m.useRef(), i = m.useState(Sa++)[0], o = m.useState(gr)[0], l = m.useRef(e);
  m.useEffect(function() {
    l.current = e;
  }, [e]), m.useEffect(function() {
    if (e.inert) {
      document.body.classList.add("block-interactivity-".concat(i));
      var b = qi([e.lockRef.current], (e.shards || []).map(Pn), !0).filter(Boolean);
      return b.forEach(function(p) {
        return p.classList.add("allow-interactivity-".concat(i));
      }), function() {
        document.body.classList.remove("block-interactivity-".concat(i)), b.forEach(function(p) {
          return p.classList.remove("allow-interactivity-".concat(i));
        });
      };
    }
  }, [e.inert, e.lockRef.current, e.shards]);
  var u = m.useCallback(function(b, p) {
    if ("touches" in b && b.touches.length === 2 || b.type === "wheel" && b.ctrlKey)
      return !l.current.allowPinchZoom;
    var N = et(b), A = n.current, P = "deltaX" in b ? b.deltaX : A[0] - N[0], O = "deltaY" in b ? b.deltaY : A[1] - N[1], D, x = b.target, $ = Math.abs(P) > Math.abs(O) ? "h" : "v";
    if ("touches" in b && $ === "h" && x.type === "range")
      return !1;
    var he = window.getSelection(), se = he && he.anchorNode, ne = se ? se === x || se.contains(x) : !1;
    if (ne)
      return !1;
    var J = En($, x);
    if (!J)
      return !0;
    if (J ? D = $ : (D = $ === "v" ? "h" : "v", J = En($, x)), !J)
      return !1;
    if (!r.current && "changedTouches" in b && (P || O) && (r.current = D), !D)
      return !0;
    var M = r.current || D;
    return pa(M, p, b, M === "h" ? P : O);
  }, []), s = m.useCallback(function(b) {
    var p = b;
    if (!(!Se.length || Se[Se.length - 1] !== o)) {
      var N = "deltaY" in p ? Nn(p) : et(p), A = t.current.filter(function(D) {
        return D.name === p.type && (D.target === p.target || p.target === D.shadowParent) && ya(D.delta, N);
      })[0];
      if (A && A.should) {
        p.cancelable && p.preventDefault();
        return;
      }
      if (!A) {
        var P = (l.current.shards || []).map(Pn).filter(Boolean).filter(function(D) {
          return D.contains(p.target);
        }), O = P.length > 0 ? u(p, P[0]) : !l.current.noIsolation;
        O && p.cancelable && p.preventDefault();
      }
    }
  }, []), c = m.useCallback(function(b, p, N, A) {
    var P = { name: b, delta: p, target: N, should: A, shadowParent: Ca(N) };
    t.current.push(P), setTimeout(function() {
      t.current = t.current.filter(function(O) {
        return O !== P;
      });
    }, 1);
  }, []), f = m.useCallback(function(b) {
    n.current = et(b), r.current = void 0;
  }, []), h = m.useCallback(function(b) {
    c(b.type, Nn(b), b.target, u(b, e.lockRef.current));
  }, []), v = m.useCallback(function(b) {
    c(b.type, et(b), b.target, u(b, e.lockRef.current));
  }, []);
  m.useEffect(function() {
    return Se.push(o), e.setCallbacks({
      onScrollCapture: h,
      onWheelCapture: h,
      onTouchMoveCapture: v
    }), document.addEventListener("wheel", s, be), document.addEventListener("touchmove", s, be), document.addEventListener("touchstart", f, be), function() {
      Se = Se.filter(function(b) {
        return b !== o;
      }), document.removeEventListener("wheel", s, be), document.removeEventListener("touchmove", s, be), document.removeEventListener("touchstart", f, be);
    };
  }, []);
  var E = e.removeScrollBar, C = e.inert;
  return m.createElement(
    m.Fragment,
    null,
    C ? m.createElement(o, { styles: ba(i) }) : null,
    E ? m.createElement(ua, { noRelative: e.noRelative, gapMode: e.gapMode }) : null
  );
}
function Ca(e) {
  for (var t = null; e !== null; )
    e instanceof ShadowRoot && (t = e.host, e = e.host), e = e.parentNode;
  return t;
}
const Ea = Ji(vr, wa);
var Sr = m.forwardRef(function(e, t) {
  return m.createElement(ut, te({}, e, { ref: t, sideCar: Ea }));
});
Sr.classNames = ut.classNames;
var Na = function(e) {
  if (typeof document > "u")
    return null;
  var t = Array.isArray(e) ? e[0] : e;
  return t.ownerDocument.body;
}, we = /* @__PURE__ */ new WeakMap(), tt = /* @__PURE__ */ new WeakMap(), nt = {}, At = 0, wr = function(e) {
  return e && (e.host || wr(e.parentNode));
}, Pa = function(e, t) {
  return t.map(function(n) {
    if (e.contains(n))
      return n;
    var r = wr(n);
    return r && e.contains(r) ? r : (console.error("aria-hidden", n, "in not contained inside", e, ". Doing nothing"), null);
  }).filter(function(n) {
    return !!n;
  });
}, Ra = function(e, t, n, r) {
  var i = Pa(t, Array.isArray(e) ? e : [e]);
  nt[n] || (nt[n] = /* @__PURE__ */ new WeakMap());
  var o = nt[n], l = [], u = /* @__PURE__ */ new Set(), s = new Set(i), c = function(h) {
    !h || u.has(h) || (u.add(h), c(h.parentNode));
  };
  i.forEach(c);
  var f = function(h) {
    !h || s.has(h) || Array.prototype.forEach.call(h.children, function(v) {
      if (u.has(v))
        f(v);
      else
        try {
          var E = v.getAttribute(r), C = E !== null && E !== "false", b = (we.get(v) || 0) + 1, p = (o.get(v) || 0) + 1;
          we.set(v, b), o.set(v, p), l.push(v), b === 1 && C && tt.set(v, !0), p === 1 && v.setAttribute(n, "true"), C || v.setAttribute(r, "true");
        } catch (N) {
          console.error("aria-hidden: cannot operate on ", v, N);
        }
    });
  };
  return f(t), u.clear(), At++, function() {
    l.forEach(function(h) {
      var v = we.get(h) - 1, E = o.get(h) - 1;
      we.set(h, v), o.set(h, E), v || (tt.has(h) || h.removeAttribute(r), tt.delete(h)), E || h.removeAttribute(n);
    }), At--, At || (we = /* @__PURE__ */ new WeakMap(), we = /* @__PURE__ */ new WeakMap(), tt = /* @__PURE__ */ new WeakMap(), nt = {});
  };
}, Aa = function(e, t, n) {
  n === void 0 && (n = "data-aria-hidden");
  var r = Array.from(Array.isArray(e) ? e : [e]), i = Na(e);
  return i ? (r.push.apply(r, Array.from(i.querySelectorAll("[aria-live], script"))), Ra(r, i, n, "aria-hidden")) : function() {
    return null;
  };
}, Da = Object.defineProperty, G = (e, t) => Da(e, "name", { value: t, configurable: !0 }), Gt = "Dialog", [Cr, xo] = /* @__PURE__ */ Wn(Gt), [ka, oe] = Cr(Gt), Oa = /* @__PURE__ */ G((e) => {
  const {
    __scopeDialog: t,
    children: n,
    open: r,
    defaultOpen: i,
    onOpenChange: o,
    modal: l = !0
  } = e, u = m.useRef(null), s = m.useRef(null), [c, f] = jn({
    prop: r,
    defaultProp: i ?? !1,
    onChange: o,
    caller: Gt
  }), [h, v] = m.useState(0), [E, C] = m.useState(0);
  return /* @__PURE__ */ a(
    ka,
    {
      scope: t,
      triggerRef: u,
      contentRef: s,
      contentId: rt(),
      titleId: rt(),
      descriptionId: rt(),
      titlePresent: h > 0,
      descriptionPresent: E > 0,
      setTitleCount: v,
      setDescriptionCount: C,
      open: c,
      onOpenChange: f,
      onOpenToggle: m.useCallback(() => f((b) => !b), [f]),
      modal: l,
      children: n
    }
  );
}, "Dialog"), Er = "DialogPortal", [Ma, Nr] = Cr(Er, {
  forceMount: void 0
}), _a = /* @__PURE__ */ G((e) => {
  const { __scopeDialog: t, forceMount: n, children: r, container: i } = e, o = oe(Er, t);
  return /* @__PURE__ */ a(Ma, { scope: t, forceMount: n, children: m.Children.map(r, (l) => /* @__PURE__ */ a(Ht, { present: n || o.open, children: /* @__PURE__ */ a(Bi, { asChild: !0, container: i, children: l }) })) });
}, "DialogPortal"), Wt = "DialogOverlay", Ta = /* @__PURE__ */ m.forwardRef(
  /* @__PURE__ */ G(function(t, n) {
    const r = Nr(Wt, t.__scopeDialog), { forceMount: i = r.forceMount, ...o } = t, l = oe(Wt, t.__scopeDialog);
    return l.modal ? /* @__PURE__ */ a(Ht, { present: i || l.open, children: /* @__PURE__ */ a(La, { ...o, ref: n }) }) : null;
  }, "DialogOverlay")
), xa = /* @__PURE__ */ zt("DialogOverlay.RemoveScroll"), La = /* @__PURE__ */ m.forwardRef(
  // blank line to reduce diff noise
  /* @__PURE__ */ G(function(t, n) {
    const { __scopeDialog: r, ...i } = t, o = oe(Wt, r), l = er(), u = Ae(n, l);
    return (
      // Make sure `Content` is scrollable even when it doesn't live inside `RemoveScroll`
      // ie. when `Overlay` and `Content` are siblings
      /* @__PURE__ */ a(Sr, { as: xa, allowPinchZoom: !0, shards: [o.contentRef], children: /* @__PURE__ */ a(
        De.div,
        {
          "data-state": Jt(o.open),
          ...i,
          ref: u,
          style: { pointerEvents: "auto", ...i.style }
        }
      ) })
    );
  }, "DialogOverlayImpl")
), Ue = "DialogContent", Ia = /* @__PURE__ */ m.forwardRef(
  /* @__PURE__ */ G(function(t, n) {
    const r = Nr(Ue, t.__scopeDialog), { forceMount: i = r.forceMount, ...o } = t, l = oe(Ue, t.__scopeDialog);
    return /* @__PURE__ */ a(Ht, { present: i || l.open, children: l.modal ? /* @__PURE__ */ a(Fa, { ...o, ref: n }) : /* @__PURE__ */ a(Ba, { ...o, ref: n }) });
  }, "DialogContent")
), Fa = /* @__PURE__ */ m.forwardRef(
  // blank line to reduce diff noise
  /* @__PURE__ */ G(function(t, n) {
    const r = oe(Ue, t.__scopeDialog), i = m.useRef(null), o = Ae(n, r.contentRef, i);
    return m.useEffect(() => {
      const l = i.current;
      if (l) return Aa(l);
    }, []), /* @__PURE__ */ a(
      Pr,
      {
        ...t,
        ref: o,
        trapFocus: r.open,
        disableOutsidePointerEvents: r.open,
        onCloseAutoFocus: me(t.onCloseAutoFocus, (l) => {
          var u;
          l.preventDefault(), (u = r.triggerRef.current) == null || u.focus();
        }),
        onPointerDownOutside: me(t.onPointerDownOutside, (l) => {
          const u = l.detail.originalEvent, s = u.button === 0 && u.ctrlKey === !0;
          (u.button === 2 || s) && l.preventDefault();
        }),
        onFocusOutside: me(
          t.onFocusOutside,
          (l) => l.preventDefault()
        )
      }
    );
  }, "DialogContentModal")
), Ba = /* @__PURE__ */ m.forwardRef(
  // blank line to reduce diff noise
  /* @__PURE__ */ G(function(t, n) {
    const r = oe(Ue, t.__scopeDialog), i = m.useRef(!1), o = m.useRef(!1);
    return /* @__PURE__ */ a(
      Pr,
      {
        ...t,
        ref: n,
        trapFocus: !1,
        disableOutsidePointerEvents: !1,
        onCloseAutoFocus: (l) => {
          var u, s;
          (u = t.onCloseAutoFocus) == null || u.call(t, l), l.defaultPrevented || (i.current || (s = r.triggerRef.current) == null || s.focus(), l.preventDefault()), i.current = !1, o.current = !1;
        },
        onInteractOutside: (l) => {
          var c, f;
          (c = t.onInteractOutside) == null || c.call(t, l), l.defaultPrevented || (i.current = !0, l.detail.originalEvent.type === "pointerdown" && (o.current = !0));
          const u = l.target;
          ((f = r.triggerRef.current) == null ? void 0 : f.contains(u)) && l.preventDefault(), l.detail.originalEvent.type === "focusin" && o.current && l.preventDefault();
        }
      }
    );
  }, "DialogContentNonModal")
), Pr = /* @__PURE__ */ m.forwardRef(
  // blank line to reduce diff noise
  /* @__PURE__ */ G(function(t, n) {
    const { __scopeDialog: r, trapFocus: i, onOpenAutoFocus: o, onCloseAutoFocus: l, ...u } = t, s = oe(Ue, r);
    return Yt(), /* @__PURE__ */ a(fe, { children: /* @__PURE__ */ a(
      Li,
      {
        asChild: !0,
        loop: !0,
        trapped: i,
        onMountAutoFocus: o,
        onUnmountAutoFocus: l,
        children: /* @__PURE__ */ a(
          _i,
          {
            role: "dialog",
            id: s.contentId,
            "aria-describedby": s.descriptionPresent ? s.descriptionId : void 0,
            "aria-labelledby": s.titlePresent ? s.titleId : void 0,
            "data-state": Jt(s.open),
            ...u,
            ref: n,
            deferPointerDownOutside: !0,
            onDismiss: () => s.onOpenChange(!1)
          }
        )
      }
    ) });
  }, "DialogContentImpl")
), Ua = "DialogTitle", Wa = /* @__PURE__ */ m.forwardRef(
  /* @__PURE__ */ G(function(t, n) {
    const { __scopeDialog: r, ...i } = t, o = oe(Ua, r), { setTitleCount: l } = o;
    return ue(() => (l((u) => u + 1), () => l((u) => u - 1)), [l]), /* @__PURE__ */ a(De.h2, { id: o.titleId, ...i, ref: n });
  }, "DialogTitle")
), $a = "DialogDescription", qa = /* @__PURE__ */ m.forwardRef(
  // blank line to reduce diff noise
  /* @__PURE__ */ G(function(t, n) {
    const { __scopeDialog: r, ...i } = t, o = oe($a, r), { setDescriptionCount: l } = o;
    return ue(() => (l((u) => u + 1), () => l((u) => u - 1)), [l]), /* @__PURE__ */ a(De.p, { id: o.descriptionId, ...i, ref: n });
  }, "DialogDescription")
);
function Jt(e) {
  return e ? "open" : "closed";
}
G(Jt, "getState");
/**
 * @license lucide-react v1.28.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const Rr = (...e) => e.filter((t, n, r) => !!t && t.trim() !== "" && r.indexOf(t) === n).join(" ").trim();
/**
 * @license lucide-react v1.28.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const ja = (e) => e.replace(/([a-z0-9])([A-Z])/g, "$1-$2").toLowerCase();
/**
 * @license lucide-react v1.28.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const za = (e) => e.replace(
  /^([A-Z])|[\s-_]+(\w)/g,
  (t, n, r) => r ? r.toUpperCase() : n.toLowerCase()
);
/**
 * @license lucide-react v1.28.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const Rn = (e) => {
  const t = za(e);
  return t.charAt(0).toUpperCase() + t.slice(1);
};
/**
 * @license lucide-react v1.28.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
var Dt = {
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
const Ka = (e) => {
  for (const t in e)
    if (t.startsWith("aria-") || t === "role" || t === "title")
      return !0;
  return !1;
}, Xa = Zr({}), Ha = () => Jr(Xa), Va = xn(
  ({ color: e, size: t, strokeWidth: n, absoluteStrokeWidth: r, className: i = "", children: o, iconNode: l, ...u }, s) => {
    const {
      size: c = 24,
      strokeWidth: f = 2,
      absoluteStrokeWidth: h = !1,
      color: v = "currentColor",
      className: E = ""
    } = Ha() ?? {}, C = r ?? h ? Number(n ?? f) * 24 / Number(t ?? c) : n ?? f;
    return kt(
      "svg",
      {
        ref: s,
        ...Dt,
        width: t ?? c ?? Dt.width,
        height: t ?? c ?? Dt.height,
        stroke: e ?? v,
        strokeWidth: C,
        className: Rr("lucide", E, i),
        ...!o && !Ka(u) && { "aria-hidden": "true" },
        ...u
      },
      [
        ...l.map(([b, p]) => kt(b, p)),
        ...Array.isArray(o) ? o : [o]
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
const dt = (e, t) => {
  const n = xn(
    ({ className: r, ...i }, o) => kt(Va, {
      ref: o,
      iconNode: t,
      className: Rr(
        `lucide-${ja(Rn(e))}`,
        `lucide-${e}`,
        r
      ),
      ...i
    })
  );
  return n.displayName = Rn(e), n;
};
/**
 * @license lucide-react v1.28.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const Ya = [["path", { d: "m6 9 6 6 6-6", key: "qrunsl" }]], Ga = dt("chevron-down", Ya);
/**
 * @license lucide-react v1.28.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const Ja = [["path", { d: "m15 18-6-6 6-6", key: "1wnfg3" }]], Za = dt("chevron-left", Ja);
/**
 * @license lucide-react v1.28.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const Qa = [["path", { d: "m9 18 6-6-6-6", key: "mthhwq" }]], eo = dt("chevron-right", Qa);
/**
 * @license lucide-react v1.28.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */
const to = [
  ["path", { d: "M18 6 6 18", key: "1bl5f8" }],
  ["path", { d: "m6 6 12 12", key: "d8bk6v" }]
], An = dt("x", to), no = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGAAAABgCAQAAABIkb+zAAAAIGNIUk0AAHomAACAhAAA+gAAAIDoAAB1MAAA6mAAADqYAAAXcJy6UTwAAAACYktHRAAAqo0jMgAAAAd0SU1FB+oHEQ4oCHiWdTwAAAAldEVYdGRhdGU6Y3JlYXRlADIwMjYtMDctMTdUMTQ6Mzk6NTArMDA6MDA+rUxAAAAAJXRFWHRkYXRlOm1vZGlmeQAyMDI2LTA3LTE3VDE0OjM5OjUwKzAwOjAwT/D0/AAAACh0RVh0ZGF0ZTp0aW1lc3RhbXAAMjAyNi0wNy0xN1QxNDo0MDowOCswMDowMLbEq2kAAAyzSURBVHja7Zt7tB1Vfcc/c+bcmxAeCeFCwiNgSIBcMUWQiIhUlihoK9RlAKE+qxZqDWqlXav0oVCr4ipLsa3KksJqsWorYC0ComBcPETKq4CQIEISwqOhJAQSQpJzZubTP2bP3DnnntclF7q6en+z1r3nzOzZ+/v97d/+7d/+7X1gSqZkSqZkSqZkSqYEAPOrLuLBXunzLhBr/m8DmzD8un/tNlWX5Xf+D0gF/hJXqbpN/YEYv4IErF5Ry1Xe7/peZCyepWrDzFRd7ZAYvXLAI2Pr1q2J7XRi68ZjT8a9HYvnq5qomqnbPUCsvRLQa9aNWyDv7CxnOsuZ7u5wC5V6lUbFeD6hpqYWkqlHiXH39us7Bh6AiBhJyYBZLOYwXsPBzGF3poUiEZtZxxM8wD3czxqS8FaKFjgSTuKrSMSYwWTEjLy8mo+C08MDPNtrfMZ+0vAev+TrSr1HuaN0rpu1on3VpvohcYfU3B184bEjT/M6t4dGU5s2TUzNKldqEu4XcpdnO7MyeK9QG21kXx4Cpc3H4pDL/HXZXGLWtwcyU5uh3HrPdRdr4pvHab8g8MFJJtCi+6WungD0Vklsqvo3Iv4wwG0vo2/vPYhfCvjc7vfxh8Gm0wnAHq/hk8XRLnVk6pJJI1CaDuLpblGbOwA+h/esI8F9Njs+3+p+veeBgaeI4O5iMuAivssMEuqDv99BMuBh1gNv7trkGp4IJbvIgMOj4q0jfsSJpEQ7NoeEStcCsLCjKjNq3AfEpK0BSDWyGAhEBf5u3MRraTK0g+AL2QLMZM82VLkk1LkRiKi10UsmSACAmITduJ3RSYSf6306MzroLGM6j3M1kJF1N6IBCJg3lALLGaXB8KTBz+UFNjETW0KIGjHf5E/YBGSM8loWMZeMmBdYySUTqD04TsQfOX6u3BFJ1J+KeJe2hXD6ERGHPMu721+cKPx82vrqJMM3xPvTxW9V3Ghmpn5AxLf4aLjbDFdDfWSiBIbED3T01JMhR7TV3lT/XMQzVW1U4qe81749MfixuNCG7tCk1Vma6rnibNerjaDhn4t4Rgm4vdfeNXHrv+Nl0n+qPmBNvKBy943iIpMOKkvVRx3UiZTm8xdOtvWPSaK+U9zXh7ze77jRW8rgbnybDfWcAcdwGfccZKYTjjUnQuBOI3GaiHu4QDyoY5SVqKsHzlGU9t850J08aarvEHHU1wev9/kS8JhkJqHkIJNv6Tzf4fiBNJmSqevdS/xTVb8u4sle6ipb54aG+rmB01zlovH+SSUw3hCb6rXi3m4LLS0pw/ZLyrYTE/UbIdc0Af2f4WSbTzbu2yXuIe7sc+aD9DO+z0s9UcSb1O0BwXljXnFwB3r7pOo/dXMLhWbQKh7uu1yhppWnp4mnhs8rPKHQ/qAGFIsnTip83e45lf7M1C3OFT/b1jvbfF69UjxB/bHvNRZrE4GfE7jayTOgTH3Baf6grDNR7xb3C6DbW3qnuJcHhhGxtzt1z6yOJ1ATF5g4ef4/Ve8Vj7HwLcWdnV3bQjPxWv/Vt1dSkYs8z8zfL3Ihgw7gz06i/vOa/k7EK8O3XDWj4pEu926/5NrQL0eKOM2jXOoXvTUQvrdXdruTA13hZIZviXqKiAeW8BP15jI1ibepL6q/JX6wDKRz+k31hIH2CoL9Hzup8DN1m/PEYfFcizgnUVd5pkucLf62qrdaFzeap8uKlFlT/feArS+Buvj1CRhQs2+ol6jLxzyJt1kdyqoPO0McdanTxU/ZHsrlPXag/XbMgv8fcs2APZCo3/Upew/3pvrhoJpYnOeLJfg0pHxvc34wpY+VkNvrOK9vKBEaeMuA8FN1jW/qkRgs/q93Zrm6qIc5pgoyUbf7E7/vio7w87YeyqfY/h7obx3MgJrquz3CThPeGKmGekGZEC7a+KitM29S+dS5N1PzxU7ci0Ak1tsiwV7wfxG6vJVuvjDfVOryBffK625JE/yl7RtKSc9sa1P9sjjUi0AcJvD+IUSupcPFi9oI5J8/7zfUxIal7Vp11LH46Tbd95ZUXdnTC4Vmvu0gBtQwX5Djdep2UzUr92A+KX7PfK93lfVC/xVTzfdlPlTWNags7kWgJu7tFvuHEA31prBqXtP2bLmLW9YSx4z573IU1KwZOSQe7bPm+8L9pal+qheBodCt/fSfqBvcU8T5bnGtt/grdatX+7YAcU83qvrxMfMpwUcliSFxT2+omF6/dq/qbUL4S/sN4PzpUSFnMcM54v6+yv3dJwAbFt+q6vmF9ksCtRCgHWixu4x4TqXm3i3/ukdaJTTbD35mHtcMBXAj3mLmVr9msf1UD+nCT1Tj+BL+XH+mpl7rSOiHmjjqzX1NN1MbvroXgX/o05V5A2cFkHk65OHy/uVj+vbLHi/GY8uQUD4OfZypt5djYkjEC+3nk1L19O4E8tg860PgD8pJqS5+Rt1q7oHCllzLnFu6vfB9mblvytxmnoOuh/AZT7PfSEjUL1Qxt+58vIF5LXn6dkmJuJGLqYc9kgR4DzBMscNyevgkMXFeqlJdBpxKviuRb1G9h2IDKAUOYhBZ1J3Ab9JzQw2AXcoyNWB/FpQYI+Cw8mlK2gY/ImMXDilbrQGHMoO0LNQ/ZRUB+3YncMQAr+/PTmREoc2dmdZCYJ+cQFRebTLctr8zg+mVb7MGIjC3O4FD6L3xGgGzGQGKeTU/o1KMUtgE9Dqe1O7EW3e/5jKIVCm3wd2ra0MFAZnOfhR2Dut4tuWdNUC3aFFgI0+3EH6c58KnLBDofzqrxchbCcxkzBw6U0iBQxkbqJu4I9zNK/5pFe/YFaQOLC/Lp+FbPqQzauwzEIHuUkktbeziTpvqN0vXF4tvUrXhNvW/3TW40Fo4elZc+VGzmniwqtvD8ZxwrNJowCgsU5/uTqDwtNf7tS4eOVV/WdFtbJFX1tS3iXEZ67RftRCBfris7aPlxBeLx9k/mMjUh6uYWx3XJnYjI+Y61vGHHTlGwCgHsooaGfnMcAE38342czGriMkQmM8oC5jDLFK28AwP8iBPku8AX8Y9/B41/pn/CPvPeb1Hh+cdNEvEBp7gMFLq/Ff3Hngo6P1oF3btzKZ6ZssCMa58qolH+EB5fmtMXvQuF1sclGp/syYut1sgkaq3emGYwy/rPojvCXee5RHWEHWc1CLgDKq+ICWmTp0aKTXgSA4lJiWpXCnTeB3HQChVp04ctA81Ml4VnnaSDLidx8O3+7sT+LeKYd1I51k5Bo7jN1o6OwdblN5O7qHqlSs/prO+BJSTqqJ4H8MkPXzQDTwTWv9FdwJX8TA1UgSuoJtLS4CPhcoqc25ENHaEZrwmY2A1kEXt83REQsxHuupf6mzhJ8wB6qzm3q4kxdPVhqNi1HVpk4+NQzol+ozEPXzOTnsxTzmjPatThtIf7xGHNtXviV9Q9Su9VmSx+I/qcRIyN42uVV5TXWtVANXF74x7s6Fe3J5XK4fvSNd5p3DsJ4a1Sp9Ffe6/7/TvzY+kPtTVLyTBiw+3UgiAFms4vD3mRfSg1sxmJUP0/R76T9Q7RLxW/Va/xFZNrPvWUPHJXSvOtbWknUIJ6Vw1tWEa/uaJlvbcUP72H9trGVPsDg/7lJlz7HUEsLLojoKBXK4dfHpR8XMuEoeLmbflZOP5LaX/rKy5Wqou/q5FJq+TNNSrRDxSPcneyV2LlNMYiboPdR0JibrF44PWq+FDvp+4xMt90Pu8zCXlvbES+Sr40xUTa+/fvIWN7iXi+/MzvO1uozeZWDzADfYazPpXpTXHFZCtR/HjCvQ4zOKz/BfHnz3Nyr/5wv+Eyqw/zmn0I5C/usANFUNKW9Lmuax0afmLgLiMPuv5ry8csl5GpwWRnTzbdeM0XvTF8xY56mUvEX6FwpC4wMdadFWc5bnNL/pkuPeYn/PwDr/caL+GfaNf8dnw1hYv8b4SeN7PZ/mz8LQKv8sWX9SbApCfF92Jf+JUYCvXcCr5bFwnZRGPcCxH8XpOYjrwND/ndn7F4zzNZiI2UWNXMmYxwnxezRs4hlnA09zIau7nauaxkjqSIXW28m6u5xlG2MB7+XGR/3jJq5yWzjvFleof+RrvLTv+0dLqZ7vUS8uj+IVprHND27h5wIs83umlTleo24MR3egBYV1wubvbllfqLH2phddrRKTAMuZzDvA7LONYpgF3ciwJUXmadoSFLGQ/9mVXZjObjCdpsJG1PMYa1rCBol8jmlzBKeH7HVwQgsl5jPCfoef7Qhygbyor2hzkEAkCB3Mc89mVC1lDDYlpORTctcUYyaiR8kkuoslabuAKlpMHc7nK60W+4uX/BdaUTMmUTMmUTMmU/D+W/wGZEHgnLlD3igAAAABJRU5ErkJggg==";
class Ar extends Error {
  constructor(n, r, i) {
    super(i);
    Ge(this, "status");
    Ge(this, "detail");
    this.name = "ApiError", this.status = n, this.detail = r;
  }
}
let Fe = (e, t) => fetch(e, t);
function ro(e) {
  const t = Fe;
  return Fe = e, () => {
    Fe === e && (Fe = t);
  };
}
async function Le(e, t = {}) {
  const n = await Fe(e, {
    headers: {
      "Content-Type": "application/json",
      ...t.headers ?? {}
    },
    ...t
  });
  if (!n.ok) {
    const i = (await n.json().catch(() => ({}))).detail, o = typeof i == "string" ? i : typeof i == "object" && i !== null && typeof i.message == "string" ? String(i.message) : `请求失败: ${n.status}`;
    throw new Ar(n.status, i, o);
  }
  return n.status === 204 ? null : n.json();
}
function io(e) {
  if (!(e instanceof Ar) || e.status !== 409) return null;
  const t = e.detail;
  if (typeof t != "object" || t === null || Array.isArray(t)) return null;
  const n = t;
  return n.code !== "interaction_delete_required" || typeof n.message_id != "string" || !n.message_id || typeof n.control_turn_id != "string" || !n.control_turn_id ? null : n;
}
function Dn(e, t) {
  return Math.max(1, Math.ceil(e / t));
}
function kn(e) {
  if (typeof e != "object" || e === null || Array.isArray(e) || !Array.isArray(e.items) || typeof e.total != "number" || !Number.isFinite(e.total) || e.total < 0)
    throw new Error("分页接口返回格式无效");
  const t = e;
  return {
    items: t.items,
    total: t.total,
    page: t.page,
    page_size: t.page_size
  };
}
function On(e) {
  return encodeURIComponent(e).replaceAll("%2F", "/");
}
function Zt(e) {
  return String(e ?? "").replace(/\*\*(.+?)\*\*/g, "$1").replace(/\*(.+?)\*/g, "$1").replace(/__(.+?)__/g, "$1").replace(/_(.+?)_/g, "$1").replace(/~~(.+?)~~/g, "$1").replace(/`{1,3}[\s\S]*?`{1,3}/g, "").replace(/\[(.+?)\]\(.+?\)/g, "$1").replace(/^#{1,6}\s+/gm, "").replace(/^>\s*/gm, "").replace(/\n+/g, " ").trim();
}
function ft(e) {
  const t = String(e || ""), n = t.split(":");
  if (n.length < 2)
    return t;
  const r = n[0], i = n.slice(1).join(":");
  return i.length <= 10 ? `${r}:${i}` : `${r}:${i.slice(0, 6)}...${i.slice(-4)}`;
}
function Qt(e) {
  if (!e)
    return "-";
  const t = new Date(String(e));
  return Number.isNaN(t.getTime()) ? String(e) : `${t.getMonth() + 1}-${String(t.getDate()).padStart(2, "0")} ${String(
    t.getHours()
  ).padStart(2, "0")}:${String(t.getMinutes()).padStart(2, "0")}`;
}
function ao(e) {
  if (!e)
    return "未更新";
  const t = new Date(String(e)).getTime();
  if (Number.isNaN(t))
    return String(e);
  const n = Date.now() - t, r = 60 * 1e3, i = 60 * r, o = 24 * i;
  return n < i ? `${Math.max(1, Math.round(n / r))} 分钟前` : n < o ? `${Math.round(n / i)} 小时前` : `${Math.round(n / o)} 天前`;
}
function le(e) {
  const t = Number(e || 0);
  return t <= 0 ? "0" : t >= 1e6 ? `${(t / 1e6).toFixed(1)}M` : t >= 1e3 ? `${(t / 1e3).toFixed(1)}k` : String(t);
}
function Dr(e) {
  return `role-${e || "unknown"}`;
}
class kr extends Qr {
  constructor() {
    super(...arguments);
    Ge(this, "state", { error: null });
  }
  static getDerivedStateFromError(n) {
    return { error: n };
  }
  componentDidCatch(n) {
    console.error(`[dashboard] ${this.props.pluginId} ${this.props.slot} failed`, n);
  }
  render() {
    return this.state.error ? /* @__PURE__ */ y("div", { className: "plugin-entry-error", role: "alert", children: [
      /* @__PURE__ */ y("strong", { children: [
        this.props.pluginId,
        " 无法显示"
      ] }),
      /* @__PURE__ */ a("span", { children: this.state.error.message })
    ] }) : this.props.children;
  }
}
function en(e, t, n, r) {
  try {
    const i = r();
    return () => {
      try {
        i == null || i();
      } catch (o) {
        console.error(`[dashboard] ${t} ${n} cleanup failed`, o);
      }
    };
  } catch (i) {
    console.error(`[dashboard] ${t} ${n} failed`, i), e.replaceChildren();
    const o = document.createElement("div");
    o.className = "plugin-entry-error", o.setAttribute("role", "alert");
    const l = document.createElement("strong");
    l.textContent = `${t} 无法显示`;
    const u = document.createElement("span");
    u.textContent = i instanceof Error ? i.message : String(i), o.append(l, u), e.append(o);
  }
}
function oo(e) {
  var r;
  const t = H(null), n = e.plugin.Detail;
  if (lt(() => t.current ? e.plugin.applyStyle(t.current) : void 0, [e.plugin]), z(() => {
    if (!n)
      if (t.current && e.plugin.renderDetail) {
        const i = t.current;
        return en(i, e.plugin.id, "detail", () => e.plugin.renderDetail(e.item, i, e.dispatch));
      } else t.current && (t.current.innerHTML = "");
  }, [n, e.item, e.plugin, e.dispatch]), n) {
    const i = String(((r = e.item) == null ? void 0 : r[e.plugin.rowKey]) ?? "empty");
    return /* @__PURE__ */ a("div", { ref: t, className: "plugin-workbench-root", "data-akashic-plugin": e.plugin.id, children: /* @__PURE__ */ a(kr, { pluginId: e.plugin.id, slot: "detail", children: /* @__PURE__ */ a(n, { item: e.item, dispatch: e.dispatch }) }, `${e.plugin.id}:${i}`) });
  }
  return /* @__PURE__ */ a("div", { ref: t, "data-akashic-plugin": e.plugin.id });
}
function so(e) {
  const t = H(null), n = e.plugin.Main, r = H(e.dispatch);
  return lt(() => t.current ? e.plugin.applyStyle(t.current) : void 0, [e.plugin]), z(() => {
    r.current = e.dispatch;
  }, [e.dispatch]), z(() => {
    if (!n && t.current && e.plugin.renderMain) {
      const i = t.current;
      return en(i, e.plugin.id, "main", () => e.plugin.renderMain(i, r.current));
    }
  }, [n, e.plugin]), n ? /* @__PURE__ */ a("div", { ref: t, className: "plugin-workbench-root", "data-akashic-plugin": e.plugin.id, children: /* @__PURE__ */ a(kr, { pluginId: e.plugin.id, slot: "main", children: /* @__PURE__ */ a(n, { dispatch: e.dispatch }) }, e.plugin.id) }) : /* @__PURE__ */ a("div", { className: "plugin-workbench-root", ref: t, "data-akashic-plugin": e.plugin.id });
}
const co = no, lo = {
  text: (e) => String(e ?? ""),
  "mono-session": (e) => ft(e),
  "mono-time": (e) => Qt(e),
  "text-preview": (e) => Zt(e),
  metric: (e) => String(e ?? 0)
};
function Or(e, t, n, r, i, o, l = (u) => console.error("[dashboard] plugin request failed", u)) {
  const u = (f) => {
    f.catch(l);
  }, s = async (f, h, v) => {
    const E = t();
    if (!E) return;
    const C = r(), b = $t(e, await e.fetchPage({
      page: 1,
      pageSize: E.pageSize,
      filters: f,
      sortBy: h,
      sortOrder: v,
      signal: C.signal
    }));
    C.signal.aborted || n((p) => ({
      ...p,
      page: 1,
      total: b.total,
      items: b.items,
      activeRowKey: null,
      activeDetail: null,
      filters: f,
      sortBy: h,
      sortOrder: v
    }));
  }, c = (f) => {
    const h = t();
    h && u(s(f({ ...h.filters }), h.sortBy, h.sortOrder));
  };
  return {
    get filters() {
      var f;
      return ((f = t()) == null ? void 0 : f.filters) ?? {};
    },
    setFilter(f, h) {
      c((v) => ({ ...v, [f]: h }));
    },
    clearFilter(f) {
      c((h) => (delete h[f], h));
    },
    setFilters(f) {
      c((h) => ({ ...h, ...f }));
    },
    clearFilters(f) {
      c((h) => {
        for (const v of f) delete h[v];
        return h;
      });
    },
    get sortBy() {
      var f;
      return ((f = t()) == null ? void 0 : f.sortBy) ?? "";
    },
    get sortOrder() {
      var f;
      return ((f = t()) == null ? void 0 : f.sortOrder) ?? "desc";
    },
    setSort(f) {
      const h = t();
      if (!h) return;
      const v = h.sortBy === f && h.sortOrder === "desc" ? "asc" : "desc";
      u(s(h.filters, f, v));
    },
    refresh() {
      const f = t();
      f && u(s(f.filters, f.sortBy, f.sortOrder));
    },
    activate() {
      i == null || i();
    },
    closePane() {
      o == null || o();
    }
  };
}
function $t(e, t) {
  if (!t || !Array.isArray(t.items) || typeof t.total != "number" || !Number.isFinite(t.total) || t.total < 0)
    throw new Error(`插件 ${e.id} 返回了无效分页数据`);
  return t;
}
function uo(e) {
  return e instanceof Error && e.name === "AbortError";
}
function fo(e) {
  const t = H(e);
  return lt(() => {
    t.current = e;
  }, [e]), j(() => t.current, []);
}
function mo({ initialPlugins: e }) {
  const [t, n] = T(null), [r, i] = T("sessions"), o = e, [l, u] = T({}), [s, c] = T([]), [f, h] = T(""), [v, E] = T(""), [C, b] = T({
    scheduler: !1,
    programmatic: !1
  }), [p, N] = T(null), [A, P] = T(null), [O, D] = T(null), [x, $] = T(!1), [he, se] = T([]), [ne, J] = T(""), [M, ke] = T(""), [ve, K] = T(1), [$e, xr] = T("ts"), [qe, Lr] = T("desc"), [tn, Ir] = T(0), [mt, Z] = T(null), [ht, vt] = T(/* @__PURE__ */ new Set()), [Fr, nn] = T({}), [Br, je] = T(null), [rn, ze] = T(null), Oe = H(null), Me = H(null), ge = H(null), Ke = H(/* @__PURE__ */ new Map()), X = H(null), an = 25, re = r.startsWith("plugin:") ? r.slice(7) : "", S = o.find((d) => d.id === re) ?? null, k = re ? l[re] : null, on = !!k, Ur = (S == null ? void 0 : S.layout) ?? "table", Xe = fo(l), sn = j((d, g) => {
    u((w) => {
      const R = w[d];
      return R ? { ...w, [d]: g(R) } : w;
    });
  }, []), cn = j((d) => {
    i(`plugin:${d}`);
  }, []), ln = j((d) => {
    var g;
    (g = X.current) == null || g.abort(), X.current = null, je(null), u((w) => {
      const R = w[d];
      return R ? { ...w, [d]: { ...R, activeRowKey: null, activeDetail: null } } : w;
    });
  }, []), Q = j((d) => {
    var w;
    (w = Ke.current.get(d)) == null || w.abort();
    const g = new AbortController();
    return Ke.current.set(d, g), g;
  }, []), Wr = xe(() => Array.from(new Set(s.map((d) => d.key.split(":")[0]).filter(Boolean))), [s]), $r = xe(
    () => s.filter((d) => !vo(d)),
    [s]
  ), qr = xe(
    () => s.filter((d) => Be(d) === "scheduler"),
    [s]
  ), jr = xe(
    () => s.filter((d) => Be(d) === "programmatic"),
    [s]
  ), ce = j((d) => {
    uo(d) || (console.error("[dashboard] request failed", d), ze(d instanceof Error ? d.message : String(d)));
  }, []), B = j(async (d) => {
    try {
      ze(null), await d();
    } catch (g) {
      ce(g);
    }
  }, [ce]), zr = j(async (d) => {
    const g = new Set(d);
    for (; g.size > 0; )
      try {
        return await Le("/api/dashboard/messages/batch-delete", {
          method: "POST",
          body: JSON.stringify({ ids: [...g] })
        }), !0;
      } catch (w) {
        const R = io(w);
        if (!R) throw w;
        if (!window.confirm("所选消息属于一次完整交互。继续会撤销这一轮的全部用户输入和最终回复，是否继续？"))
          return !1;
        const L = await Le(
          `/api/dashboard/interactions/${On(R.control_turn_id)}`,
          { method: "DELETE" }
        );
        for (const _ of L.message_ids) g.delete(_);
        if (g.has(R.message_id))
          throw new Error("整轮撤销响应未包含触发删除的消息", { cause: w });
      }
    return !0;
  }, []), He = j(async () => {
    var w;
    (w = Oe.current) == null || w.abort();
    const d = new AbortController();
    Oe.current = d;
    const g = new URLSearchParams();
    f && g.set("q", f), v && g.set("channel", v), g.set("page_size", "200");
    try {
      const R = kn(await Le(`/api/dashboard/sessions?${g.toString()}`, { signal: d.signal }));
      c(R.items), P((L) => L ? R.items.find((_) => _.key === L.key) ?? null : null);
    } finally {
      Oe.current === d && (Oe.current = null);
    }
  }, [v, f]), Ve = j(async () => {
    var w;
    (w = Me.current) == null || w.abort();
    const d = new AbortController();
    Me.current = d;
    const g = new URLSearchParams();
    p && g.set("session_key", p), ne && g.set("q", ne), M && g.set("role", M), g.set("page", String(ve)), g.set("page_size", String(an)), g.set("sort_by", $e), g.set("sort_order", qe);
    try {
      const R = kn(await Le(`/api/dashboard/messages?${g.toString()}`, { signal: d.signal }));
      se(R.items), Ir(R.total), Z((L) => L && R.items.some((_) => _.id === L.id) ? L : null);
    } finally {
      Me.current === d && (Me.current = null);
    }
  }, [p, ve, M, ne, $e, qe]), Ye = j(async () => {
    var g;
    if ((g = ge.current) == null || g.abort(), !p) {
      ge.current = null, D(null);
      return;
    }
    const d = new AbortController();
    ge.current = d, $(!0);
    try {
      const w = await Le(
        `/api/dashboard/sessions/${On(p)}/compaction`,
        { signal: d.signal }
      );
      if (d.signal.aborted) return;
      D(w);
    } finally {
      ge.current === d && (ge.current = null, $(!1));
    }
  }, [p]), _e = j(async (d) => {
    const g = o.find((_) => _.id === d), w = Xe()[d];
    if (!g || !w) return;
    const R = Q(d), L = $t(g, await g.fetchPage({
      page: w.page,
      pageSize: w.pageSize,
      filters: w.filters,
      sortBy: w.sortBy,
      sortOrder: w.sortOrder,
      signal: R.signal
    }));
    R.signal.aborted || u((_) => {
      var q, ye;
      return {
        ..._,
        [d]: {
          ..._[d],
          total: L.total,
          items: L.items,
          activeRowKey: (q = _[d]) != null && q.activeRowKey && L.items.some((bt) => String(bt[g.rowKey] ?? "") === _[d].activeRowKey) ? _[d].activeRowKey : null,
          activeDetail: (ye = _[d]) != null && ye.activeRowKey && L.items.some((bt) => String(bt[g.rowKey] ?? "") === _[d].activeRowKey) ? _[d].activeDetail : null
        }
      };
    });
  }, [o, Xe, Q]), gt = j(async () => {
    await He(), r === "compaction" ? await Ye() : r.startsWith("plugin:") ? await _e(r.slice(7)) : await Ve();
  }, [Ye, Ve, _e, He, r]);
  z(() => {
    const d = () => {
      B(gt);
    };
    return window.addEventListener("akashic-dashboard-refresh", d), () => window.removeEventListener("akashic-dashboard-refresh", d);
  }, [gt, B]), z(() => () => {
    var d, g, w, R;
    (d = Oe.current) == null || d.abort(), (g = Me.current) == null || g.abort(), (w = ge.current) == null || w.abort(), (R = X.current) == null || R.abort();
    for (const L of Ke.current.values()) L.abort();
    Ke.current.clear();
  }, []), z(() => {
    u(Object.fromEntries(e.map((d) => [d.id, {
      page: 1,
      pageSize: d.pageSize || 25,
      total: 0,
      items: [],
      activeRowKey: null,
      activeDetail: null,
      filters: {},
      sortBy: d.defaultSortBy ?? "",
      sortOrder: d.defaultSortOrder ?? "desc",
      selectedIds: /* @__PURE__ */ new Set()
    }])));
  }, [e]), z(() => {
    B(He);
  }, [He, B]), z(() => {
    for (const d of o)
      B(async () => {
        const g = Q(d.id), w = await d.getCount({ signal: g.signal });
        if (!g.signal.aborted)
          if (w === null)
            nn((R) => ({ ...R, [d.id]: !0 }));
          else {
            if (!Number.isFinite(w) || w < 0)
              throw new Error(`插件 ${d.id} 返回了无效计数`);
            nn((R) => ({ ...R, [d.id]: !1 })), u((R) => ({
              ...R,
              [d.id]: { ...R[d.id], total: w }
            }));
          }
      });
  }, [o, B, Q]);
  const pt = j((d) => {
    i(d);
  }, []), pe = (d) => {
    pt(d);
  }, Kr = Ce((d) => {
    N(d), P(s.find((w) => w.key === d) ?? null), Z(null), K(1);
    const g = Be({ key: d });
    (g === "scheduler" || g === "programmatic") && b((w) => ({ ...w, [g]: !0 })), pe("sessions");
  });
  z(() => {
    const d = (g) => {
      const w = g.detail;
      w && Kr(w);
    };
    return window.addEventListener("akashic:goto-session", d), () => window.removeEventListener("akashic:goto-session", d);
  }, []);
  const Xr = (d) => {
    Lr(((w, R) => w === d && R === "desc" ? "asc" : "desc")($e, qe)), xr(d), K(1);
  };
  z(() => {
    r === "sessions" && B(Ve);
  }, [Ve, B, r]), z(() => {
    r === "compaction" && B(Ye);
  }, [Ye, B, r]), z(() => {
    r.startsWith("plugin:") && B(() => _e(r.slice(7)));
  }, [_e, B, r]);
  const yt = k ? Dn(k.total, k.pageSize) : Dn(tn, an), Te = (k == null ? void 0 : k.page) ?? ve, un = (d) => {
    Te + d < 1 || Te + d > yt || (re ? B(async () => {
      const g = o.find((q) => q.id === re), w = l[re];
      if (!g || !w) return;
      const R = w.page + d, L = Q(re), _ = $t(g, await g.fetchPage({
        page: R,
        pageSize: w.pageSize,
        filters: w.filters,
        sortBy: w.sortBy,
        sortOrder: w.sortOrder,
        signal: L.signal
      }));
      L.signal.aborted || u((q) => ({
        ...q,
        [re]: {
          ...q[re],
          page: R,
          total: _.total,
          items: _.items,
          activeRowKey: null,
          activeDetail: null
        }
      }));
    }) : K((g) => g + d));
  }, Hr = (k == null ? void 0 : k.selectedIds.size) ?? 0, dn = r.startsWith("plugin:") ? Hr : ht.size, de = xe(() => S && on ? Or(
    S,
    () => Xe()[S.id] ?? null,
    (d) => sn(S.id, d),
    () => Q(S.id),
    () => cn(S.id),
    () => ln(S.id),
    ce
  ) : void 0, [cn, ln, S, on, Xe, ce, sn, Q]), fn = !!(S && k && de && Ur === "workbench" && (S.renderMain || S.Main)), Vr = r.startsWith("plugin:") ? !!(k != null && k.activeRowKey) : r === "compaction" ? !1 : !!(mt || A);
  return /* @__PURE__ */ a("div", { ref: n, className: "workbench-root", children: /* @__PURE__ */ y("div", { className: "shell", children: [
    /* @__PURE__ */ y("aside", { className: "sessions-pane", children: [
      /* @__PURE__ */ y("div", { className: "brand", children: [
        /* @__PURE__ */ a("img", { className: "brand-mark", src: co, alt: "" }),
        /* @__PURE__ */ y("div", { children: [
          /* @__PURE__ */ a("div", { className: "brand-title", children: "Akashic" }),
          /* @__PURE__ */ a("div", { className: "brand-sub", children: "Dashboard" })
        ] })
      ] }),
      /* @__PURE__ */ a(
        ho,
        {
          viewMode: r,
          sessionsCount: s.length,
          plugins: o.filter((d) => !Fr[d.id]),
          pluginState: l,
          onSelect: (d) => {
            d === "sessions" && (N(null), P(null), Z(null), K(1)), pe(d);
          }
        }
      ),
      /* @__PURE__ */ y("div", { className: "explorer-body", children: [
        (r === "sessions" || r === "compaction") && /* @__PURE__ */ y(fe, { children: [
          /* @__PURE__ */ y("div", { className: "filters-stack session-filters", children: [
            /* @__PURE__ */ y("label", { className: "search search-small", children: [
              /* @__PURE__ */ a("span", { "aria-hidden": "true", children: "⌕" }),
              /* @__PURE__ */ a("input", { "aria-label": "搜索会话", type: "text", placeholder: "搜索会话", value: f, onChange: (d) => h(d.target.value.trim()) })
            ] }),
            /* @__PURE__ */ y("select", { "aria-label": "会话来源", value: v, onChange: (d) => {
              const g = d.target.value;
              E(g), (g === "scheduler" || g === "programmatic") && b((w) => ({ ...w, [g]: !0 }));
            }, children: [
              /* @__PURE__ */ a("option", { value: "", children: "全部来源" }),
              Wr.map((d) => /* @__PURE__ */ a("option", { value: d, children: d }, d))
            ] })
          ] }),
          /* @__PURE__ */ y("div", { className: "session-list", children: [
            /* @__PURE__ */ y("button", { className: `all-messages-row ${p ? "" : "active"}`, type: "button", onClick: () => {
              N(null), P(null), Z(null), K(1), pe(r === "compaction" ? "compaction" : "sessions");
            }, children: [
              /* @__PURE__ */ a("span", { children: "全部会话" }),
              /* @__PURE__ */ a("strong", { children: s.length })
            ] }),
            $r.map((d) => /* @__PURE__ */ a(
              Mr,
              {
                session: d,
                active: p === d.key,
                onSelect: () => {
                  N(d.key), P(d), Z(null), K(1), pe(r === "compaction" ? "compaction" : "sessions");
                }
              },
              d.key
            )),
            /* @__PURE__ */ a(
              Mn,
              {
                id: "scheduler-sessions",
                label: "定时任务",
                sessions: qr,
                open: C.scheduler,
                activeSessionKey: p,
                onOpenChange: () => b((d) => ({ ...d, scheduler: !d.scheduler })),
                onSelect: (d) => {
                  N(d.key), P(d), Z(null), K(1), pe(r === "compaction" ? "compaction" : "sessions");
                }
              }
            ),
            /* @__PURE__ */ a(
              Mn,
              {
                id: "programmatic-sessions",
                label: "程序会话",
                sessions: jr,
                open: C.programmatic,
                activeSessionKey: p,
                onOpenChange: () => b((d) => ({ ...d, programmatic: !d.programmatic })),
                onSelect: (d) => {
                  N(d.key), P(d), Z(null), K(1), pe(r === "compaction" ? "compaction" : "sessions");
                }
              }
            )
          ] })
        ] }),
        r.startsWith("plugin:") && S && k && S.renderNavBody && /* @__PURE__ */ a(
          qt,
          {
            plugin: S,
            pluginId: S.id,
            render: S.renderNavBody,
            slot: "navigation",
            redrawOnTotal: k.total,
            state: k,
            onSetState: (d) => u((g) => ({ ...g, [S.id]: d(g[S.id]) })),
            startRead: () => Q(S.id),
            onActivate: () => pt(`plugin:${S.id}`),
            onError: ce
          }
        )
      ] })
    ] }),
    /* @__PURE__ */ y("section", { className: "content-shell", children: [
      /* @__PURE__ */ y("header", { className: "content-toolbar", children: [
        /* @__PURE__ */ a(
          yo,
          {
            viewMode: r,
            messageSearch: ne,
            setMessageSearch: (d) => {
              J(d), K(1);
            },
            messageRole: M,
            setMessageRole: (d) => {
              ke(d), K(1);
            },
            activeSessionKey: p,
            clearSession: () => {
              N(null), P(null), Z(null), K(1);
            },
            currentPlugin: S,
            currentPluginState: k,
            onSetPluginState: S ? (d) => u((g) => ({ ...g, [S.id]: d(g[S.id]) })) : void 0,
            startPluginRead: Q,
            onError: ce
          }
        ),
        r.startsWith("plugin:") && (S == null ? void 0 : S.renderTopbarAction) && k && de && /* @__PURE__ */ a("div", { className: "content-toolbar-actions", children: /* @__PURE__ */ a(
          qt,
          {
            plugin: S,
            pluginId: S.id,
            render: S.renderTopbarAction,
            slot: "topbar action",
            state: k,
            onSetState: (d) => u((g) => ({ ...g, [S.id]: d(g[S.id]) })),
            startRead: () => Q(S.id),
            onActivate: () => pt(`plugin:${S.id}`),
            onError: ce
          }
        ) })
      ] }),
      /* @__PURE__ */ a("main", { className: `workspace${fn ? " plugin-workbench-mode" : ""}`, children: fn && S && de ? /* @__PURE__ */ a("section", { className: "plugin-workbench-pane", children: /* @__PURE__ */ a(so, { plugin: S, dispatch: de }) }) : /* @__PURE__ */ y(fe, { children: [
        /* @__PURE__ */ a("section", { className: "messages-pane", children: r === "compaction" ? /* @__PURE__ */ a(
          Ao,
          {
            compaction: O,
            pending: x,
            activeSessionKey: p
          }
        ) : /* @__PURE__ */ y(fe, { children: [
          dn > 0 && /* @__PURE__ */ y("div", { className: "batch-bar", children: [
            /* @__PURE__ */ y("span", { children: [
              "已选 ",
              dn,
              " 条"
            ] }),
            r.startsWith("plugin:") && (S != null && S.batchActions) && k ? S.batchActions.map((d) => /* @__PURE__ */ a("button", { className: d.className, type: "button", onClick: () => void B(async () => {
              const g = [...k.selectedIds];
              await d.run(g), u((w) => ({ ...w, [S.id]: { ...w[S.id], selectedIds: /* @__PURE__ */ new Set() } })), await _e(S.id);
            }), children: d.label }, d.label)) : /* @__PURE__ */ a(St, { size: "sm", variant: "danger", onClick: () => void B(async () => {
              await zr([...ht]) && (vt(/* @__PURE__ */ new Set()), await gt());
            }), children: "批量删除" }),
            /* @__PURE__ */ a(St, { size: "sm", variant: "ghost", onClick: () => {
              r.startsWith("plugin:") && S ? u((d) => ({ ...d, [S.id]: { ...d[S.id], selectedIds: /* @__PURE__ */ new Set() } })) : vt(/* @__PURE__ */ new Set());
            }, children: "取消选择" })
          ] }),
          /* @__PURE__ */ a(So, { viewMode: r, plugin: S, pluginState: k, messageSortBy: $e, messageSortOrder: qe, onSort: Xr, onPluginSort: de ? (d) => de.setSort(d) : void 0 }),
          /* @__PURE__ */ a("div", { className: "table-body", children: /* @__PURE__ */ a(
            wo,
            {
              viewMode: r,
              messages: he,
              plugin: S,
              pluginState: k,
              selectedMessageIds: ht,
              activeMessage: mt,
              onSelectMessage: (d) => Z((g) => (g == null ? void 0 : g.id) === d.id ? null : d),
              onSelectPluginRow: (d) => {
                var L;
                if (!S) return;
                const g = String(d[S.rowKey] ?? "");
                (L = X.current) == null || L.abort();
                const w = (k == null ? void 0 : k.activeRowKey) === g;
                if (je(w ? null : `${S.id}:${g}`), u((_) => {
                  const q = _[S.id];
                  return q ? { ..._, [S.id]: { ...q, activeRowKey: w ? null : g, activeDetail: null } } : _;
                }), w) {
                  X.current = null;
                  return;
                }
                const R = new AbortController();
                X.current = R, (async () => {
                  try {
                    const _ = S.fetchDetail ? await S.fetchDetail(d, { signal: R.signal }) : d;
                    if (R.signal.aborted || X.current !== R) return;
                    u((q) => {
                      const ye = q[S.id];
                      return !ye || ye.activeRowKey !== g ? q : { ...q, [S.id]: { ...ye, activeDetail: _ } };
                    });
                  } catch (_) {
                    X.current === R && ce(_);
                  } finally {
                    X.current === R && (X.current = null, je(null));
                  }
                })();
              },
              onTogglePluginRow: (d) => {
                S && u((g) => {
                  const w = g[S.id];
                  if (!w) return g;
                  const R = new Set(w.selectedIds);
                  return R.has(d) ? R.delete(d) : R.add(d), { ...g, [S.id]: { ...w, selectedIds: R } };
                });
              },
              setSelectedMessageIds: vt
            }
          ) }),
          /* @__PURE__ */ y("footer", { className: "table-foot", children: [
            /* @__PURE__ */ a("div", { children: Ro(tn, S, k) }),
            /* @__PURE__ */ y("div", { className: "pager", children: [
              /* @__PURE__ */ a(ct, { variant: "standard", label: "上一页", disabled: Te <= 1, onClick: () => un(-1), children: /* @__PURE__ */ a(Za, { size: 18, "aria-hidden": "true" }) }),
              /* @__PURE__ */ y("span", { children: [
                Te,
                " / ",
                yt
              ] }),
              /* @__PURE__ */ a(ct, { variant: "standard", label: "下一页", disabled: Te >= yt, onClick: () => un(1), children: /* @__PURE__ */ a(eo, { size: 18, "aria-hidden": "true" }) })
            ] })
          ] })
        ] }) }),
        /* @__PURE__ */ a("aside", { className: `detail-pane${Vr ? " is-open" : ""}`, "aria-label": "详情", children: /* @__PURE__ */ a(
          Co,
          {
            viewMode: r,
            activeSession: A,
            activeMessage: mt,
            plugin: S,
            pluginState: k,
            loading: !!(S && (k != null && k.activeRowKey) && Br === `${S.id}:${k.activeRowKey}`),
            dispatch: de,
            onClose: () => {
              var d;
              P(null), Z(null), S && ((d = X.current) == null || d.abort(), X.current = null, je(null), u((g) => {
                const w = g[S.id];
                return w ? { ...g, [S.id]: { ...w, activeRowKey: null, activeDetail: null } } : g;
              }));
            }
          }
        ) })
      ] }) })
    ] }),
    /* @__PURE__ */ a(Oa, { open: !!rn, onOpenChange: (d) => {
      d || ze(null);
    }, children: /* @__PURE__ */ y(_a, { container: t, children: [
      /* @__PURE__ */ a(Ta, { className: "modal-backdrop" }),
      /* @__PURE__ */ y(Ia, { className: "modal", "aria-describedby": "dashboard-error-description", children: [
        /* @__PURE__ */ a(Wa, { className: "modal-title", children: "请求失败" }),
        /* @__PURE__ */ a(qa, { id: "dashboard-error-description", className: "modal-sub", children: rn }),
        /* @__PURE__ */ a("div", { className: "modal-actions", children: /* @__PURE__ */ a(St, { onClick: () => ze(null), children: "关闭" }) })
      ] })
    ] }) })
  ] }) });
}
function Mn(e) {
  return e.sessions.length ? /* @__PURE__ */ y("div", { className: `nav-group session-group ${e.open ? "open" : ""}`, children: [
    /* @__PURE__ */ y(
      "button",
      {
        className: "nav-group-toggle",
        type: "button",
        "aria-expanded": e.open,
        "aria-controls": e.id,
        onClick: e.onOpenChange,
        children: [
          /* @__PURE__ */ a("span", { className: "nav-group-caret", "aria-hidden": "true", children: "›" }),
          /* @__PURE__ */ a("span", { className: "nav-group-label", children: e.label }),
          /* @__PURE__ */ a("span", { className: "nav-group-count", children: e.sessions.length })
        ]
      }
    ),
    /* @__PURE__ */ a(
      "div",
      {
        id: e.id,
        className: `nav-group-body ${e.open ? "open" : ""}`,
        hidden: !e.open,
        children: /* @__PURE__ */ a("div", { className: "nav-group-body-inner", children: e.sessions.map((t) => /* @__PURE__ */ a(
          Mr,
          {
            session: t,
            active: e.activeSessionKey === t.key,
            nested: !0,
            onSelect: () => e.onSelect(t)
          },
          t.key
        )) })
      }
    )
  ] }) : null;
}
function ho(e) {
  var c;
  const [t, n] = T(!1), r = H(null), i = H(null), o = e.viewMode.startsWith("plugin:") ? e.plugins.find((f) => `plugin:${f.id}` === e.viewMode) ?? null : null, l = e.viewMode === "sessions" ? "Sessions" : (o == null ? void 0 : o.label) ?? "Explorer", u = e.viewMode === "sessions" ? e.sessionsCount : o ? ((c = e.pluginState[o.id]) == null ? void 0 : c.total) ?? 0 : 0;
  z(() => {
    if (!t) return;
    const f = (v) => {
      var E;
      (E = r.current) != null && E.contains(v.target) || n(!1);
    }, h = (v) => {
      var E;
      v.key === "Escape" && (n(!1), (E = i.current) == null || E.focus());
    };
    return document.addEventListener("pointerdown", f), document.addEventListener("keydown", h), () => {
      document.removeEventListener("pointerdown", f), document.removeEventListener("keydown", h);
    };
  }, [t]);
  const s = (f) => {
    e.onSelect(f), n(!1), queueMicrotask(() => {
      var h;
      return (h = i.current) == null ? void 0 : h.focus();
    });
  };
  return /* @__PURE__ */ y("div", { className: "module-switcher", ref: r, children: [
    /* @__PURE__ */ y(
      "button",
      {
        ref: i,
        className: "module-switcher-trigger",
        type: "button",
        "aria-expanded": t,
        "aria-controls": "dashboard-module-options",
        onClick: () => n((f) => !f),
        children: [
          /* @__PURE__ */ a("span", { className: "module-switcher-label", children: l }),
          /* @__PURE__ */ y("span", { className: "module-switcher-meta", children: [
            /* @__PURE__ */ a("span", { className: "module-switcher-count", children: u }),
            /* @__PURE__ */ a(Ga, { className: t ? "open" : "", size: 16, "aria-hidden": "true" })
          ] })
        ]
      }
    ),
    /* @__PURE__ */ y("div", { id: "dashboard-module-options", className: "module-switcher-options", hidden: !t, children: [
      /* @__PURE__ */ y(
        "button",
        {
          className: `module-switcher-option ${e.viewMode === "sessions" ? "active" : ""}`,
          type: "button",
          "aria-current": e.viewMode === "sessions" ? "page" : void 0,
          onClick: () => s("sessions"),
          children: [
            /* @__PURE__ */ a("span", { children: "Sessions" }),
            /* @__PURE__ */ a("span", { children: e.sessionsCount })
          ]
        }
      ),
      /* @__PURE__ */ y(
        "button",
        {
          className: `module-switcher-option ${e.viewMode === "compaction" ? "active" : ""}`,
          type: "button",
          "aria-current": e.viewMode === "compaction" ? "page" : void 0,
          onClick: () => s("compaction"),
          children: [
            /* @__PURE__ */ a("span", { children: "Compaction" }),
            /* @__PURE__ */ a("span", { "aria-hidden": "true" })
          ]
        }
      ),
      e.plugins.map((f) => {
        var v;
        const h = `plugin:${f.id}`;
        return /* @__PURE__ */ y(
          "button",
          {
            className: `module-switcher-option ${e.viewMode === h ? "active" : ""}`,
            type: "button",
            "aria-current": e.viewMode === h ? "page" : void 0,
            onClick: () => s(h),
            children: [
              /* @__PURE__ */ a("span", { children: f.label }),
              /* @__PURE__ */ a("span", { children: ((v = e.pluginState[f.id]) == null ? void 0 : v.total) ?? 0 })
            ]
          },
          f.id
        );
      })
    ] })
  ] });
}
function Mr(e) {
  const t = go(e.session), n = Be(e.session);
  return /* @__PURE__ */ y(
    "button",
    {
      className: `session-item ${e.nested ? "nested" : ""} ${e.active ? "active" : ""}`,
      type: "button",
      "aria-current": e.active ? "page" : void 0,
      title: `${t}
${e.session.key}`,
      onClick: e.onSelect,
      children: [
        /* @__PURE__ */ y("div", { className: "nav-item-row", children: [
          /* @__PURE__ */ a("span", { className: "nav-item-name", children: t }),
          /* @__PURE__ */ a("span", { className: "nav-item-count", title: `${e.session.message_count} 条消息`, children: e.session.message_count })
        ] }),
        /* @__PURE__ */ y("div", { className: "nav-item-desc", children: [
          /* @__PURE__ */ a("span", { children: po(n) }),
          /* @__PURE__ */ a("span", { "aria-hidden": "true", children: "·" }),
          /* @__PURE__ */ a("span", { children: ao(e.session.updated_at) })
        ] })
      ]
    }
  );
}
function vo(e) {
  const t = Be(e);
  return t === "scheduler" || t === "programmatic";
}
function Be(e) {
  return e.key.split(":", 1)[0] || "unknown";
}
function go(e) {
  return Zt(e.first_message_content).trim() || ft(e.key);
}
function po(e) {
  return {
    cli: "CLI",
    cross_mem: "Cross Memory",
    dashboard: "Dashboard",
    feishu: "飞书",
    mobile: "Mobile",
    programmatic: "Programmatic",
    qq: "QQ",
    qqbot: "QQ Bot",
    scheduler: "定时任务",
    telegram: "Telegram",
    web: "Web"
  }[e] || e;
}
function qt(e) {
  const t = H(null), n = Ce(() => e.state), r = Ce((s) => e.onSetState(s)), i = Ce(() => e.onActivate()), o = Ce((s) => e.onError(s)), l = Ce(() => e.startRead()), u = JSON.stringify(e.state.filters);
  return z(() => {
    if (t.current) {
      const s = t.current, c = Or(e.plugin, n, r, l, i, void 0, o);
      return en(s, e.plugin.id, e.slot, () => e.render(s, c));
    }
  }, [u, e.plugin, e.pluginId, e.redrawOnTotal, e.render, e.slot, e.state.sortBy, e.state.sortOrder]), lt(() => t.current ? e.plugin.applyStyle(t.current) : void 0, [e.plugin]), /* @__PURE__ */ a("div", { ref: t });
}
function yo(e) {
  var t;
  return /* @__PURE__ */ a("div", { className: "content-filters", children: e.viewMode.startsWith("plugin:") ? (t = e.currentPlugin) != null && t.renderFilters && e.currentPluginState && e.onSetPluginState ? /* @__PURE__ */ a(
    qt,
    {
      plugin: e.currentPlugin,
      pluginId: e.currentPlugin.id,
      render: e.currentPlugin.renderFilters,
      slot: "filters",
      state: e.currentPluginState,
      onSetState: e.onSetPluginState,
      startRead: () => e.startPluginRead(e.currentPlugin.id),
      onActivate: () => {
      },
      onError: e.onError
    }
  ) : null : /* @__PURE__ */ y("div", { className: "filter-row", children: [
    /* @__PURE__ */ y("label", { className: "search", children: [
      /* @__PURE__ */ a("span", { "aria-hidden": "true", children: "⌕" }),
      /* @__PURE__ */ a("input", { "aria-label": "搜索消息内容", type: "text", placeholder: "搜索消息内容", value: e.messageSearch, onChange: (n) => e.setMessageSearch(n.target.value.trim()) })
    ] }),
    /* @__PURE__ */ y("select", { "aria-label": "消息角色", value: e.messageRole, onChange: (n) => e.setMessageRole(n.target.value), children: [
      /* @__PURE__ */ a("option", { value: "", children: "全部 role" }),
      /* @__PURE__ */ a("option", { value: "user", children: "user" }),
      /* @__PURE__ */ a("option", { value: "assistant", children: "assistant" }),
      /* @__PURE__ */ a("option", { value: "system", children: "system" }),
      /* @__PURE__ */ a("option", { value: "tool", children: "tool" })
    ] }),
    e.activeSessionKey && /* @__PURE__ */ a(bo, { label: "session", value: e.activeSessionKey, onClear: e.clearSession })
  ] }) });
}
function bo(e) {
  return /* @__PURE__ */ y("div", { className: "active-session-chip", children: [
    /* @__PURE__ */ a("span", { children: e.label }),
    /* @__PURE__ */ a("code", { children: e.value }),
    /* @__PURE__ */ a("button", { "aria-label": `清除 ${e.label} 筛选`, type: "button", onClick: e.onClear, children: "×" })
  ] });
}
function So(e) {
  var t, n, r;
  if (e.viewMode.startsWith("plugin:") && e.plugin) {
    const i = !!((t = e.plugin.batchActions) != null && t.length), o = (i ? "32px " : "") + Tr(e.plugin.columns), l = ((n = e.pluginState) == null ? void 0 : n.sortBy) ?? "", u = ((r = e.pluginState) == null ? void 0 : r.sortOrder) ?? "desc";
    return /* @__PURE__ */ y("div", { className: "table-head", style: { gridTemplateColumns: o }, children: [
      i && /* @__PURE__ */ a("div", {}),
      e.plugin.columns.map(
        (s) => s.sortable && e.onPluginSort ? /* @__PURE__ */ a(Ie, { label: s.label, active: l === s.key, order: u, onClick: () => e.onPluginSort(s.key) }, s.key) : /* @__PURE__ */ a("div", { children: s.label }, s.key)
      )
    ] });
  }
  return /* @__PURE__ */ y("div", { className: "table-head mode-messages", children: [
    /* @__PURE__ */ a("div", {}),
    /* @__PURE__ */ a(Ie, { label: "Session Key", active: e.messageSortBy === "session_key", order: e.messageSortOrder, onClick: () => e.onSort("session_key") }),
    /* @__PURE__ */ a(Ie, { label: "Seq", active: e.messageSortBy === "seq", order: e.messageSortOrder, onClick: () => e.onSort("seq") }),
    /* @__PURE__ */ a("div", { children: "Content" }),
    /* @__PURE__ */ a(Ie, { label: "Timestamp", active: e.messageSortBy === "ts", order: e.messageSortOrder, onClick: () => e.onSort("ts") }),
    /* @__PURE__ */ a(Ie, { label: "Role", active: e.messageSortBy === "role", order: e.messageSortOrder, onClick: () => e.onSort("role") }),
    /* @__PURE__ */ a("div", {})
  ] });
}
function Ie(e) {
  return /* @__PURE__ */ y("button", { className: `table-sort-btn ${e.active ? "active" : ""}`, type: "button", onClick: e.onClick, children: [
    /* @__PURE__ */ a("span", { children: e.label }),
    /* @__PURE__ */ a("span", { className: "table-sort-arrow", children: e.active ? e.order === "asc" ? "↑" : "↓" : "" })
  ] });
}
function wo(e) {
  var t;
  if (e.viewMode.startsWith("plugin:") && e.plugin && e.pluginState) {
    const n = !!((t = e.plugin.batchActions) != null && t.length), r = (n ? "32px " : "") + Tr(e.plugin.columns);
    return /* @__PURE__ */ a(fe, { children: e.pluginState.items.length ? e.pluginState.items.map((i) => {
      var u, s;
      const o = String(i[e.plugin.rowKey] ?? ""), l = e.pluginState.selectedIds.has(o);
      return /* @__PURE__ */ y("div", { className: "table-row-wrap", children: [
        n && /* @__PURE__ */ a("label", { className: "checkbox-cell", children: /* @__PURE__ */ a("input", { "aria-label": `选择 ${e.plugin.label} 记录 ${o}`, type: "checkbox", checked: l, onChange: () => e.onTogglePluginRow(o) }) }),
        /* @__PURE__ */ y("button", { className: `table-row ${e.pluginState.activeRowKey === o ? "active" : ""} ${l ? "selected" : ""} ${((s = (u = e.plugin).rowClass) == null ? void 0 : s.call(u, i)) ?? ""}`, style: { gridTemplateColumns: r }, type: "button", "aria-expanded": e.pluginState.activeRowKey === o, onClick: () => e.onSelectPluginRow(i), children: [
          n && /* @__PURE__ */ a("span", { "aria-hidden": "true" }),
          e.plugin.columns.map((c) => {
            const f = Po(c);
            return c.renderCell ? /* @__PURE__ */ a("span", { className: f, title: c.rawTitle ? String(i[c.key] ?? "") : void 0, dangerouslySetInnerHTML: { __html: c.renderCell(i[c.key], i) } }, c.key) : /* @__PURE__ */ a("span", { className: f, title: c.rawTitle ? String(i[c.key] ?? "") : void 0, children: No(e.plugin, c, i) }, c.key);
          })
        ] })
      ] }, o);
    }) : /* @__PURE__ */ a("div", { className: "empty-state", children: e.plugin.emptyMessage || "暂无记录。" }) });
  }
  return /* @__PURE__ */ a(fe, { children: e.messages.map((n) => {
    var r, i;
    return /* @__PURE__ */ y("div", { className: "table-row-wrap", children: [
      /* @__PURE__ */ a("label", { className: "checkbox-cell", children: /* @__PURE__ */ a("input", { "aria-label": `选择消息 ${n.seq}`, type: "checkbox", checked: e.selectedMessageIds.has(n.id), onChange: (o) => Eo(n.id, o.target.checked, e.selectedMessageIds, e.setSelectedMessageIds) }) }),
      /* @__PURE__ */ y("button", { className: `table-row mode-messages ${((r = e.activeMessage) == null ? void 0 : r.id) === n.id ? "active" : ""} ${e.selectedMessageIds.has(n.id) ? "selected" : ""}`, type: "button", "aria-expanded": ((i = e.activeMessage) == null ? void 0 : i.id) === n.id, onClick: () => e.onSelectMessage(n), children: [
        /* @__PURE__ */ a("span", { "aria-hidden": "true" }),
        /* @__PURE__ */ a("span", { className: "cell-session mono", title: n.session_key, children: ft(n.session_key) }),
        /* @__PURE__ */ y("span", { className: "cell-seq mono", children: [
          "#",
          n.seq
        ] }),
        /* @__PURE__ */ a("span", { className: "content-preview", children: Zt(n.content) }),
        /* @__PURE__ */ a("span", { className: "cell-time mono", children: Qt(n.timestamp) }),
        /* @__PURE__ */ a("span", { children: /* @__PURE__ */ a("span", { className: `role-pill ${Dr(n.role)}`, children: n.role }) }),
        /* @__PURE__ */ a("span", { "aria-hidden": "true" })
      ] })
    ] }, n.id);
  }) });
}
function Co(e) {
  var t;
  if (e.loading) return /* @__PURE__ */ a(_r, {});
  if (e.viewMode.startsWith("plugin:") && e.plugin)
    return /* @__PURE__ */ a(oo, { plugin: e.plugin, item: ((t = e.pluginState) == null ? void 0 : t.activeDetail) ?? null, dispatch: e.dispatch });
  if (e.activeMessage) {
    const n = e.activeMessage;
    return /* @__PURE__ */ y("div", { className: "detail-wrap", children: [
      /* @__PURE__ */ y("div", { className: "detail-toolbar", children: [
        /* @__PURE__ */ y("div", { children: [
          /* @__PURE__ */ a("div", { className: "detail-title", children: "消息详情" }),
          /* @__PURE__ */ y("div", { className: "detail-subtext", children: [
            n.session_key,
            " · #",
            n.seq
          ] })
        ] }),
        /* @__PURE__ */ a(ct, { variant: "standard", label: "关闭详情", onClick: e.onClose, children: /* @__PURE__ */ a(An, { size: 18, "aria-hidden": "true" }) })
      ] }),
      /* @__PURE__ */ y("div", { className: "detail-grid", children: [
        U("role", /* @__PURE__ */ a("span", { className: `role-pill ${Dr(n.role)}`, children: n.role })),
        U("time", /* @__PURE__ */ a("code", { children: n.timestamp })),
        U("id", /* @__PURE__ */ a("code", { children: n.id }))
      ] }),
      /* @__PURE__ */ y("div", { className: "detail-block", children: [
        /* @__PURE__ */ a("div", { className: "detail-label", children: "Content" }),
        /* @__PURE__ */ a(Ln, { className: "detail-content", children: n.content })
      ] }),
      /* @__PURE__ */ y("div", { className: "detail-block", children: [
        /* @__PURE__ */ a("div", { className: "detail-label", children: "Extra" }),
        /* @__PURE__ */ a(st, { data: n.extra })
      ] }),
      /* @__PURE__ */ y("div", { className: "detail-block", children: [
        /* @__PURE__ */ a("div", { className: "detail-label", children: "Tool Chain" }),
        /* @__PURE__ */ a(st, { data: n.tool_chain })
      ] })
    ] });
  }
  if (e.activeSession) {
    const n = e.activeSession;
    return /* @__PURE__ */ y("div", { className: "detail-wrap", children: [
      /* @__PURE__ */ y("div", { className: "detail-toolbar", children: [
        /* @__PURE__ */ y("div", { children: [
          /* @__PURE__ */ a("div", { className: "detail-title", children: "Session 详情" }),
          /* @__PURE__ */ a("div", { className: "detail-subtext", children: n.key })
        ] }),
        /* @__PURE__ */ a(ct, { variant: "standard", label: "关闭详情", onClick: e.onClose, children: /* @__PURE__ */ a(An, { size: 18, "aria-hidden": "true" }) })
      ] }),
      /* @__PURE__ */ y("div", { className: "detail-grid", children: [
        U("messages", /* @__PURE__ */ a("code", { children: n.message_count })),
        U("updated", /* @__PURE__ */ a("code", { children: n.updated_at })),
        U("last_consolidated", /* @__PURE__ */ a("code", { children: n.last_consolidated }))
      ] }),
      /* @__PURE__ */ y("div", { className: "detail-block", children: [
        /* @__PURE__ */ a("div", { className: "detail-label", children: "Metadata" }),
        /* @__PURE__ */ a(st, { data: n.metadata })
      ] })
    ] });
  }
  return /* @__PURE__ */ a(ot, { text: "点开消息、session 或 memory 后，这里会显示完整内容、字段和 JSON 信息。" });
}
function _r() {
  return /* @__PURE__ */ y("div", { className: "detail-loading", role: "status", "aria-label": "正在加载详情", children: [
    Tn.createElement("md-linear-progress", { className: "detail-loading-progress", indeterminate: !0, "aria-label": "正在加载详情" }),
    /* @__PURE__ */ a("div", { className: "detail-loading-line detail-loading-line-short" }),
    /* @__PURE__ */ a("div", { className: "detail-loading-line detail-loading-line-title" }),
    /* @__PURE__ */ a("div", { className: "detail-loading-block" }),
    /* @__PURE__ */ a("div", { className: "detail-loading-line" }),
    /* @__PURE__ */ a("div", { className: "detail-loading-line" })
  ] });
}
function ot(e) {
  return /* @__PURE__ */ y("div", { className: "detail-empty", children: [
    /* @__PURE__ */ a("div", { className: "detail-empty-title", children: "详情" }),
    /* @__PURE__ */ a("div", { className: "detail-empty-text", children: e.text })
  ] });
}
function U(e, t) {
  return /* @__PURE__ */ y("div", { className: "detail-row", children: [
    /* @__PURE__ */ a("div", { className: "detail-row-label", children: e }),
    /* @__PURE__ */ a("div", { className: "detail-row-val", children: t })
  ] });
}
function st(e) {
  return /* @__PURE__ */ a(ti, { value: e.data });
}
function Eo(e, t, n, r) {
  const i = new Set(n);
  t ? i.add(e) : i.delete(e), r(i);
}
function Tr(e) {
  return e.map((t) => t.flex ? "minmax(0, 1fr)" : t.width ? `minmax(0, ${t.width}px)` : "minmax(0, auto)").join(" ");
}
function No(e, t, n) {
  var o;
  const r = n[t.key], i = ((o = e.formatters) == null ? void 0 : o[t.fmt || ""]) ?? lo[t.fmt || "text"];
  return i ? i(r, n) : String(r ?? "");
}
function Po(e) {
  const t = [e.cellClass ?? ""];
  return !e.cellClass && e.fmt === "text-preview" && t.push("content-preview"), !e.cellClass && (e.fmt === "mono-session" || e.fmt === "mono-time") && t.push(e.fmt === "mono-session" ? "mono cell-session" : "mono cell-time"), e.align === "right" && t.push("align-right"), t.filter(Boolean).join(" ");
}
function Ro(e, t, n) {
  return t && n ? t.countTitle ? t.countTitle(n.total) : `共 ${n.total} 条` : `共 ${e} 条`;
}
function _n(e) {
  return e === "context_overflow" ? "overflow" : e || "unknown";
}
function Ao(e) {
  if (!e.activeSessionKey)
    return /* @__PURE__ */ a(ot, { text: "从左侧选择一个 session，查看其上下文压缩状态。" });
  if (e.pending && !e.compaction)
    return /* @__PURE__ */ a(_r, {});
  if (!e.compaction)
    return /* @__PURE__ */ a(ot, { text: "加载失败，请重试。" });
  const { head: t, active: n, history: r } = e.compaction;
  return /* @__PURE__ */ a("div", { className: "compaction-view-scroll", children: /* @__PURE__ */ y("div", { className: "detail-wrap", children: [
    /* @__PURE__ */ a("div", { className: "detail-toolbar", children: /* @__PURE__ */ y("div", { children: [
      /* @__PURE__ */ a("div", { className: "detail-title", children: "Compaction" }),
      /* @__PURE__ */ y("div", { className: "detail-subtext", children: [
        ft(e.activeSessionKey),
        " · ",
        n ? `generation ${n.generation} · 下一代 ${t.next_generation}` : "尚未压缩"
      ] })
    ] }) }),
    n ? /* @__PURE__ */ y(fe, { children: [
      /* @__PURE__ */ y("div", { className: "detail-grid", children: [
        U("generation", /* @__PURE__ */ a("code", { children: n.generation })),
        U("source", /* @__PURE__ */ y("code", { children: [
          n.source_from_seq,
          " → ",
          n.consolidated_through_seq
        ] })),
        U("messages", /* @__PURE__ */ a("code", { children: n.source_message_count })),
        U("tokens", /* @__PURE__ */ y("code", { children: [
          le(n.tokens_before),
          " → ",
          le(n.tokens_after)
        ] })),
        U("threshold", /* @__PURE__ */ y("code", { children: [
          "soft ",
          le(n.threshold_tokens),
          " · hard ",
          le(n.hard_input_tokens),
          " · tail ",
          le(n.keep_recent_tokens)
        ] })),
        U("model", /* @__PURE__ */ a("code", { children: n.model })),
        U("window", /* @__PURE__ */ a("code", { children: le(n.context_window) })),
        U("trigger", /* @__PURE__ */ a("span", { className: "status-pill", children: _n(n.trigger) })),
        U("created", /* @__PURE__ */ a("code", { children: n.created_at }))
      ] }),
      /* @__PURE__ */ y("div", { className: "detail-block", children: [
        /* @__PURE__ */ a("div", { className: "detail-label", children: "当前摘要" }),
        /* @__PURE__ */ a(Ln, { className: "detail-content", children: n.summary })
      ] }),
      /* @__PURE__ */ y("div", { className: "detail-block", children: [
        /* @__PURE__ */ a("div", { className: "detail-label", children: "Summary Usage" }),
        /* @__PURE__ */ a(st, { data: n.summary_usage })
      ] }),
      /* @__PURE__ */ y("div", { className: "detail-block", children: [
        /* @__PURE__ */ a("div", { className: "detail-label", children: "Source Plan Digest" }),
        /* @__PURE__ */ a("div", { className: "detail-content mono compaction-digest", children: n.source_plan_digest })
      ] })
    ] }) : /* @__PURE__ */ a(ot, { text: "该 session 尚未发生压缩——模型上下文达到 74% 水位后自动生成摘要。" }),
    r.length > 0 && /* @__PURE__ */ y("div", { className: "detail-block", children: [
      /* @__PURE__ */ a("div", { className: "detail-label", children: "历史 generations" }),
      r.map((i) => /* @__PURE__ */ y("div", { className: "compaction-history-row", children: [
        /* @__PURE__ */ y("code", { children: [
          "gen ",
          i.generation
        ] }),
        /* @__PURE__ */ a("span", { className: "muted-text", children: Qt(i.created_at) }),
        /* @__PURE__ */ y("code", { children: [
          le(i.tokens_before),
          " → ",
          le(i.tokens_after)
        ] }),
        /* @__PURE__ */ a("span", { className: "status-pill", children: _n(i.trigger) }),
        i.invalidated_at ? /* @__PURE__ */ a("span", { className: "type-pill compaction-invalidated", title: i.invalidated_reason ?? void 0, children: "已失效" }) : /* @__PURE__ */ a("span", { className: "type-pill compaction-valid", children: "有效" })
      ] }, i.generation))
    ] })
  ] }) });
}
function Lo(e) {
  const t = ro(e.http.request), n = e.ui.inject("shell.pages.v1", (r) => r.register({
    id: "workbench",
    label: "工作台",
    route: "workbench",
    order: 20,
    iconSvg: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-gauge" aria-hidden="true"><path d="m12 14 4-4"></path><path d="M3.34 19a10 10 0 1 1 17.32 0"></path></svg>',
    children: [{ id: "workbench.panels.v2", cardinality: "list" }],
    render(i, o) {
      const l = o.child("workbench.panels.v2"), u = l.entries.map((c) => ({
        ...Do(c),
        applyStyle: (f) => l.style(c.id, f)
      })), s = ei(i);
      return s.render(/* @__PURE__ */ a(mo, { initialPlugins: u })), () => s.unmount();
    }
  }));
  return () => {
    n(), t();
  };
}
function Do(e) {
  const t = e;
  if (typeof t.id != "string" || typeof t.label != "string" || typeof t.rowKey != "string" || !Array.isArray(t.columns) || typeof t.getCount != "function" || typeof t.fetchPage != "function")
    throw new Error(`工作台面板合同无效: ${String(e.id ?? "unknown")}`);
  return t;
}
export {
  Lo as activate
};
