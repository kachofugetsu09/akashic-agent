var xc = Object.defineProperty;
var d = (r, e) => {
  for (var o in e) xc(r, o, { get: e[o], enumerable: !0 });
};
var _c = (r) => {
    if (r === null) return null;
    if (typeof r == "string" || typeof r == "boolean") return r;
    if (typeof r == "number") return Number.isFinite(r) ? r : void 0;
    if (r instanceof String || r instanceof Boolean) return r.valueOf();
    if (r instanceof Number) {
      let e = r.valueOf();
      return Number.isFinite(e) ? e : void 0;
    }
  },
  Cc = (r) => typeof r == "number" && Number.isInteger(r) && r > 0,
  Yo = (r, e) => {
    if (!Cc(e)) return r;
    let o = e;
    if (r.length <= o) return r;
    let a = r.length - o;
    return `${r.slice(0, o)}[truncated ${a} chars]`;
  },
  Sc = (r, e) => {
    switch (r.type) {
      case "function":
        return r;
      case "error":
      case "object":
        return { ...r, value: Yo(r.value, e) };
      case "value":
        return typeof r.value != "string" ? r : { ...r, value: Yo(r.value, e) };
    }
  },
  vc = async (r, e, o) => {
    let { displayImage: a, displayValue: n } = r,
      s = null;
    try {
      if (e instanceof Uint8Array) {
        await a(e);
        return;
      } else if (e === void 0) s = { type: "value", value: void 0 };
      else if (typeof e == "function") s = { type: "function", value: void 0 };
      else {
        let c = _c(e);
        c !== void 0 && (s = { type: "value", value: c });
      }
      if (s === null) {
        let c = JSON.stringify(e);
        (c === void 0 && (c = String(e)), (s = { type: "object", value: c }));
      }
    } catch (c) {
      s = { type: "error", value: String(c) };
    }
    s && (await n(Sc(s, o)));
  };
function Qo({ displayBridge: r, displayTruncateMaxChars: e }) {
  return (o) => vc(r, o, e);
}
var Jo = {};
d(Jo, {
  AXAPI: () => Qe,
  Agent: () => Ve,
  AlertDialog: () => Vr,
  AtlasCommand: () => i,
  BeforeUnloadDialog: () => Zr,
  BotDetectionTabCapability: () => nt,
  Browser: () => wt,
  BrowserAuthTabCapability: () => dt,
  BrowserManagementCommands: () => xo,
  BrowserUser: () => Ge,
  BrowserViewportCommands: () => Ao,
  BrowserVisibilityCommands: () => vo,
  Browsers: () => We,
  CUAAPI: () => Ye,
  CdpTabCapability: () => ct,
  Commands: () => P,
  ConfirmDialog: () => Gr,
  ContentAPI: () => et,
  Documentation: () => Q,
  DomCUAAPI: () => Ke,
  FunctionAgentTransport: () => xt,
  ManagementBrowserCapability: () => ht,
  PageAssetsTabCapability: () => ut,
  PlaywrightAPI: () => Xe,
  PlaywrightDownload: () => Ur,
  PlaywrightFileChooser: () => Wr,
  PlaywrightFrameLocator: () => qr,
  PlaywrightLocator: () => H,
  PromptDialog: () => Hr,
  Tab: () => te,
  TabBotDetectionCommands: () => no,
  TabBrowserAuthCommands: () => po,
  TabCdpCommands: () => co,
  TabClipboardAPI: () => tt,
  TabDevAPI: () => rt,
  TabPageAssetsCommands: () => uo,
  TabWebMcpCapability: () => ft,
  Tabs: () => Ze,
  ViewportBrowserCapability: () => gt,
  VisibilityBrowserCapability: () => bt,
  botDetectionTabCapabilityDefinition: () => so,
  browserAuthTabCapabilityDefinition: () => mo,
  browserCapabilityDefinitions: () => Ho,
  cdpTabCapabilityDefinition: () => lo,
  generateRequestId: () => vi,
  hasErrorMessage: () => Si,
  isPlainObject: () => ro,
  isRegExp: () => He,
  managementBrowserCapabilityDefinition: () => _o,
  pageAssetsTabCapabilityDefinition: () => fo,
  tabCapabilityDefinitions: () => Go,
  toWebMcpToolDescriptor: () => ho,
  viewportBrowserCapabilityDefinition: () => Io,
  visibilityBrowserCapabilityDefinition: () => To,
  webMcpTabCapabilityDefinition: () => bo,
  webMcpTabCapabilityInfo: () => yo,
});
var P = {};
d(P, {
  AtlasCommand: () => i,
  BrowserUserClaimTab: () => qt,
  BrowserUserHistory: () => Wt,
  BrowserUserOpenTabs: () => Ut,
  CloseTab: () => Ot,
  CreateTab: () => jt,
  CuaClick: () => rr,
  CuaDoubleClick: () => ar,
  CuaDownloadMedia: () => or,
  CuaDrag: () => sr,
  CuaKeypress: () => nr,
  CuaMove: () => ir,
  CuaScroll: () => cr,
  CuaType: () => lr,
  DomCuaClick: () => qe,
  DomCuaDoubleClick: () => pr,
  DomCuaDownloadMedia: () => dr,
  DomCuaGetVisibleDom: () => mr,
  DomCuaKeypress: () => ur,
  DomCuaScroll: () => fr,
  DomCuaType: () => yr,
  GetBrowser: () => Pt,
  GetBrowserDocumentation: () => It,
  GetBrowserForUrl: () => kt,
  GetDefaultBrowser: () => Rt,
  GetDocumentation: () => At,
  GetTab: () => ge,
  ListBrowsers: () => Tt,
  ListTabs: () => Mt,
  ManualHandoff: () => zt,
  MarkTab: () => Bt,
  NameSession: () => Et,
  NavigateTab: () => Nt,
  NavigateTabBack: () => we,
  NavigateTabForward: () => xe,
  NavigateTabReload: () => $t,
  PlaywrightDomSnapshot: () => Er,
  PlaywrightDownloadPath: () => Fr,
  PlaywrightElementInfo: () => Dr,
  PlaywrightElementScreenshot: () => Mr,
  PlaywrightEvaluate: () => Ce,
  PlaywrightFileChooserSetFiles: () => $r,
  PlaywrightLocatorAllTextContents: () => Rr,
  PlaywrightLocatorClick: () => hr,
  PlaywrightLocatorCount: () => Cr,
  PlaywrightLocatorDblclick: () => br,
  PlaywrightLocatorDownloadMedia: () => gr,
  PlaywrightLocatorFill: () => Ue,
  PlaywrightLocatorGetAttribute: () => kr,
  PlaywrightLocatorInnerText: () => Ir,
  PlaywrightLocatorIsEnabled: () => Pr,
  PlaywrightLocatorIsVisible: () => Tr,
  PlaywrightLocatorPress: () => wr,
  PlaywrightLocatorPressSequentially: () => xr,
  PlaywrightLocatorReadAll: () => jr,
  PlaywrightLocatorSelectOption: () => Sr,
  PlaywrightLocatorSetChecked: () => vr,
  PlaywrightLocatorTextContent: () => Ar,
  PlaywrightLocatorWaitFor: () => _r,
  PlaywrightWaitForDownload: () => Lr,
  PlaywrightWaitForFileChooser: () => Nr,
  PlaywrightWaitForLoadState: () => Br,
  PlaywrightWaitForTimeout: () => zr,
  PlaywrightWaitForURL: () => Or,
  RuntimeConfig: () => Eo,
  SelectedTab: () => Dt,
  TabAxAction: () => tr,
  TabAxGetState: () => er,
  TabClipboardRead: () => Kt,
  TabClipboardReadText: () => Jt,
  TabClipboardWrite: () => Yt,
  TabClipboardWriteText: () => Xt,
  TabContentExport: () => Zt,
  TabContentExportGSuite: () => Gt,
  TabContentExportYouTubeTranscript: () => Ht,
  TabDevLogs: () => $e,
  TabGetJsDialog: () => Vt,
  TabHandleJsDialog: () => _e,
  TabScreenshot: () => Qt,
  TabsContent: () => Lt,
  WebMcpInvokeTool: () => Ft,
  WebMcpListTools: () => Ne,
});
var i = class {
  constructor(e, o, a) {
    ((this.type = e), (this.schema = o), (this.payload = a));
  }
  parse() {
    return this.schema.parse(this.payload);
  }
  toJSON() {
    return { type: this.type, ...this.payload };
  }
};
var Tt = {};
d(Tt, {
  PayloadSchema: () => ma,
  ResultSchema: () => Ll,
  commandType: () => ua,
  create: () => Fl,
});
var t = {};
d(t, {
  BRAND: () => Xc,
  DIRTY: () => re,
  EMPTY_PATH: () => Ac,
  INVALID: () => b,
  NEVER: () => zl,
  OK: () => T,
  ParseStatus: () => v,
  Schema: () => _,
  ZodAny: () => K,
  ZodArray: () => G,
  ZodBigInt: () => ae,
  ZodBoolean: () => ne,
  ZodBranded: () => St,
  ZodCatch: () => he,
  ZodDate: () => se,
  ZodDefault: () => ye,
  ZodDiscriminatedUnion: () => Yr,
  ZodEffects: () => O,
  ZodEnum: () => ue,
  ZodError: () => A,
  ZodFirstPartyTypeKind: () => g,
  ZodFunction: () => eo,
  ZodIntersection: () => de,
  ZodIssueCode: () => m,
  ZodLazy: () => pe,
  ZodLiteral: () => me,
  ZodMap: () => De,
  ZodNaN: () => Le,
  ZodNativeEnum: () => fe,
  ZodNever: () => E,
  ZodNull: () => ce,
  ZodNullable: () => F,
  ZodNumber: () => oe,
  ZodObject: () => I,
  ZodOptional: () => k,
  ZodParsedType: () => y,
  ZodPipeline: () => vt,
  ZodPromise: () => Y,
  ZodReadonly: () => be,
  ZodRecord: () => Qr,
  ZodSchema: () => _,
  ZodSet: () => Me,
  ZodString: () => X,
  ZodSymbol: () => ze,
  ZodTransformer: () => O,
  ZodTuple: () => L,
  ZodType: () => _,
  ZodUndefined: () => ie,
  ZodUnion: () => le,
  ZodUnknown: () => Z,
  ZodVoid: () => Ee,
  addIssueToContext: () => f,
  any: () => nl,
  array: () => ll,
  bigint: () => el,
  boolean: () => da,
  coerce: () => Bl,
  custom: () => ia,
  date: () => tl,
  datetimeRegex: () => na,
  defaultErrorMap: () => W,
  discriminatedUnion: () => ul,
  effect: () => Tl,
  enum: () => Cl,
  function: () => wl,
  getErrorMap: () => je,
  getParsedType: () => M,
  instanceof: () => Yc,
  intersection: () => fl,
  isAborted: () => Xr,
  isAsync: () => Oe,
  isDirty: () => Kr,
  isValid: () => J,
  late: () => Kc,
  lazy: () => xl,
  literal: () => _l,
  makeIssue: () => Ct,
  map: () => bl,
  nan: () => Qc,
  nativeEnum: () => Sl,
  never: () => il,
  null: () => al,
  nullable: () => Rl,
  number: () => la,
  object: () => dl,
  objectUtil: () => jo,
  oboolean: () => Ol,
  onumber: () => jl,
  optional: () => Pl,
  ostring: () => kl,
  pipeline: () => Il,
  preprocess: () => Al,
  promise: () => vl,
  quotelessJson: () => Tc,
  record: () => hl,
  set: () => gl,
  setErrorMap: () => Rc,
  strictObject: () => pl,
  string: () => ca,
  symbol: () => rl,
  transformer: () => Tl,
  tuple: () => yl,
  undefined: () => ol,
  union: () => ml,
  unknown: () => sl,
  util: () => C,
  void: () => cl,
});
var C;
(function (r) {
  r.assertEqual = (n) => {};
  function e(n) {}
  r.assertIs = e;
  function o(n) {
    throw new Error();
  }
  ((r.assertNever = o),
    (r.arrayToEnum = (n) => {
      let s = {};
      for (let c of n) s[c] = c;
      return s;
    }),
    (r.getValidEnumValues = (n) => {
      let s = r.objectKeys(n).filter((l) => typeof n[n[l]] != "number"),
        c = {};
      for (let l of s) c[l] = n[l];
      return r.objectValues(c);
    }),
    (r.objectValues = (n) =>
      r.objectKeys(n).map(function (s) {
        return n[s];
      })),
    (r.objectKeys =
      typeof Object.keys == "function"
        ? (n) => Object.keys(n)
        : (n) => {
            let s = [];
            for (let c in n)
              Object.prototype.hasOwnProperty.call(n, c) && s.push(c);
            return s;
          }),
    (r.find = (n, s) => {
      for (let c of n) if (s(c)) return c;
    }),
    (r.isInteger =
      typeof Number.isInteger == "function"
        ? (n) => Number.isInteger(n)
        : (n) =>
            typeof n == "number" && Number.isFinite(n) && Math.floor(n) === n));
  function a(n, s = " | ") {
    return n.map((c) => (typeof c == "string" ? `'${c}'` : c)).join(s);
  }
  ((r.joinValues = a),
    (r.jsonStringifyReplacer = (n, s) =>
      typeof s == "bigint" ? s.toString() : s));
})(C || (C = {}));
var jo;
(function (r) {
  r.mergeShapes = (e, o) => ({ ...e, ...o });
})(jo || (jo = {}));
var y = C.arrayToEnum([
    "string",
    "nan",
    "number",
    "integer",
    "float",
    "boolean",
    "date",
    "bigint",
    "symbol",
    "function",
    "undefined",
    "null",
    "array",
    "object",
    "unknown",
    "promise",
    "void",
    "never",
    "map",
    "set",
  ]),
  M = (r) => {
    switch (typeof r) {
      case "undefined":
        return y.undefined;
      case "string":
        return y.string;
      case "number":
        return Number.isNaN(r) ? y.nan : y.number;
      case "boolean":
        return y.boolean;
      case "function":
        return y.function;
      case "bigint":
        return y.bigint;
      case "symbol":
        return y.symbol;
      case "object":
        return Array.isArray(r)
          ? y.array
          : r === null
            ? y.null
            : r.then &&
                typeof r.then == "function" &&
                r.catch &&
                typeof r.catch == "function"
              ? y.promise
              : typeof Map < "u" && r instanceof Map
                ? y.map
                : typeof Set < "u" && r instanceof Set
                  ? y.set
                  : typeof Date < "u" && r instanceof Date
                    ? y.date
                    : y.object;
      default:
        return y.unknown;
    }
  };
var m = C.arrayToEnum([
    "invalid_type",
    "invalid_literal",
    "custom",
    "invalid_union",
    "invalid_union_discriminator",
    "invalid_enum_value",
    "unrecognized_keys",
    "invalid_arguments",
    "invalid_return_type",
    "invalid_date",
    "invalid_string",
    "too_small",
    "too_big",
    "invalid_intersection_types",
    "not_multiple_of",
    "not_finite",
  ]),
  Tc = (r) => JSON.stringify(r, null, 2).replace(/"([^"]+)":/g, "$1:"),
  A = class r extends Error {
    get errors() {
      return this.issues;
    }
    constructor(e) {
      (super(),
        (this.issues = []),
        (this.addIssue = (a) => {
          this.issues = [...this.issues, a];
        }),
        (this.addIssues = (a = []) => {
          this.issues = [...this.issues, ...a];
        }));
      let o = new.target.prototype;
      (Object.setPrototypeOf
        ? Object.setPrototypeOf(this, o)
        : (this.__proto__ = o),
        (this.name = "ZodError"),
        (this.issues = e));
    }
    format(e) {
      let o =
          e ||
          function (s) {
            return s.message;
          },
        a = { _errors: [] },
        n = (s) => {
          for (let c of s.issues)
            if (c.code === "invalid_union") c.unionErrors.map(n);
            else if (c.code === "invalid_return_type") n(c.returnTypeError);
            else if (c.code === "invalid_arguments") n(c.argumentsError);
            else if (c.path.length === 0) a._errors.push(o(c));
            else {
              let l = a,
                u = 0;
              for (; u < c.path.length; ) {
                let p = c.path[u];
                (u === c.path.length - 1
                  ? ((l[p] = l[p] || { _errors: [] }), l[p]._errors.push(o(c)))
                  : (l[p] = l[p] || { _errors: [] }),
                  (l = l[p]),
                  u++);
              }
            }
        };
      return (n(this), a);
    }
    static assert(e) {
      if (!(e instanceof r)) throw new Error(`Not a ZodError: ${e}`);
    }
    toString() {
      return this.message;
    }
    get message() {
      return JSON.stringify(this.issues, C.jsonStringifyReplacer, 2);
    }
    get isEmpty() {
      return this.issues.length === 0;
    }
    flatten(e = (o) => o.message) {
      let o = {},
        a = [];
      for (let n of this.issues)
        if (n.path.length > 0) {
          let s = n.path[0];
          ((o[s] = o[s] || []), o[s].push(e(n)));
        } else a.push(e(n));
      return { formErrors: a, fieldErrors: o };
    }
    get formErrors() {
      return this.flatten();
    }
  };
A.create = (r) => new A(r);
var Pc = (r, e) => {
    let o;
    switch (r.code) {
      case m.invalid_type:
        r.received === y.undefined
          ? (o = "Required")
          : (o = `Expected ${r.expected}, received ${r.received}`);
        break;
      case m.invalid_literal:
        o = `Invalid literal value, expected ${JSON.stringify(r.expected, C.jsonStringifyReplacer)}`;
        break;
      case m.unrecognized_keys:
        o = `Unrecognized key(s) in object: ${C.joinValues(r.keys, ", ")}`;
        break;
      case m.invalid_union:
        o = "Invalid input";
        break;
      case m.invalid_union_discriminator:
        o = `Invalid discriminator value. Expected ${C.joinValues(r.options)}`;
        break;
      case m.invalid_enum_value:
        o = `Invalid enum value. Expected ${C.joinValues(r.options)}, received '${r.received}'`;
        break;
      case m.invalid_arguments:
        o = "Invalid function arguments";
        break;
      case m.invalid_return_type:
        o = "Invalid function return type";
        break;
      case m.invalid_date:
        o = "Invalid date";
        break;
      case m.invalid_string:
        typeof r.validation == "object"
          ? "includes" in r.validation
            ? ((o = `Invalid input: must include "${r.validation.includes}"`),
              typeof r.validation.position == "number" &&
                (o = `${o} at one or more positions greater than or equal to ${r.validation.position}`))
            : "startsWith" in r.validation
              ? (o = `Invalid input: must start with "${r.validation.startsWith}"`)
              : "endsWith" in r.validation
                ? (o = `Invalid input: must end with "${r.validation.endsWith}"`)
                : C.assertNever(r.validation)
          : r.validation !== "regex"
            ? (o = `Invalid ${r.validation}`)
            : (o = "Invalid");
        break;
      case m.too_small:
        r.type === "array"
          ? (o = `Array must contain ${r.exact ? "exactly" : r.inclusive ? "at least" : "more than"} ${r.minimum} element(s)`)
          : r.type === "string"
            ? (o = `String must contain ${r.exact ? "exactly" : r.inclusive ? "at least" : "over"} ${r.minimum} character(s)`)
            : r.type === "number"
              ? (o = `Number must be ${r.exact ? "exactly equal to " : r.inclusive ? "greater than or equal to " : "greater than "}${r.minimum}`)
              : r.type === "bigint"
                ? (o = `Number must be ${r.exact ? "exactly equal to " : r.inclusive ? "greater than or equal to " : "greater than "}${r.minimum}`)
                : r.type === "date"
                  ? (o = `Date must be ${r.exact ? "exactly equal to " : r.inclusive ? "greater than or equal to " : "greater than "}${new Date(Number(r.minimum))}`)
                  : (o = "Invalid input");
        break;
      case m.too_big:
        r.type === "array"
          ? (o = `Array must contain ${r.exact ? "exactly" : r.inclusive ? "at most" : "less than"} ${r.maximum} element(s)`)
          : r.type === "string"
            ? (o = `String must contain ${r.exact ? "exactly" : r.inclusive ? "at most" : "under"} ${r.maximum} character(s)`)
            : r.type === "number"
              ? (o = `Number must be ${r.exact ? "exactly" : r.inclusive ? "less than or equal to" : "less than"} ${r.maximum}`)
              : r.type === "bigint"
                ? (o = `BigInt must be ${r.exact ? "exactly" : r.inclusive ? "less than or equal to" : "less than"} ${r.maximum}`)
                : r.type === "date"
                  ? (o = `Date must be ${r.exact ? "exactly" : r.inclusive ? "smaller than or equal to" : "smaller than"} ${new Date(Number(r.maximum))}`)
                  : (o = "Invalid input");
        break;
      case m.custom:
        o = "Invalid input";
        break;
      case m.invalid_intersection_types:
        o = "Intersection results could not be merged";
        break;
      case m.not_multiple_of:
        o = `Number must be a multiple of ${r.multipleOf}`;
        break;
      case m.not_finite:
        o = "Number must be finite";
        break;
      default:
        ((o = e.defaultError), C.assertNever(r));
    }
    return { message: o };
  },
  W = Pc;
var ea = W;
function Rc(r) {
  ea = r;
}
function je() {
  return ea;
}
var Ct = (r) => {
    let { data: e, path: o, errorMaps: a, issueData: n } = r,
      s = [...o, ...(n.path || [])],
      c = { ...n, path: s };
    if (n.message !== void 0) return { ...n, path: s, message: n.message };
    let l = "",
      u = a
        .filter((p) => !!p)
        .slice()
        .reverse();
    for (let p of u) l = p(c, { data: e, defaultError: l }).message;
    return { ...n, path: s, message: l };
  },
  Ac = [];
function f(r, e) {
  let o = je(),
    a = Ct({
      issueData: e,
      data: r.data,
      path: r.path,
      errorMaps: [
        r.common.contextualErrorMap,
        r.schemaErrorMap,
        o,
        o === W ? void 0 : W,
      ].filter((n) => !!n),
    });
  r.common.issues.push(a);
}
var v = class r {
    constructor() {
      this.value = "valid";
    }
    dirty() {
      this.value === "valid" && (this.value = "dirty");
    }
    abort() {
      this.value !== "aborted" && (this.value = "aborted");
    }
    static mergeArray(e, o) {
      let a = [];
      for (let n of o) {
        if (n.status === "aborted") return b;
        (n.status === "dirty" && e.dirty(), a.push(n.value));
      }
      return { status: e.value, value: a };
    }
    static async mergeObjectAsync(e, o) {
      let a = [];
      for (let n of o) {
        let s = await n.key,
          c = await n.value;
        a.push({ key: s, value: c });
      }
      return r.mergeObjectSync(e, a);
    }
    static mergeObjectSync(e, o) {
      let a = {};
      for (let n of o) {
        let { key: s, value: c } = n;
        if (s.status === "aborted" || c.status === "aborted") return b;
        (s.status === "dirty" && e.dirty(),
          c.status === "dirty" && e.dirty(),
          s.value !== "__proto__" &&
            (typeof c.value < "u" || n.alwaysSet) &&
            (a[s.value] = c.value));
      }
      return { status: e.value, value: a };
    }
  },
  b = Object.freeze({ status: "aborted" }),
  re = (r) => ({ status: "dirty", value: r }),
  T = (r) => ({ status: "valid", value: r }),
  Xr = (r) => r.status === "aborted",
  Kr = (r) => r.status === "dirty",
  J = (r) => r.status === "valid",
  Oe = (r) => typeof Promise < "u" && r instanceof Promise;
var h;
(function (r) {
  ((r.errToObj = (e) => (typeof e == "string" ? { message: e } : e || {})),
    (r.toString = (e) => (typeof e == "string" ? e : e?.message)));
})(h || (h = {}));
var j = class {
    constructor(e, o, a, n) {
      ((this._cachedPath = []),
        (this.parent = e),
        (this.data = o),
        (this._path = a),
        (this._key = n));
    }
    get path() {
      return (
        this._cachedPath.length ||
          (Array.isArray(this._key)
            ? this._cachedPath.push(...this._path, ...this._key)
            : this._cachedPath.push(...this._path, this._key)),
        this._cachedPath
      );
    }
  },
  ta = (r, e) => {
    if (J(e)) return { success: !0, data: e.value };
    if (!r.common.issues.length)
      throw new Error("Validation failed but no issues detected.");
    return {
      success: !1,
      get error() {
        if (this._error) return this._error;
        let o = new A(r.common.issues);
        return ((this._error = o), this._error);
      },
    };
  };
function x(r) {
  if (!r) return {};
  let {
    errorMap: e,
    invalid_type_error: o,
    required_error: a,
    description: n,
  } = r;
  if (e && (o || a))
    throw new Error(
      `Can't use "invalid_type_error" or "required_error" in conjunction with custom error map.`,
    );
  return e
    ? { errorMap: e, description: n }
    : {
        errorMap: (c, l) => {
          let { message: u } = r;
          return c.code === "invalid_enum_value"
            ? { message: u ?? l.defaultError }
            : typeof l.data > "u"
              ? { message: u ?? a ?? l.defaultError }
              : c.code !== "invalid_type"
                ? { message: l.defaultError }
                : { message: u ?? o ?? l.defaultError };
        },
        description: n,
      };
}
var _ = class {
    get description() {
      return this._def.description;
    }
    _getType(e) {
      return M(e.data);
    }
    _getOrReturnCtx(e, o) {
      return (
        o || {
          common: e.parent.common,
          data: e.data,
          parsedType: M(e.data),
          schemaErrorMap: this._def.errorMap,
          path: e.path,
          parent: e.parent,
        }
      );
    }
    _processInputParams(e) {
      return {
        status: new v(),
        ctx: {
          common: e.parent.common,
          data: e.data,
          parsedType: M(e.data),
          schemaErrorMap: this._def.errorMap,
          path: e.path,
          parent: e.parent,
        },
      };
    }
    _parseSync(e) {
      let o = this._parse(e);
      if (Oe(o)) throw new Error("Synchronous parse encountered promise.");
      return o;
    }
    _parseAsync(e) {
      let o = this._parse(e);
      return Promise.resolve(o);
    }
    parse(e, o) {
      let a = this.safeParse(e, o);
      if (a.success) return a.data;
      throw a.error;
    }
    safeParse(e, o) {
      let a = {
          common: {
            issues: [],
            async: o?.async ?? !1,
            contextualErrorMap: o?.errorMap,
          },
          path: o?.path || [],
          schemaErrorMap: this._def.errorMap,
          parent: null,
          data: e,
          parsedType: M(e),
        },
        n = this._parseSync({ data: e, path: a.path, parent: a });
      return ta(a, n);
    }
    "~validate"(e) {
      let o = {
        common: { issues: [], async: !!this["~standard"].async },
        path: [],
        schemaErrorMap: this._def.errorMap,
        parent: null,
        data: e,
        parsedType: M(e),
      };
      if (!this["~standard"].async)
        try {
          let a = this._parseSync({ data: e, path: [], parent: o });
          return J(a) ? { value: a.value } : { issues: o.common.issues };
        } catch (a) {
          (a?.message?.toLowerCase()?.includes("encountered") &&
            (this["~standard"].async = !0),
            (o.common = { issues: [], async: !0 }));
        }
      return this._parseAsync({ data: e, path: [], parent: o }).then((a) =>
        J(a) ? { value: a.value } : { issues: o.common.issues },
      );
    }
    async parseAsync(e, o) {
      let a = await this.safeParseAsync(e, o);
      if (a.success) return a.data;
      throw a.error;
    }
    async safeParseAsync(e, o) {
      let a = {
          common: { issues: [], contextualErrorMap: o?.errorMap, async: !0 },
          path: o?.path || [],
          schemaErrorMap: this._def.errorMap,
          parent: null,
          data: e,
          parsedType: M(e),
        },
        n = this._parse({ data: e, path: a.path, parent: a }),
        s = await (Oe(n) ? n : Promise.resolve(n));
      return ta(a, s);
    }
    refine(e, o) {
      let a = (n) =>
        typeof o == "string" || typeof o > "u"
          ? { message: o }
          : typeof o == "function"
            ? o(n)
            : o;
      return this._refinement((n, s) => {
        let c = e(n),
          l = () => s.addIssue({ code: m.custom, ...a(n) });
        return typeof Promise < "u" && c instanceof Promise
          ? c.then((u) => (u ? !0 : (l(), !1)))
          : c
            ? !0
            : (l(), !1);
      });
    }
    refinement(e, o) {
      return this._refinement((a, n) =>
        e(a) ? !0 : (n.addIssue(typeof o == "function" ? o(a, n) : o), !1),
      );
    }
    _refinement(e) {
      return new O({
        schema: this,
        typeName: g.ZodEffects,
        effect: { type: "refinement", refinement: e },
      });
    }
    superRefine(e) {
      return this._refinement(e);
    }
    constructor(e) {
      ((this.spa = this.safeParseAsync),
        (this._def = e),
        (this.parse = this.parse.bind(this)),
        (this.safeParse = this.safeParse.bind(this)),
        (this.parseAsync = this.parseAsync.bind(this)),
        (this.safeParseAsync = this.safeParseAsync.bind(this)),
        (this.spa = this.spa.bind(this)),
        (this.refine = this.refine.bind(this)),
        (this.refinement = this.refinement.bind(this)),
        (this.superRefine = this.superRefine.bind(this)),
        (this.optional = this.optional.bind(this)),
        (this.nullable = this.nullable.bind(this)),
        (this.nullish = this.nullish.bind(this)),
        (this.array = this.array.bind(this)),
        (this.promise = this.promise.bind(this)),
        (this.or = this.or.bind(this)),
        (this.and = this.and.bind(this)),
        (this.transform = this.transform.bind(this)),
        (this.brand = this.brand.bind(this)),
        (this.default = this.default.bind(this)),
        (this.catch = this.catch.bind(this)),
        (this.describe = this.describe.bind(this)),
        (this.pipe = this.pipe.bind(this)),
        (this.readonly = this.readonly.bind(this)),
        (this.isNullable = this.isNullable.bind(this)),
        (this.isOptional = this.isOptional.bind(this)),
        (this["~standard"] = {
          version: 1,
          vendor: "zod",
          validate: (o) => this["~validate"](o),
        }));
    }
    optional() {
      return k.create(this, this._def);
    }
    nullable() {
      return F.create(this, this._def);
    }
    nullish() {
      return this.nullable().optional();
    }
    array() {
      return G.create(this);
    }
    promise() {
      return Y.create(this, this._def);
    }
    or(e) {
      return le.create([this, e], this._def);
    }
    and(e) {
      return de.create(this, e, this._def);
    }
    transform(e) {
      return new O({
        ...x(this._def),
        schema: this,
        typeName: g.ZodEffects,
        effect: { type: "transform", transform: e },
      });
    }
    default(e) {
      let o = typeof e == "function" ? e : () => e;
      return new ye({
        ...x(this._def),
        innerType: this,
        defaultValue: o,
        typeName: g.ZodDefault,
      });
    }
    brand() {
      return new St({ typeName: g.ZodBranded, type: this, ...x(this._def) });
    }
    catch(e) {
      let o = typeof e == "function" ? e : () => e;
      return new he({
        ...x(this._def),
        innerType: this,
        catchValue: o,
        typeName: g.ZodCatch,
      });
    }
    describe(e) {
      let o = this.constructor;
      return new o({ ...this._def, description: e });
    }
    pipe(e) {
      return vt.create(this, e);
    }
    readonly() {
      return be.create(this);
    }
    isOptional() {
      return this.safeParse(void 0).success;
    }
    isNullable() {
      return this.safeParse(null).success;
    }
  },
  Ic = /^c[^\s-]{8,}$/i,
  kc = /^[0-9a-z]+$/,
  jc = /^[0-9A-HJKMNP-TV-Z]{26}$/i,
  Oc =
    /^[0-9a-fA-F]{8}\b-[0-9a-fA-F]{4}\b-[0-9a-fA-F]{4}\b-[0-9a-fA-F]{4}\b-[0-9a-fA-F]{12}$/i,
  Bc = /^[a-z0-9_-]{21}$/i,
  zc = /^[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]*$/,
  Ec =
    /^[-+]?P(?!$)(?:(?:[-+]?\d+Y)|(?:[-+]?\d+[.,]\d+Y$))?(?:(?:[-+]?\d+M)|(?:[-+]?\d+[.,]\d+M$))?(?:(?:[-+]?\d+W)|(?:[-+]?\d+[.,]\d+W$))?(?:(?:[-+]?\d+D)|(?:[-+]?\d+[.,]\d+D$))?(?:T(?=[\d+-])(?:(?:[-+]?\d+H)|(?:[-+]?\d+[.,]\d+H$))?(?:(?:[-+]?\d+M)|(?:[-+]?\d+[.,]\d+M$))?(?:[-+]?\d+(?:[.,]\d+)?S)?)??$/,
  Dc =
    /^(?!\.)(?!.*\.\.)([A-Z0-9_'+\-\.]*)[A-Z0-9_+-]@([A-Z0-9][A-Z0-9\-]*\.)+[A-Z]{2,}$/i,
  Mc = "^(\\p{Extended_Pictographic}|\\p{Emoji_Component})+$",
  Oo,
  Lc =
    /^(?:(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])$/,
  Fc =
    /^(?:(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])\/(3[0-2]|[12]?[0-9])$/,
  Nc =
    /^(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))$/,
  $c =
    /^(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))\/(12[0-8]|1[01][0-9]|[1-9]?[0-9])$/,
  qc = /^([0-9a-zA-Z+/]{4})*(([0-9a-zA-Z+/]{2}==)|([0-9a-zA-Z+/]{3}=))?$/,
  Uc = /^([0-9a-zA-Z-_]{4})*(([0-9a-zA-Z-_]{2}(==)?)|([0-9a-zA-Z-_]{3}(=)?))?$/,
  oa =
    "((\\d\\d[2468][048]|\\d\\d[13579][26]|\\d\\d0[48]|[02468][048]00|[13579][26]00)-02-29|\\d{4}-((0[13578]|1[02])-(0[1-9]|[12]\\d|3[01])|(0[469]|11)-(0[1-9]|[12]\\d|30)|(02)-(0[1-9]|1\\d|2[0-8])))",
  Wc = new RegExp(`^${oa}$`);
function aa(r) {
  let e = "[0-5]\\d";
  r.precision
    ? (e = `${e}\\.\\d{${r.precision}}`)
    : r.precision == null && (e = `${e}(\\.\\d+)?`);
  let o = r.precision ? "+" : "?";
  return `([01]\\d|2[0-3]):[0-5]\\d(:${e})${o}`;
}
function Vc(r) {
  return new RegExp(`^${aa(r)}$`);
}
function na(r) {
  let e = `${oa}T${aa(r)}`,
    o = [];
  return (
    o.push(r.local ? "Z?" : "Z"),
    r.offset && o.push("([+-]\\d{2}:?\\d{2})"),
    (e = `${e}(${o.join("|")})`),
    new RegExp(`^${e}$`)
  );
}
function Zc(r, e) {
  return !!(
    ((e === "v4" || !e) && Lc.test(r)) ||
    ((e === "v6" || !e) && Nc.test(r))
  );
}
function Gc(r, e) {
  if (!zc.test(r)) return !1;
  try {
    let [o] = r.split(".");
    if (!o) return !1;
    let a = o
        .replace(/-/g, "+")
        .replace(/_/g, "/")
        .padEnd(o.length + ((4 - (o.length % 4)) % 4), "="),
      n = JSON.parse(atob(a));
    return !(
      typeof n != "object" ||
      n === null ||
      ("typ" in n && n?.typ !== "JWT") ||
      !n.alg ||
      (e && n.alg !== e)
    );
  } catch {
    return !1;
  }
}
function Hc(r, e) {
  return !!(
    ((e === "v4" || !e) && Fc.test(r)) ||
    ((e === "v6" || !e) && $c.test(r))
  );
}
var X = class r extends _ {
  _parse(e) {
    if (
      (this._def.coerce && (e.data = String(e.data)),
      this._getType(e) !== y.string)
    ) {
      let s = this._getOrReturnCtx(e);
      return (
        f(s, {
          code: m.invalid_type,
          expected: y.string,
          received: s.parsedType,
        }),
        b
      );
    }
    let a = new v(),
      n;
    for (let s of this._def.checks)
      if (s.kind === "min")
        e.data.length < s.value &&
          ((n = this._getOrReturnCtx(e, n)),
          f(n, {
            code: m.too_small,
            minimum: s.value,
            type: "string",
            inclusive: !0,
            exact: !1,
            message: s.message,
          }),
          a.dirty());
      else if (s.kind === "max")
        e.data.length > s.value &&
          ((n = this._getOrReturnCtx(e, n)),
          f(n, {
            code: m.too_big,
            maximum: s.value,
            type: "string",
            inclusive: !0,
            exact: !1,
            message: s.message,
          }),
          a.dirty());
      else if (s.kind === "length") {
        let c = e.data.length > s.value,
          l = e.data.length < s.value;
        (c || l) &&
          ((n = this._getOrReturnCtx(e, n)),
          c
            ? f(n, {
                code: m.too_big,
                maximum: s.value,
                type: "string",
                inclusive: !0,
                exact: !0,
                message: s.message,
              })
            : l &&
              f(n, {
                code: m.too_small,
                minimum: s.value,
                type: "string",
                inclusive: !0,
                exact: !0,
                message: s.message,
              }),
          a.dirty());
      } else if (s.kind === "email")
        Dc.test(e.data) ||
          ((n = this._getOrReturnCtx(e, n)),
          f(n, {
            validation: "email",
            code: m.invalid_string,
            message: s.message,
          }),
          a.dirty());
      else if (s.kind === "emoji")
        (Oo || (Oo = new RegExp(Mc, "u")),
          Oo.test(e.data) ||
            ((n = this._getOrReturnCtx(e, n)),
            f(n, {
              validation: "emoji",
              code: m.invalid_string,
              message: s.message,
            }),
            a.dirty()));
      else if (s.kind === "uuid")
        Oc.test(e.data) ||
          ((n = this._getOrReturnCtx(e, n)),
          f(n, {
            validation: "uuid",
            code: m.invalid_string,
            message: s.message,
          }),
          a.dirty());
      else if (s.kind === "nanoid")
        Bc.test(e.data) ||
          ((n = this._getOrReturnCtx(e, n)),
          f(n, {
            validation: "nanoid",
            code: m.invalid_string,
            message: s.message,
          }),
          a.dirty());
      else if (s.kind === "cuid")
        Ic.test(e.data) ||
          ((n = this._getOrReturnCtx(e, n)),
          f(n, {
            validation: "cuid",
            code: m.invalid_string,
            message: s.message,
          }),
          a.dirty());
      else if (s.kind === "cuid2")
        kc.test(e.data) ||
          ((n = this._getOrReturnCtx(e, n)),
          f(n, {
            validation: "cuid2",
            code: m.invalid_string,
            message: s.message,
          }),
          a.dirty());
      else if (s.kind === "ulid")
        jc.test(e.data) ||
          ((n = this._getOrReturnCtx(e, n)),
          f(n, {
            validation: "ulid",
            code: m.invalid_string,
            message: s.message,
          }),
          a.dirty());
      else if (s.kind === "url")
        try {
          new URL(e.data);
        } catch {
          ((n = this._getOrReturnCtx(e, n)),
            f(n, {
              validation: "url",
              code: m.invalid_string,
              message: s.message,
            }),
            a.dirty());
        }
      else
        s.kind === "regex"
          ? ((s.regex.lastIndex = 0),
            s.regex.test(e.data) ||
              ((n = this._getOrReturnCtx(e, n)),
              f(n, {
                validation: "regex",
                code: m.invalid_string,
                message: s.message,
              }),
              a.dirty()))
          : s.kind === "trim"
            ? (e.data = e.data.trim())
            : s.kind === "includes"
              ? e.data.includes(s.value, s.position) ||
                ((n = this._getOrReturnCtx(e, n)),
                f(n, {
                  code: m.invalid_string,
                  validation: { includes: s.value, position: s.position },
                  message: s.message,
                }),
                a.dirty())
              : s.kind === "toLowerCase"
                ? (e.data = e.data.toLowerCase())
                : s.kind === "toUpperCase"
                  ? (e.data = e.data.toUpperCase())
                  : s.kind === "startsWith"
                    ? e.data.startsWith(s.value) ||
                      ((n = this._getOrReturnCtx(e, n)),
                      f(n, {
                        code: m.invalid_string,
                        validation: { startsWith: s.value },
                        message: s.message,
                      }),
                      a.dirty())
                    : s.kind === "endsWith"
                      ? e.data.endsWith(s.value) ||
                        ((n = this._getOrReturnCtx(e, n)),
                        f(n, {
                          code: m.invalid_string,
                          validation: { endsWith: s.value },
                          message: s.message,
                        }),
                        a.dirty())
                      : s.kind === "datetime"
                        ? na(s).test(e.data) ||
                          ((n = this._getOrReturnCtx(e, n)),
                          f(n, {
                            code: m.invalid_string,
                            validation: "datetime",
                            message: s.message,
                          }),
                          a.dirty())
                        : s.kind === "date"
                          ? Wc.test(e.data) ||
                            ((n = this._getOrReturnCtx(e, n)),
                            f(n, {
                              code: m.invalid_string,
                              validation: "date",
                              message: s.message,
                            }),
                            a.dirty())
                          : s.kind === "time"
                            ? Vc(s).test(e.data) ||
                              ((n = this._getOrReturnCtx(e, n)),
                              f(n, {
                                code: m.invalid_string,
                                validation: "time",
                                message: s.message,
                              }),
                              a.dirty())
                            : s.kind === "duration"
                              ? Ec.test(e.data) ||
                                ((n = this._getOrReturnCtx(e, n)),
                                f(n, {
                                  validation: "duration",
                                  code: m.invalid_string,
                                  message: s.message,
                                }),
                                a.dirty())
                              : s.kind === "ip"
                                ? Zc(e.data, s.version) ||
                                  ((n = this._getOrReturnCtx(e, n)),
                                  f(n, {
                                    validation: "ip",
                                    code: m.invalid_string,
                                    message: s.message,
                                  }),
                                  a.dirty())
                                : s.kind === "jwt"
                                  ? Gc(e.data, s.alg) ||
                                    ((n = this._getOrReturnCtx(e, n)),
                                    f(n, {
                                      validation: "jwt",
                                      code: m.invalid_string,
                                      message: s.message,
                                    }),
                                    a.dirty())
                                  : s.kind === "cidr"
                                    ? Hc(e.data, s.version) ||
                                      ((n = this._getOrReturnCtx(e, n)),
                                      f(n, {
                                        validation: "cidr",
                                        code: m.invalid_string,
                                        message: s.message,
                                      }),
                                      a.dirty())
                                    : s.kind === "base64"
                                      ? qc.test(e.data) ||
                                        ((n = this._getOrReturnCtx(e, n)),
                                        f(n, {
                                          validation: "base64",
                                          code: m.invalid_string,
                                          message: s.message,
                                        }),
                                        a.dirty())
                                      : s.kind === "base64url"
                                        ? Uc.test(e.data) ||
                                          ((n = this._getOrReturnCtx(e, n)),
                                          f(n, {
                                            validation: "base64url",
                                            code: m.invalid_string,
                                            message: s.message,
                                          }),
                                          a.dirty())
                                        : C.assertNever(s);
    return { status: a.value, value: e.data };
  }
  _regex(e, o, a) {
    return this.refinement((n) => e.test(n), {
      validation: o,
      code: m.invalid_string,
      ...h.errToObj(a),
    });
  }
  _addCheck(e) {
    return new r({ ...this._def, checks: [...this._def.checks, e] });
  }
  email(e) {
    return this._addCheck({ kind: "email", ...h.errToObj(e) });
  }
  url(e) {
    return this._addCheck({ kind: "url", ...h.errToObj(e) });
  }
  emoji(e) {
    return this._addCheck({ kind: "emoji", ...h.errToObj(e) });
  }
  uuid(e) {
    return this._addCheck({ kind: "uuid", ...h.errToObj(e) });
  }
  nanoid(e) {
    return this._addCheck({ kind: "nanoid", ...h.errToObj(e) });
  }
  cuid(e) {
    return this._addCheck({ kind: "cuid", ...h.errToObj(e) });
  }
  cuid2(e) {
    return this._addCheck({ kind: "cuid2", ...h.errToObj(e) });
  }
  ulid(e) {
    return this._addCheck({ kind: "ulid", ...h.errToObj(e) });
  }
  base64(e) {
    return this._addCheck({ kind: "base64", ...h.errToObj(e) });
  }
  base64url(e) {
    return this._addCheck({ kind: "base64url", ...h.errToObj(e) });
  }
  jwt(e) {
    return this._addCheck({ kind: "jwt", ...h.errToObj(e) });
  }
  ip(e) {
    return this._addCheck({ kind: "ip", ...h.errToObj(e) });
  }
  cidr(e) {
    return this._addCheck({ kind: "cidr", ...h.errToObj(e) });
  }
  datetime(e) {
    return typeof e == "string"
      ? this._addCheck({
          kind: "datetime",
          precision: null,
          offset: !1,
          local: !1,
          message: e,
        })
      : this._addCheck({
          kind: "datetime",
          precision: typeof e?.precision > "u" ? null : e?.precision,
          offset: e?.offset ?? !1,
          local: e?.local ?? !1,
          ...h.errToObj(e?.message),
        });
  }
  date(e) {
    return this._addCheck({ kind: "date", message: e });
  }
  time(e) {
    return typeof e == "string"
      ? this._addCheck({ kind: "time", precision: null, message: e })
      : this._addCheck({
          kind: "time",
          precision: typeof e?.precision > "u" ? null : e?.precision,
          ...h.errToObj(e?.message),
        });
  }
  duration(e) {
    return this._addCheck({ kind: "duration", ...h.errToObj(e) });
  }
  regex(e, o) {
    return this._addCheck({ kind: "regex", regex: e, ...h.errToObj(o) });
  }
  includes(e, o) {
    return this._addCheck({
      kind: "includes",
      value: e,
      position: o?.position,
      ...h.errToObj(o?.message),
    });
  }
  startsWith(e, o) {
    return this._addCheck({ kind: "startsWith", value: e, ...h.errToObj(o) });
  }
  endsWith(e, o) {
    return this._addCheck({ kind: "endsWith", value: e, ...h.errToObj(o) });
  }
  min(e, o) {
    return this._addCheck({ kind: "min", value: e, ...h.errToObj(o) });
  }
  max(e, o) {
    return this._addCheck({ kind: "max", value: e, ...h.errToObj(o) });
  }
  length(e, o) {
    return this._addCheck({ kind: "length", value: e, ...h.errToObj(o) });
  }
  nonempty(e) {
    return this.min(1, h.errToObj(e));
  }
  trim() {
    return new r({
      ...this._def,
      checks: [...this._def.checks, { kind: "trim" }],
    });
  }
  toLowerCase() {
    return new r({
      ...this._def,
      checks: [...this._def.checks, { kind: "toLowerCase" }],
    });
  }
  toUpperCase() {
    return new r({
      ...this._def,
      checks: [...this._def.checks, { kind: "toUpperCase" }],
    });
  }
  get isDatetime() {
    return !!this._def.checks.find((e) => e.kind === "datetime");
  }
  get isDate() {
    return !!this._def.checks.find((e) => e.kind === "date");
  }
  get isTime() {
    return !!this._def.checks.find((e) => e.kind === "time");
  }
  get isDuration() {
    return !!this._def.checks.find((e) => e.kind === "duration");
  }
  get isEmail() {
    return !!this._def.checks.find((e) => e.kind === "email");
  }
  get isURL() {
    return !!this._def.checks.find((e) => e.kind === "url");
  }
  get isEmoji() {
    return !!this._def.checks.find((e) => e.kind === "emoji");
  }
  get isUUID() {
    return !!this._def.checks.find((e) => e.kind === "uuid");
  }
  get isNANOID() {
    return !!this._def.checks.find((e) => e.kind === "nanoid");
  }
  get isCUID() {
    return !!this._def.checks.find((e) => e.kind === "cuid");
  }
  get isCUID2() {
    return !!this._def.checks.find((e) => e.kind === "cuid2");
  }
  get isULID() {
    return !!this._def.checks.find((e) => e.kind === "ulid");
  }
  get isIP() {
    return !!this._def.checks.find((e) => e.kind === "ip");
  }
  get isCIDR() {
    return !!this._def.checks.find((e) => e.kind === "cidr");
  }
  get isBase64() {
    return !!this._def.checks.find((e) => e.kind === "base64");
  }
  get isBase64url() {
    return !!this._def.checks.find((e) => e.kind === "base64url");
  }
  get minLength() {
    let e = null;
    for (let o of this._def.checks)
      o.kind === "min" && (e === null || o.value > e) && (e = o.value);
    return e;
  }
  get maxLength() {
    let e = null;
    for (let o of this._def.checks)
      o.kind === "max" && (e === null || o.value < e) && (e = o.value);
    return e;
  }
};
X.create = (r) =>
  new X({
    checks: [],
    typeName: g.ZodString,
    coerce: r?.coerce ?? !1,
    ...x(r),
  });
function Jc(r, e) {
  let o = (r.toString().split(".")[1] || "").length,
    a = (e.toString().split(".")[1] || "").length,
    n = o > a ? o : a,
    s = Number.parseInt(r.toFixed(n).replace(".", "")),
    c = Number.parseInt(e.toFixed(n).replace(".", ""));
  return (s % c) / 10 ** n;
}
var oe = class r extends _ {
  constructor() {
    (super(...arguments),
      (this.min = this.gte),
      (this.max = this.lte),
      (this.step = this.multipleOf));
  }
  _parse(e) {
    if (
      (this._def.coerce && (e.data = Number(e.data)),
      this._getType(e) !== y.number)
    ) {
      let s = this._getOrReturnCtx(e);
      return (
        f(s, {
          code: m.invalid_type,
          expected: y.number,
          received: s.parsedType,
        }),
        b
      );
    }
    let a,
      n = new v();
    for (let s of this._def.checks)
      s.kind === "int"
        ? C.isInteger(e.data) ||
          ((a = this._getOrReturnCtx(e, a)),
          f(a, {
            code: m.invalid_type,
            expected: "integer",
            received: "float",
            message: s.message,
          }),
          n.dirty())
        : s.kind === "min"
          ? (s.inclusive ? e.data < s.value : e.data <= s.value) &&
            ((a = this._getOrReturnCtx(e, a)),
            f(a, {
              code: m.too_small,
              minimum: s.value,
              type: "number",
              inclusive: s.inclusive,
              exact: !1,
              message: s.message,
            }),
            n.dirty())
          : s.kind === "max"
            ? (s.inclusive ? e.data > s.value : e.data >= s.value) &&
              ((a = this._getOrReturnCtx(e, a)),
              f(a, {
                code: m.too_big,
                maximum: s.value,
                type: "number",
                inclusive: s.inclusive,
                exact: !1,
                message: s.message,
              }),
              n.dirty())
            : s.kind === "multipleOf"
              ? Jc(e.data, s.value) !== 0 &&
                ((a = this._getOrReturnCtx(e, a)),
                f(a, {
                  code: m.not_multiple_of,
                  multipleOf: s.value,
                  message: s.message,
                }),
                n.dirty())
              : s.kind === "finite"
                ? Number.isFinite(e.data) ||
                  ((a = this._getOrReturnCtx(e, a)),
                  f(a, { code: m.not_finite, message: s.message }),
                  n.dirty())
                : C.assertNever(s);
    return { status: n.value, value: e.data };
  }
  gte(e, o) {
    return this.setLimit("min", e, !0, h.toString(o));
  }
  gt(e, o) {
    return this.setLimit("min", e, !1, h.toString(o));
  }
  lte(e, o) {
    return this.setLimit("max", e, !0, h.toString(o));
  }
  lt(e, o) {
    return this.setLimit("max", e, !1, h.toString(o));
  }
  setLimit(e, o, a, n) {
    return new r({
      ...this._def,
      checks: [
        ...this._def.checks,
        { kind: e, value: o, inclusive: a, message: h.toString(n) },
      ],
    });
  }
  _addCheck(e) {
    return new r({ ...this._def, checks: [...this._def.checks, e] });
  }
  int(e) {
    return this._addCheck({ kind: "int", message: h.toString(e) });
  }
  positive(e) {
    return this._addCheck({
      kind: "min",
      value: 0,
      inclusive: !1,
      message: h.toString(e),
    });
  }
  negative(e) {
    return this._addCheck({
      kind: "max",
      value: 0,
      inclusive: !1,
      message: h.toString(e),
    });
  }
  nonpositive(e) {
    return this._addCheck({
      kind: "max",
      value: 0,
      inclusive: !0,
      message: h.toString(e),
    });
  }
  nonnegative(e) {
    return this._addCheck({
      kind: "min",
      value: 0,
      inclusive: !0,
      message: h.toString(e),
    });
  }
  multipleOf(e, o) {
    return this._addCheck({
      kind: "multipleOf",
      value: e,
      message: h.toString(o),
    });
  }
  finite(e) {
    return this._addCheck({ kind: "finite", message: h.toString(e) });
  }
  safe(e) {
    return this._addCheck({
      kind: "min",
      inclusive: !0,
      value: Number.MIN_SAFE_INTEGER,
      message: h.toString(e),
    })._addCheck({
      kind: "max",
      inclusive: !0,
      value: Number.MAX_SAFE_INTEGER,
      message: h.toString(e),
    });
  }
  get minValue() {
    let e = null;
    for (let o of this._def.checks)
      o.kind === "min" && (e === null || o.value > e) && (e = o.value);
    return e;
  }
  get maxValue() {
    let e = null;
    for (let o of this._def.checks)
      o.kind === "max" && (e === null || o.value < e) && (e = o.value);
    return e;
  }
  get isInt() {
    return !!this._def.checks.find(
      (e) =>
        e.kind === "int" || (e.kind === "multipleOf" && C.isInteger(e.value)),
    );
  }
  get isFinite() {
    let e = null,
      o = null;
    for (let a of this._def.checks) {
      if (a.kind === "finite" || a.kind === "int" || a.kind === "multipleOf")
        return !0;
      a.kind === "min"
        ? (o === null || a.value > o) && (o = a.value)
        : a.kind === "max" && (e === null || a.value < e) && (e = a.value);
    }
    return Number.isFinite(o) && Number.isFinite(e);
  }
};
oe.create = (r) =>
  new oe({
    checks: [],
    typeName: g.ZodNumber,
    coerce: r?.coerce || !1,
    ...x(r),
  });
var ae = class r extends _ {
  constructor() {
    (super(...arguments), (this.min = this.gte), (this.max = this.lte));
  }
  _parse(e) {
    if (this._def.coerce)
      try {
        e.data = BigInt(e.data);
      } catch {
        return this._getInvalidInput(e);
      }
    if (this._getType(e) !== y.bigint) return this._getInvalidInput(e);
    let a,
      n = new v();
    for (let s of this._def.checks)
      s.kind === "min"
        ? (s.inclusive ? e.data < s.value : e.data <= s.value) &&
          ((a = this._getOrReturnCtx(e, a)),
          f(a, {
            code: m.too_small,
            type: "bigint",
            minimum: s.value,
            inclusive: s.inclusive,
            message: s.message,
          }),
          n.dirty())
        : s.kind === "max"
          ? (s.inclusive ? e.data > s.value : e.data >= s.value) &&
            ((a = this._getOrReturnCtx(e, a)),
            f(a, {
              code: m.too_big,
              type: "bigint",
              maximum: s.value,
              inclusive: s.inclusive,
              message: s.message,
            }),
            n.dirty())
          : s.kind === "multipleOf"
            ? e.data % s.value !== BigInt(0) &&
              ((a = this._getOrReturnCtx(e, a)),
              f(a, {
                code: m.not_multiple_of,
                multipleOf: s.value,
                message: s.message,
              }),
              n.dirty())
            : C.assertNever(s);
    return { status: n.value, value: e.data };
  }
  _getInvalidInput(e) {
    let o = this._getOrReturnCtx(e);
    return (
      f(o, {
        code: m.invalid_type,
        expected: y.bigint,
        received: o.parsedType,
      }),
      b
    );
  }
  gte(e, o) {
    return this.setLimit("min", e, !0, h.toString(o));
  }
  gt(e, o) {
    return this.setLimit("min", e, !1, h.toString(o));
  }
  lte(e, o) {
    return this.setLimit("max", e, !0, h.toString(o));
  }
  lt(e, o) {
    return this.setLimit("max", e, !1, h.toString(o));
  }
  setLimit(e, o, a, n) {
    return new r({
      ...this._def,
      checks: [
        ...this._def.checks,
        { kind: e, value: o, inclusive: a, message: h.toString(n) },
      ],
    });
  }
  _addCheck(e) {
    return new r({ ...this._def, checks: [...this._def.checks, e] });
  }
  positive(e) {
    return this._addCheck({
      kind: "min",
      value: BigInt(0),
      inclusive: !1,
      message: h.toString(e),
    });
  }
  negative(e) {
    return this._addCheck({
      kind: "max",
      value: BigInt(0),
      inclusive: !1,
      message: h.toString(e),
    });
  }
  nonpositive(e) {
    return this._addCheck({
      kind: "max",
      value: BigInt(0),
      inclusive: !0,
      message: h.toString(e),
    });
  }
  nonnegative(e) {
    return this._addCheck({
      kind: "min",
      value: BigInt(0),
      inclusive: !0,
      message: h.toString(e),
    });
  }
  multipleOf(e, o) {
    return this._addCheck({
      kind: "multipleOf",
      value: e,
      message: h.toString(o),
    });
  }
  get minValue() {
    let e = null;
    for (let o of this._def.checks)
      o.kind === "min" && (e === null || o.value > e) && (e = o.value);
    return e;
  }
  get maxValue() {
    let e = null;
    for (let o of this._def.checks)
      o.kind === "max" && (e === null || o.value < e) && (e = o.value);
    return e;
  }
};
ae.create = (r) =>
  new ae({
    checks: [],
    typeName: g.ZodBigInt,
    coerce: r?.coerce ?? !1,
    ...x(r),
  });
var ne = class extends _ {
  _parse(e) {
    if (
      (this._def.coerce && (e.data = !!e.data), this._getType(e) !== y.boolean)
    ) {
      let a = this._getOrReturnCtx(e);
      return (
        f(a, {
          code: m.invalid_type,
          expected: y.boolean,
          received: a.parsedType,
        }),
        b
      );
    }
    return T(e.data);
  }
};
ne.create = (r) =>
  new ne({ typeName: g.ZodBoolean, coerce: r?.coerce || !1, ...x(r) });
var se = class r extends _ {
  _parse(e) {
    if (
      (this._def.coerce && (e.data = new Date(e.data)),
      this._getType(e) !== y.date)
    ) {
      let s = this._getOrReturnCtx(e);
      return (
        f(s, {
          code: m.invalid_type,
          expected: y.date,
          received: s.parsedType,
        }),
        b
      );
    }
    if (Number.isNaN(e.data.getTime())) {
      let s = this._getOrReturnCtx(e);
      return (f(s, { code: m.invalid_date }), b);
    }
    let a = new v(),
      n;
    for (let s of this._def.checks)
      s.kind === "min"
        ? e.data.getTime() < s.value &&
          ((n = this._getOrReturnCtx(e, n)),
          f(n, {
            code: m.too_small,
            message: s.message,
            inclusive: !0,
            exact: !1,
            minimum: s.value,
            type: "date",
          }),
          a.dirty())
        : s.kind === "max"
          ? e.data.getTime() > s.value &&
            ((n = this._getOrReturnCtx(e, n)),
            f(n, {
              code: m.too_big,
              message: s.message,
              inclusive: !0,
              exact: !1,
              maximum: s.value,
              type: "date",
            }),
            a.dirty())
          : C.assertNever(s);
    return { status: a.value, value: new Date(e.data.getTime()) };
  }
  _addCheck(e) {
    return new r({ ...this._def, checks: [...this._def.checks, e] });
  }
  min(e, o) {
    return this._addCheck({
      kind: "min",
      value: e.getTime(),
      message: h.toString(o),
    });
  }
  max(e, o) {
    return this._addCheck({
      kind: "max",
      value: e.getTime(),
      message: h.toString(o),
    });
  }
  get minDate() {
    let e = null;
    for (let o of this._def.checks)
      o.kind === "min" && (e === null || o.value > e) && (e = o.value);
    return e != null ? new Date(e) : null;
  }
  get maxDate() {
    let e = null;
    for (let o of this._def.checks)
      o.kind === "max" && (e === null || o.value < e) && (e = o.value);
    return e != null ? new Date(e) : null;
  }
};
se.create = (r) =>
  new se({ checks: [], coerce: r?.coerce || !1, typeName: g.ZodDate, ...x(r) });
var ze = class extends _ {
  _parse(e) {
    if (this._getType(e) !== y.symbol) {
      let a = this._getOrReturnCtx(e);
      return (
        f(a, {
          code: m.invalid_type,
          expected: y.symbol,
          received: a.parsedType,
        }),
        b
      );
    }
    return T(e.data);
  }
};
ze.create = (r) => new ze({ typeName: g.ZodSymbol, ...x(r) });
var ie = class extends _ {
  _parse(e) {
    if (this._getType(e) !== y.undefined) {
      let a = this._getOrReturnCtx(e);
      return (
        f(a, {
          code: m.invalid_type,
          expected: y.undefined,
          received: a.parsedType,
        }),
        b
      );
    }
    return T(e.data);
  }
};
ie.create = (r) => new ie({ typeName: g.ZodUndefined, ...x(r) });
var ce = class extends _ {
  _parse(e) {
    if (this._getType(e) !== y.null) {
      let a = this._getOrReturnCtx(e);
      return (
        f(a, {
          code: m.invalid_type,
          expected: y.null,
          received: a.parsedType,
        }),
        b
      );
    }
    return T(e.data);
  }
};
ce.create = (r) => new ce({ typeName: g.ZodNull, ...x(r) });
var K = class extends _ {
  constructor() {
    (super(...arguments), (this._any = !0));
  }
  _parse(e) {
    return T(e.data);
  }
};
K.create = (r) => new K({ typeName: g.ZodAny, ...x(r) });
var Z = class extends _ {
  constructor() {
    (super(...arguments), (this._unknown = !0));
  }
  _parse(e) {
    return T(e.data);
  }
};
Z.create = (r) => new Z({ typeName: g.ZodUnknown, ...x(r) });
var E = class extends _ {
  _parse(e) {
    let o = this._getOrReturnCtx(e);
    return (
      f(o, { code: m.invalid_type, expected: y.never, received: o.parsedType }),
      b
    );
  }
};
E.create = (r) => new E({ typeName: g.ZodNever, ...x(r) });
var Ee = class extends _ {
  _parse(e) {
    if (this._getType(e) !== y.undefined) {
      let a = this._getOrReturnCtx(e);
      return (
        f(a, {
          code: m.invalid_type,
          expected: y.void,
          received: a.parsedType,
        }),
        b
      );
    }
    return T(e.data);
  }
};
Ee.create = (r) => new Ee({ typeName: g.ZodVoid, ...x(r) });
var G = class r extends _ {
  _parse(e) {
    let { ctx: o, status: a } = this._processInputParams(e),
      n = this._def;
    if (o.parsedType !== y.array)
      return (
        f(o, {
          code: m.invalid_type,
          expected: y.array,
          received: o.parsedType,
        }),
        b
      );
    if (n.exactLength !== null) {
      let c = o.data.length > n.exactLength.value,
        l = o.data.length < n.exactLength.value;
      (c || l) &&
        (f(o, {
          code: c ? m.too_big : m.too_small,
          minimum: l ? n.exactLength.value : void 0,
          maximum: c ? n.exactLength.value : void 0,
          type: "array",
          inclusive: !0,
          exact: !0,
          message: n.exactLength.message,
        }),
        a.dirty());
    }
    if (
      (n.minLength !== null &&
        o.data.length < n.minLength.value &&
        (f(o, {
          code: m.too_small,
          minimum: n.minLength.value,
          type: "array",
          inclusive: !0,
          exact: !1,
          message: n.minLength.message,
        }),
        a.dirty()),
      n.maxLength !== null &&
        o.data.length > n.maxLength.value &&
        (f(o, {
          code: m.too_big,
          maximum: n.maxLength.value,
          type: "array",
          inclusive: !0,
          exact: !1,
          message: n.maxLength.message,
        }),
        a.dirty()),
      o.common.async)
    )
      return Promise.all(
        [...o.data].map((c, l) => n.type._parseAsync(new j(o, c, o.path, l))),
      ).then((c) => v.mergeArray(a, c));
    let s = [...o.data].map((c, l) =>
      n.type._parseSync(new j(o, c, o.path, l)),
    );
    return v.mergeArray(a, s);
  }
  get element() {
    return this._def.type;
  }
  min(e, o) {
    return new r({
      ...this._def,
      minLength: { value: e, message: h.toString(o) },
    });
  }
  max(e, o) {
    return new r({
      ...this._def,
      maxLength: { value: e, message: h.toString(o) },
    });
  }
  length(e, o) {
    return new r({
      ...this._def,
      exactLength: { value: e, message: h.toString(o) },
    });
  }
  nonempty(e) {
    return this.min(1, e);
  }
};
G.create = (r, e) =>
  new G({
    type: r,
    minLength: null,
    maxLength: null,
    exactLength: null,
    typeName: g.ZodArray,
    ...x(e),
  });
function Be(r) {
  if (r instanceof I) {
    let e = {};
    for (let o in r.shape) {
      let a = r.shape[o];
      e[o] = k.create(Be(a));
    }
    return new I({ ...r._def, shape: () => e });
  } else
    return r instanceof G
      ? new G({ ...r._def, type: Be(r.element) })
      : r instanceof k
        ? k.create(Be(r.unwrap()))
        : r instanceof F
          ? F.create(Be(r.unwrap()))
          : r instanceof L
            ? L.create(r.items.map((e) => Be(e)))
            : r;
}
var I = class r extends _ {
  constructor() {
    (super(...arguments),
      (this._cached = null),
      (this.nonstrict = this.passthrough),
      (this.augment = this.extend));
  }
  _getCached() {
    if (this._cached !== null) return this._cached;
    let e = this._def.shape(),
      o = C.objectKeys(e);
    return ((this._cached = { shape: e, keys: o }), this._cached);
  }
  _parse(e) {
    if (this._getType(e) !== y.object) {
      let p = this._getOrReturnCtx(e);
      return (
        f(p, {
          code: m.invalid_type,
          expected: y.object,
          received: p.parsedType,
        }),
        b
      );
    }
    let { status: a, ctx: n } = this._processInputParams(e),
      { shape: s, keys: c } = this._getCached(),
      l = [];
    if (!(this._def.catchall instanceof E && this._def.unknownKeys === "strip"))
      for (let p in n.data) c.includes(p) || l.push(p);
    let u = [];
    for (let p of c) {
      let w = s[p],
        S = n.data[p];
      u.push({
        key: { status: "valid", value: p },
        value: w._parse(new j(n, S, n.path, p)),
        alwaysSet: p in n.data,
      });
    }
    if (this._def.catchall instanceof E) {
      let p = this._def.unknownKeys;
      if (p === "passthrough")
        for (let w of l)
          u.push({
            key: { status: "valid", value: w },
            value: { status: "valid", value: n.data[w] },
          });
      else if (p === "strict")
        l.length > 0 &&
          (f(n, { code: m.unrecognized_keys, keys: l }), a.dirty());
      else if (p !== "strip")
        throw new Error("Internal ZodObject error: invalid unknownKeys value.");
    } else {
      let p = this._def.catchall;
      for (let w of l) {
        let S = n.data[w];
        u.push({
          key: { status: "valid", value: w },
          value: p._parse(new j(n, S, n.path, w)),
          alwaysSet: w in n.data,
        });
      }
    }
    return n.common.async
      ? Promise.resolve()
          .then(async () => {
            let p = [];
            for (let w of u) {
              let S = await w.key,
                U = await w.value;
              p.push({ key: S, value: U, alwaysSet: w.alwaysSet });
            }
            return p;
          })
          .then((p) => v.mergeObjectSync(a, p))
      : v.mergeObjectSync(a, u);
  }
  get shape() {
    return this._def.shape();
  }
  strict(e) {
    return (
      h.errToObj,
      new r({
        ...this._def,
        unknownKeys: "strict",
        ...(e !== void 0
          ? {
              errorMap: (o, a) => {
                let n = this._def.errorMap?.(o, a).message ?? a.defaultError;
                return o.code === "unrecognized_keys"
                  ? { message: h.errToObj(e).message ?? n }
                  : { message: n };
              },
            }
          : {}),
      })
    );
  }
  strip() {
    return new r({ ...this._def, unknownKeys: "strip" });
  }
  passthrough() {
    return new r({ ...this._def, unknownKeys: "passthrough" });
  }
  extend(e) {
    return new r({
      ...this._def,
      shape: () => ({ ...this._def.shape(), ...e }),
    });
  }
  merge(e) {
    return new r({
      unknownKeys: e._def.unknownKeys,
      catchall: e._def.catchall,
      shape: () => ({ ...this._def.shape(), ...e._def.shape() }),
      typeName: g.ZodObject,
    });
  }
  setKey(e, o) {
    return this.augment({ [e]: o });
  }
  catchall(e) {
    return new r({ ...this._def, catchall: e });
  }
  pick(e) {
    let o = {};
    for (let a of C.objectKeys(e))
      e[a] && this.shape[a] && (o[a] = this.shape[a]);
    return new r({ ...this._def, shape: () => o });
  }
  omit(e) {
    let o = {};
    for (let a of C.objectKeys(this.shape)) e[a] || (o[a] = this.shape[a]);
    return new r({ ...this._def, shape: () => o });
  }
  deepPartial() {
    return Be(this);
  }
  partial(e) {
    let o = {};
    for (let a of C.objectKeys(this.shape)) {
      let n = this.shape[a];
      e && !e[a] ? (o[a] = n) : (o[a] = n.optional());
    }
    return new r({ ...this._def, shape: () => o });
  }
  required(e) {
    let o = {};
    for (let a of C.objectKeys(this.shape))
      if (e && !e[a]) o[a] = this.shape[a];
      else {
        let s = this.shape[a];
        for (; s instanceof k; ) s = s._def.innerType;
        o[a] = s;
      }
    return new r({ ...this._def, shape: () => o });
  }
  keyof() {
    return sa(C.objectKeys(this.shape));
  }
};
I.create = (r, e) =>
  new I({
    shape: () => r,
    unknownKeys: "strip",
    catchall: E.create(),
    typeName: g.ZodObject,
    ...x(e),
  });
I.strictCreate = (r, e) =>
  new I({
    shape: () => r,
    unknownKeys: "strict",
    catchall: E.create(),
    typeName: g.ZodObject,
    ...x(e),
  });
I.lazycreate = (r, e) =>
  new I({
    shape: r,
    unknownKeys: "strip",
    catchall: E.create(),
    typeName: g.ZodObject,
    ...x(e),
  });
var le = class extends _ {
  _parse(e) {
    let { ctx: o } = this._processInputParams(e),
      a = this._def.options;
    function n(s) {
      for (let l of s) if (l.result.status === "valid") return l.result;
      for (let l of s)
        if (l.result.status === "dirty")
          return (o.common.issues.push(...l.ctx.common.issues), l.result);
      let c = s.map((l) => new A(l.ctx.common.issues));
      return (f(o, { code: m.invalid_union, unionErrors: c }), b);
    }
    if (o.common.async)
      return Promise.all(
        a.map(async (s) => {
          let c = { ...o, common: { ...o.common, issues: [] }, parent: null };
          return {
            result: await s._parseAsync({
              data: o.data,
              path: o.path,
              parent: c,
            }),
            ctx: c,
          };
        }),
      ).then(n);
    {
      let s,
        c = [];
      for (let u of a) {
        let p = { ...o, common: { ...o.common, issues: [] }, parent: null },
          w = u._parseSync({ data: o.data, path: o.path, parent: p });
        if (w.status === "valid") return w;
        (w.status === "dirty" && !s && (s = { result: w, ctx: p }),
          p.common.issues.length && c.push(p.common.issues));
      }
      if (s) return (o.common.issues.push(...s.ctx.common.issues), s.result);
      let l = c.map((u) => new A(u));
      return (f(o, { code: m.invalid_union, unionErrors: l }), b);
    }
  }
  get options() {
    return this._def.options;
  }
};
le.create = (r, e) => new le({ options: r, typeName: g.ZodUnion, ...x(e) });
var V = (r) =>
    r instanceof pe
      ? V(r.schema)
      : r instanceof O
        ? V(r.innerType())
        : r instanceof me
          ? [r.value]
          : r instanceof ue
            ? r.options
            : r instanceof fe
              ? C.objectValues(r.enum)
              : r instanceof ye
                ? V(r._def.innerType)
                : r instanceof ie
                  ? [void 0]
                  : r instanceof ce
                    ? [null]
                    : r instanceof k
                      ? [void 0, ...V(r.unwrap())]
                      : r instanceof F
                        ? [null, ...V(r.unwrap())]
                        : r instanceof St || r instanceof be
                          ? V(r.unwrap())
                          : r instanceof he
                            ? V(r._def.innerType)
                            : [],
  Yr = class r extends _ {
    _parse(e) {
      let { ctx: o } = this._processInputParams(e);
      if (o.parsedType !== y.object)
        return (
          f(o, {
            code: m.invalid_type,
            expected: y.object,
            received: o.parsedType,
          }),
          b
        );
      let a = this.discriminator,
        n = o.data[a],
        s = this.optionsMap.get(n);
      return s
        ? o.common.async
          ? s._parseAsync({ data: o.data, path: o.path, parent: o })
          : s._parseSync({ data: o.data, path: o.path, parent: o })
        : (f(o, {
            code: m.invalid_union_discriminator,
            options: Array.from(this.optionsMap.keys()),
            path: [a],
          }),
          b);
    }
    get discriminator() {
      return this._def.discriminator;
    }
    get options() {
      return this._def.options;
    }
    get optionsMap() {
      return this._def.optionsMap;
    }
    static create(e, o, a) {
      let n = new Map();
      for (let s of o) {
        let c = V(s.shape[e]);
        if (!c.length)
          throw new Error(
            `A discriminator value for key \`${e}\` could not be extracted from all schema options`,
          );
        for (let l of c) {
          if (n.has(l))
            throw new Error(
              `Discriminator property ${String(e)} has duplicate value ${String(l)}`,
            );
          n.set(l, s);
        }
      }
      return new r({
        typeName: g.ZodDiscriminatedUnion,
        discriminator: e,
        options: o,
        optionsMap: n,
        ...x(a),
      });
    }
  };
function Bo(r, e) {
  let o = M(r),
    a = M(e);
  if (r === e) return { valid: !0, data: r };
  if (o === y.object && a === y.object) {
    let n = C.objectKeys(e),
      s = C.objectKeys(r).filter((l) => n.indexOf(l) !== -1),
      c = { ...r, ...e };
    for (let l of s) {
      let u = Bo(r[l], e[l]);
      if (!u.valid) return { valid: !1 };
      c[l] = u.data;
    }
    return { valid: !0, data: c };
  } else if (o === y.array && a === y.array) {
    if (r.length !== e.length) return { valid: !1 };
    let n = [];
    for (let s = 0; s < r.length; s++) {
      let c = r[s],
        l = e[s],
        u = Bo(c, l);
      if (!u.valid) return { valid: !1 };
      n.push(u.data);
    }
    return { valid: !0, data: n };
  } else
    return o === y.date && a === y.date && +r == +e
      ? { valid: !0, data: r }
      : { valid: !1 };
}
var de = class extends _ {
  _parse(e) {
    let { status: o, ctx: a } = this._processInputParams(e),
      n = (s, c) => {
        if (Xr(s) || Xr(c)) return b;
        let l = Bo(s.value, c.value);
        return l.valid
          ? ((Kr(s) || Kr(c)) && o.dirty(), { status: o.value, value: l.data })
          : (f(a, { code: m.invalid_intersection_types }), b);
      };
    return a.common.async
      ? Promise.all([
          this._def.left._parseAsync({ data: a.data, path: a.path, parent: a }),
          this._def.right._parseAsync({
            data: a.data,
            path: a.path,
            parent: a,
          }),
        ]).then(([s, c]) => n(s, c))
      : n(
          this._def.left._parseSync({ data: a.data, path: a.path, parent: a }),
          this._def.right._parseSync({ data: a.data, path: a.path, parent: a }),
        );
  }
};
de.create = (r, e, o) =>
  new de({ left: r, right: e, typeName: g.ZodIntersection, ...x(o) });
var L = class r extends _ {
  _parse(e) {
    let { status: o, ctx: a } = this._processInputParams(e);
    if (a.parsedType !== y.array)
      return (
        f(a, {
          code: m.invalid_type,
          expected: y.array,
          received: a.parsedType,
        }),
        b
      );
    if (a.data.length < this._def.items.length)
      return (
        f(a, {
          code: m.too_small,
          minimum: this._def.items.length,
          inclusive: !0,
          exact: !1,
          type: "array",
        }),
        b
      );
    !this._def.rest &&
      a.data.length > this._def.items.length &&
      (f(a, {
        code: m.too_big,
        maximum: this._def.items.length,
        inclusive: !0,
        exact: !1,
        type: "array",
      }),
      o.dirty());
    let s = [...a.data]
      .map((c, l) => {
        let u = this._def.items[l] || this._def.rest;
        return u ? u._parse(new j(a, c, a.path, l)) : null;
      })
      .filter((c) => !!c);
    return a.common.async
      ? Promise.all(s).then((c) => v.mergeArray(o, c))
      : v.mergeArray(o, s);
  }
  get items() {
    return this._def.items;
  }
  rest(e) {
    return new r({ ...this._def, rest: e });
  }
};
L.create = (r, e) => {
  if (!Array.isArray(r))
    throw new Error("You must pass an array of schemas to z.tuple([ ... ])");
  return new L({ items: r, typeName: g.ZodTuple, rest: null, ...x(e) });
};
var Qr = class r extends _ {
    get keySchema() {
      return this._def.keyType;
    }
    get valueSchema() {
      return this._def.valueType;
    }
    _parse(e) {
      let { status: o, ctx: a } = this._processInputParams(e);
      if (a.parsedType !== y.object)
        return (
          f(a, {
            code: m.invalid_type,
            expected: y.object,
            received: a.parsedType,
          }),
          b
        );
      let n = [],
        s = this._def.keyType,
        c = this._def.valueType;
      for (let l in a.data)
        n.push({
          key: s._parse(new j(a, l, a.path, l)),
          value: c._parse(new j(a, a.data[l], a.path, l)),
          alwaysSet: l in a.data,
        });
      return a.common.async
        ? v.mergeObjectAsync(o, n)
        : v.mergeObjectSync(o, n);
    }
    get element() {
      return this._def.valueType;
    }
    static create(e, o, a) {
      return o instanceof _
        ? new r({ keyType: e, valueType: o, typeName: g.ZodRecord, ...x(a) })
        : new r({
            keyType: X.create(),
            valueType: e,
            typeName: g.ZodRecord,
            ...x(o),
          });
    }
  },
  De = class extends _ {
    get keySchema() {
      return this._def.keyType;
    }
    get valueSchema() {
      return this._def.valueType;
    }
    _parse(e) {
      let { status: o, ctx: a } = this._processInputParams(e);
      if (a.parsedType !== y.map)
        return (
          f(a, {
            code: m.invalid_type,
            expected: y.map,
            received: a.parsedType,
          }),
          b
        );
      let n = this._def.keyType,
        s = this._def.valueType,
        c = [...a.data.entries()].map(([l, u], p) => ({
          key: n._parse(new j(a, l, a.path, [p, "key"])),
          value: s._parse(new j(a, u, a.path, [p, "value"])),
        }));
      if (a.common.async) {
        let l = new Map();
        return Promise.resolve().then(async () => {
          for (let u of c) {
            let p = await u.key,
              w = await u.value;
            if (p.status === "aborted" || w.status === "aborted") return b;
            ((p.status === "dirty" || w.status === "dirty") && o.dirty(),
              l.set(p.value, w.value));
          }
          return { status: o.value, value: l };
        });
      } else {
        let l = new Map();
        for (let u of c) {
          let p = u.key,
            w = u.value;
          if (p.status === "aborted" || w.status === "aborted") return b;
          ((p.status === "dirty" || w.status === "dirty") && o.dirty(),
            l.set(p.value, w.value));
        }
        return { status: o.value, value: l };
      }
    }
  };
De.create = (r, e, o) =>
  new De({ valueType: e, keyType: r, typeName: g.ZodMap, ...x(o) });
var Me = class r extends _ {
  _parse(e) {
    let { status: o, ctx: a } = this._processInputParams(e);
    if (a.parsedType !== y.set)
      return (
        f(a, { code: m.invalid_type, expected: y.set, received: a.parsedType }),
        b
      );
    let n = this._def;
    (n.minSize !== null &&
      a.data.size < n.minSize.value &&
      (f(a, {
        code: m.too_small,
        minimum: n.minSize.value,
        type: "set",
        inclusive: !0,
        exact: !1,
        message: n.minSize.message,
      }),
      o.dirty()),
      n.maxSize !== null &&
        a.data.size > n.maxSize.value &&
        (f(a, {
          code: m.too_big,
          maximum: n.maxSize.value,
          type: "set",
          inclusive: !0,
          exact: !1,
          message: n.maxSize.message,
        }),
        o.dirty()));
    let s = this._def.valueType;
    function c(u) {
      let p = new Set();
      for (let w of u) {
        if (w.status === "aborted") return b;
        (w.status === "dirty" && o.dirty(), p.add(w.value));
      }
      return { status: o.value, value: p };
    }
    let l = [...a.data.values()].map((u, p) =>
      s._parse(new j(a, u, a.path, p)),
    );
    return a.common.async ? Promise.all(l).then((u) => c(u)) : c(l);
  }
  min(e, o) {
    return new r({
      ...this._def,
      minSize: { value: e, message: h.toString(o) },
    });
  }
  max(e, o) {
    return new r({
      ...this._def,
      maxSize: { value: e, message: h.toString(o) },
    });
  }
  size(e, o) {
    return this.min(e, o).max(e, o);
  }
  nonempty(e) {
    return this.min(1, e);
  }
};
Me.create = (r, e) =>
  new Me({
    valueType: r,
    minSize: null,
    maxSize: null,
    typeName: g.ZodSet,
    ...x(e),
  });
var eo = class r extends _ {
    constructor() {
      (super(...arguments), (this.validate = this.implement));
    }
    _parse(e) {
      let { ctx: o } = this._processInputParams(e);
      if (o.parsedType !== y.function)
        return (
          f(o, {
            code: m.invalid_type,
            expected: y.function,
            received: o.parsedType,
          }),
          b
        );
      function a(l, u) {
        return Ct({
          data: l,
          path: o.path,
          errorMaps: [
            o.common.contextualErrorMap,
            o.schemaErrorMap,
            je(),
            W,
          ].filter((p) => !!p),
          issueData: { code: m.invalid_arguments, argumentsError: u },
        });
      }
      function n(l, u) {
        return Ct({
          data: l,
          path: o.path,
          errorMaps: [
            o.common.contextualErrorMap,
            o.schemaErrorMap,
            je(),
            W,
          ].filter((p) => !!p),
          issueData: { code: m.invalid_return_type, returnTypeError: u },
        });
      }
      let s = { errorMap: o.common.contextualErrorMap },
        c = o.data;
      if (this._def.returns instanceof Y) {
        let l = this;
        return T(async function (...u) {
          let p = new A([]),
            w = await l._def.args.parseAsync(u, s).catch((ke) => {
              throw (p.addIssue(a(u, ke)), p);
            }),
            S = await Reflect.apply(c, this, w);
          return await l._def.returns._def.type.parseAsync(S, s).catch((ke) => {
            throw (p.addIssue(n(S, ke)), p);
          });
        });
      } else {
        let l = this;
        return T(function (...u) {
          let p = l._def.args.safeParse(u, s);
          if (!p.success) throw new A([a(u, p.error)]);
          let w = Reflect.apply(c, this, p.data),
            S = l._def.returns.safeParse(w, s);
          if (!S.success) throw new A([n(w, S.error)]);
          return S.data;
        });
      }
    }
    parameters() {
      return this._def.args;
    }
    returnType() {
      return this._def.returns;
    }
    args(...e) {
      return new r({ ...this._def, args: L.create(e).rest(Z.create()) });
    }
    returns(e) {
      return new r({ ...this._def, returns: e });
    }
    implement(e) {
      return this.parse(e);
    }
    strictImplement(e) {
      return this.parse(e);
    }
    static create(e, o, a) {
      return new r({
        args: e || L.create([]).rest(Z.create()),
        returns: o || Z.create(),
        typeName: g.ZodFunction,
        ...x(a),
      });
    }
  },
  pe = class extends _ {
    get schema() {
      return this._def.getter();
    }
    _parse(e) {
      let { ctx: o } = this._processInputParams(e);
      return this._def
        .getter()
        ._parse({ data: o.data, path: o.path, parent: o });
    }
  };
pe.create = (r, e) => new pe({ getter: r, typeName: g.ZodLazy, ...x(e) });
var me = class extends _ {
  _parse(e) {
    if (e.data !== this._def.value) {
      let o = this._getOrReturnCtx(e);
      return (
        f(o, {
          received: o.data,
          code: m.invalid_literal,
          expected: this._def.value,
        }),
        b
      );
    }
    return { status: "valid", value: e.data };
  }
  get value() {
    return this._def.value;
  }
};
me.create = (r, e) => new me({ value: r, typeName: g.ZodLiteral, ...x(e) });
function sa(r, e) {
  return new ue({ values: r, typeName: g.ZodEnum, ...x(e) });
}
var ue = class r extends _ {
  _parse(e) {
    if (typeof e.data != "string") {
      let o = this._getOrReturnCtx(e),
        a = this._def.values;
      return (
        f(o, {
          expected: C.joinValues(a),
          received: o.parsedType,
          code: m.invalid_type,
        }),
        b
      );
    }
    if (
      (this._cache || (this._cache = new Set(this._def.values)),
      !this._cache.has(e.data))
    ) {
      let o = this._getOrReturnCtx(e),
        a = this._def.values;
      return (
        f(o, { received: o.data, code: m.invalid_enum_value, options: a }),
        b
      );
    }
    return T(e.data);
  }
  get options() {
    return this._def.values;
  }
  get enum() {
    let e = {};
    for (let o of this._def.values) e[o] = o;
    return e;
  }
  get Values() {
    let e = {};
    for (let o of this._def.values) e[o] = o;
    return e;
  }
  get Enum() {
    let e = {};
    for (let o of this._def.values) e[o] = o;
    return e;
  }
  extract(e, o = this._def) {
    return r.create(e, { ...this._def, ...o });
  }
  exclude(e, o = this._def) {
    return r.create(
      this.options.filter((a) => !e.includes(a)),
      { ...this._def, ...o },
    );
  }
};
ue.create = sa;
var fe = class extends _ {
  _parse(e) {
    let o = C.getValidEnumValues(this._def.values),
      a = this._getOrReturnCtx(e);
    if (a.parsedType !== y.string && a.parsedType !== y.number) {
      let n = C.objectValues(o);
      return (
        f(a, {
          expected: C.joinValues(n),
          received: a.parsedType,
          code: m.invalid_type,
        }),
        b
      );
    }
    if (
      (this._cache ||
        (this._cache = new Set(C.getValidEnumValues(this._def.values))),
      !this._cache.has(e.data))
    ) {
      let n = C.objectValues(o);
      return (
        f(a, { received: a.data, code: m.invalid_enum_value, options: n }),
        b
      );
    }
    return T(e.data);
  }
  get enum() {
    return this._def.values;
  }
};
fe.create = (r, e) => new fe({ values: r, typeName: g.ZodNativeEnum, ...x(e) });
var Y = class extends _ {
  unwrap() {
    return this._def.type;
  }
  _parse(e) {
    let { ctx: o } = this._processInputParams(e);
    if (o.parsedType !== y.promise && o.common.async === !1)
      return (
        f(o, {
          code: m.invalid_type,
          expected: y.promise,
          received: o.parsedType,
        }),
        b
      );
    let a = o.parsedType === y.promise ? o.data : Promise.resolve(o.data);
    return T(
      a.then((n) =>
        this._def.type.parseAsync(n, {
          path: o.path,
          errorMap: o.common.contextualErrorMap,
        }),
      ),
    );
  }
};
Y.create = (r, e) => new Y({ type: r, typeName: g.ZodPromise, ...x(e) });
var O = class extends _ {
  innerType() {
    return this._def.schema;
  }
  sourceType() {
    return this._def.schema._def.typeName === g.ZodEffects
      ? this._def.schema.sourceType()
      : this._def.schema;
  }
  _parse(e) {
    let { status: o, ctx: a } = this._processInputParams(e),
      n = this._def.effect || null,
      s = {
        addIssue: (c) => {
          (f(a, c), c.fatal ? o.abort() : o.dirty());
        },
        get path() {
          return a.path;
        },
      };
    if (((s.addIssue = s.addIssue.bind(s)), n.type === "preprocess")) {
      let c = n.transform(a.data, s);
      if (a.common.async)
        return Promise.resolve(c).then(async (l) => {
          if (o.value === "aborted") return b;
          let u = await this._def.schema._parseAsync({
            data: l,
            path: a.path,
            parent: a,
          });
          return u.status === "aborted"
            ? b
            : u.status === "dirty"
              ? re(u.value)
              : o.value === "dirty"
                ? re(u.value)
                : u;
        });
      {
        if (o.value === "aborted") return b;
        let l = this._def.schema._parseSync({
          data: c,
          path: a.path,
          parent: a,
        });
        return l.status === "aborted"
          ? b
          : l.status === "dirty"
            ? re(l.value)
            : o.value === "dirty"
              ? re(l.value)
              : l;
      }
    }
    if (n.type === "refinement") {
      let c = (l) => {
        let u = n.refinement(l, s);
        if (a.common.async) return Promise.resolve(u);
        if (u instanceof Promise)
          throw new Error(
            "Async refinement encountered during synchronous parse operation. Use .parseAsync instead.",
          );
        return l;
      };
      if (a.common.async === !1) {
        let l = this._def.schema._parseSync({
          data: a.data,
          path: a.path,
          parent: a,
        });
        return l.status === "aborted"
          ? b
          : (l.status === "dirty" && o.dirty(),
            c(l.value),
            { status: o.value, value: l.value });
      } else
        return this._def.schema
          ._parseAsync({ data: a.data, path: a.path, parent: a })
          .then((l) =>
            l.status === "aborted"
              ? b
              : (l.status === "dirty" && o.dirty(),
                c(l.value).then(() => ({ status: o.value, value: l.value }))),
          );
    }
    if (n.type === "transform")
      if (a.common.async === !1) {
        let c = this._def.schema._parseSync({
          data: a.data,
          path: a.path,
          parent: a,
        });
        if (!J(c)) return b;
        let l = n.transform(c.value, s);
        if (l instanceof Promise)
          throw new Error(
            "Asynchronous transform encountered during synchronous parse operation. Use .parseAsync instead.",
          );
        return { status: o.value, value: l };
      } else
        return this._def.schema
          ._parseAsync({ data: a.data, path: a.path, parent: a })
          .then((c) =>
            J(c)
              ? Promise.resolve(n.transform(c.value, s)).then((l) => ({
                  status: o.value,
                  value: l,
                }))
              : b,
          );
    C.assertNever(n);
  }
};
O.create = (r, e, o) =>
  new O({ schema: r, typeName: g.ZodEffects, effect: e, ...x(o) });
O.createWithPreprocess = (r, e, o) =>
  new O({
    schema: e,
    effect: { type: "preprocess", transform: r },
    typeName: g.ZodEffects,
    ...x(o),
  });
var k = class extends _ {
  _parse(e) {
    return this._getType(e) === y.undefined
      ? T(void 0)
      : this._def.innerType._parse(e);
  }
  unwrap() {
    return this._def.innerType;
  }
};
k.create = (r, e) => new k({ innerType: r, typeName: g.ZodOptional, ...x(e) });
var F = class extends _ {
  _parse(e) {
    return this._getType(e) === y.null
      ? T(null)
      : this._def.innerType._parse(e);
  }
  unwrap() {
    return this._def.innerType;
  }
};
F.create = (r, e) => new F({ innerType: r, typeName: g.ZodNullable, ...x(e) });
var ye = class extends _ {
  _parse(e) {
    let { ctx: o } = this._processInputParams(e),
      a = o.data;
    return (
      o.parsedType === y.undefined && (a = this._def.defaultValue()),
      this._def.innerType._parse({ data: a, path: o.path, parent: o })
    );
  }
  removeDefault() {
    return this._def.innerType;
  }
};
ye.create = (r, e) =>
  new ye({
    innerType: r,
    typeName: g.ZodDefault,
    defaultValue: typeof e.default == "function" ? e.default : () => e.default,
    ...x(e),
  });
var he = class extends _ {
  _parse(e) {
    let { ctx: o } = this._processInputParams(e),
      a = { ...o, common: { ...o.common, issues: [] } },
      n = this._def.innerType._parse({
        data: a.data,
        path: a.path,
        parent: { ...a },
      });
    return Oe(n)
      ? n.then((s) => ({
          status: "valid",
          value:
            s.status === "valid"
              ? s.value
              : this._def.catchValue({
                  get error() {
                    return new A(a.common.issues);
                  },
                  input: a.data,
                }),
        }))
      : {
          status: "valid",
          value:
            n.status === "valid"
              ? n.value
              : this._def.catchValue({
                  get error() {
                    return new A(a.common.issues);
                  },
                  input: a.data,
                }),
        };
  }
  removeCatch() {
    return this._def.innerType;
  }
};
he.create = (r, e) =>
  new he({
    innerType: r,
    typeName: g.ZodCatch,
    catchValue: typeof e.catch == "function" ? e.catch : () => e.catch,
    ...x(e),
  });
var Le = class extends _ {
  _parse(e) {
    if (this._getType(e) !== y.nan) {
      let a = this._getOrReturnCtx(e);
      return (
        f(a, { code: m.invalid_type, expected: y.nan, received: a.parsedType }),
        b
      );
    }
    return { status: "valid", value: e.data };
  }
};
Le.create = (r) => new Le({ typeName: g.ZodNaN, ...x(r) });
var Xc = Symbol("zod_brand"),
  St = class extends _ {
    _parse(e) {
      let { ctx: o } = this._processInputParams(e),
        a = o.data;
      return this._def.type._parse({ data: a, path: o.path, parent: o });
    }
    unwrap() {
      return this._def.type;
    }
  },
  vt = class r extends _ {
    _parse(e) {
      let { status: o, ctx: a } = this._processInputParams(e);
      if (a.common.async)
        return (async () => {
          let s = await this._def.in._parseAsync({
            data: a.data,
            path: a.path,
            parent: a,
          });
          return s.status === "aborted"
            ? b
            : s.status === "dirty"
              ? (o.dirty(), re(s.value))
              : this._def.out._parseAsync({
                  data: s.value,
                  path: a.path,
                  parent: a,
                });
        })();
      {
        let n = this._def.in._parseSync({
          data: a.data,
          path: a.path,
          parent: a,
        });
        return n.status === "aborted"
          ? b
          : n.status === "dirty"
            ? (o.dirty(), { status: "dirty", value: n.value })
            : this._def.out._parseSync({
                data: n.value,
                path: a.path,
                parent: a,
              });
      }
    }
    static create(e, o) {
      return new r({ in: e, out: o, typeName: g.ZodPipeline });
    }
  },
  be = class extends _ {
    _parse(e) {
      let o = this._def.innerType._parse(e),
        a = (n) => (J(n) && (n.value = Object.freeze(n.value)), n);
      return Oe(o) ? o.then((n) => a(n)) : a(o);
    }
    unwrap() {
      return this._def.innerType;
    }
  };
be.create = (r, e) =>
  new be({ innerType: r, typeName: g.ZodReadonly, ...x(e) });
function ra(r, e) {
  let o =
    typeof r == "function" ? r(e) : typeof r == "string" ? { message: r } : r;
  return typeof o == "string" ? { message: o } : o;
}
function ia(r, e = {}, o) {
  return r
    ? K.create().superRefine((a, n) => {
        let s = r(a);
        if (s instanceof Promise)
          return s.then((c) => {
            if (!c) {
              let l = ra(e, a),
                u = l.fatal ?? o ?? !0;
              n.addIssue({ code: "custom", ...l, fatal: u });
            }
          });
        if (!s) {
          let c = ra(e, a),
            l = c.fatal ?? o ?? !0;
          n.addIssue({ code: "custom", ...c, fatal: l });
        }
      })
    : K.create();
}
var Kc = { object: I.lazycreate },
  g;
(function (r) {
  ((r.ZodString = "ZodString"),
    (r.ZodNumber = "ZodNumber"),
    (r.ZodNaN = "ZodNaN"),
    (r.ZodBigInt = "ZodBigInt"),
    (r.ZodBoolean = "ZodBoolean"),
    (r.ZodDate = "ZodDate"),
    (r.ZodSymbol = "ZodSymbol"),
    (r.ZodUndefined = "ZodUndefined"),
    (r.ZodNull = "ZodNull"),
    (r.ZodAny = "ZodAny"),
    (r.ZodUnknown = "ZodUnknown"),
    (r.ZodNever = "ZodNever"),
    (r.ZodVoid = "ZodVoid"),
    (r.ZodArray = "ZodArray"),
    (r.ZodObject = "ZodObject"),
    (r.ZodUnion = "ZodUnion"),
    (r.ZodDiscriminatedUnion = "ZodDiscriminatedUnion"),
    (r.ZodIntersection = "ZodIntersection"),
    (r.ZodTuple = "ZodTuple"),
    (r.ZodRecord = "ZodRecord"),
    (r.ZodMap = "ZodMap"),
    (r.ZodSet = "ZodSet"),
    (r.ZodFunction = "ZodFunction"),
    (r.ZodLazy = "ZodLazy"),
    (r.ZodLiteral = "ZodLiteral"),
    (r.ZodEnum = "ZodEnum"),
    (r.ZodEffects = "ZodEffects"),
    (r.ZodNativeEnum = "ZodNativeEnum"),
    (r.ZodOptional = "ZodOptional"),
    (r.ZodNullable = "ZodNullable"),
    (r.ZodDefault = "ZodDefault"),
    (r.ZodCatch = "ZodCatch"),
    (r.ZodPromise = "ZodPromise"),
    (r.ZodBranded = "ZodBranded"),
    (r.ZodPipeline = "ZodPipeline"),
    (r.ZodReadonly = "ZodReadonly"));
})(g || (g = {}));
var Yc = (r, e = { message: `Input not instance of ${r.name}` }) =>
    ia((o) => o instanceof r, e),
  ca = X.create,
  la = oe.create,
  Qc = Le.create,
  el = ae.create,
  da = ne.create,
  tl = se.create,
  rl = ze.create,
  ol = ie.create,
  al = ce.create,
  nl = K.create,
  sl = Z.create,
  il = E.create,
  cl = Ee.create,
  ll = G.create,
  dl = I.create,
  pl = I.strictCreate,
  ml = le.create,
  ul = Yr.create,
  fl = de.create,
  yl = L.create,
  hl = Qr.create,
  bl = De.create,
  gl = Me.create,
  wl = eo.create,
  xl = pe.create,
  _l = me.create,
  Cl = ue.create,
  Sl = fe.create,
  vl = Y.create,
  Tl = O.create,
  Pl = k.create,
  Rl = F.create,
  Al = O.createWithPreprocess,
  Il = vt.create,
  kl = () => ca().optional(),
  jl = () => la().optional(),
  Ol = () => da().optional(),
  Bl = {
    string: (r) => X.create({ ...r, coerce: !0 }),
    number: (r) => oe.create({ ...r, coerce: !0 }),
    boolean: (r) => ne.create({ ...r, coerce: !0 }),
    bigint: (r) => ae.create({ ...r, coerce: !0 }),
    date: (r) => se.create({ ...r, coerce: !0 }),
  };
var zl = b;
var pa = t.object({ id: t.string(), description: t.string() }),
  El = t.object({
    browser: t.array(pa).optional(),
    tab: t.array(pa).optional(),
  }),
  Dl = t.enum(["iab", "extension", "cdp"]),
  Ml = t.string().trim().min(1),
  zo = t.object({
    family: Ml.optional(),
    id: t.string(),
    name: t.string(),
    type: Dl,
    profileName: t.string().optional(),
    metadata: t
      .object({
        extensionInstanceId: t.string().optional(),
        codexSessionId: t.string().optional(),
      })
      .optional(),
  }),
  Fe = zo.extend({
    apiSupportOverrides: t.record(t.boolean()).optional(),
    capabilities: El,
  });
var ma = t.object({}),
  Ll = t.array(zo),
  ua = "list_browsers";
function Fl(r = {}) {
  return new i(ua, ma, r);
}
var Pt = {};
d(Pt, {
  PayloadSchema: () => fa,
  ResultSchema: () => Nl,
  commandType: () => ya,
  create: () => $l,
});
var fa = t.object({ id: t.string() }),
  Nl = Fe,
  ya = "get_browser";
function $l(r) {
  return new i(ya, fa, r);
}
var Rt = {};
d(Rt, {
  PayloadSchema: () => ha,
  ResultSchema: () => ql,
  commandType: () => ba,
  create: () => Ul,
});
var ha = t.object({}),
  ql = Fe,
  ba = "get_default_browser";
function Ul(r = {}) {
  return new i(ba, ha, r);
}
var At = {};
d(At, {
  PayloadSchema: () => ga,
  ResultSchema: () => Wl,
  commandType: () => wa,
  create: () => Vl,
});
var ga = t.object({ name: t.string() }),
  Wl = t.string(),
  wa = "get_documentation";
function Vl(r) {
  return new i(wa, ga, r);
}
var It = {};
d(It, {
  PayloadSchema: () => xa,
  ResultSchema: () => Zl,
  commandType: () => _a,
  create: () => Gl,
});
var xa = t.object({ browser_id: t.string() }),
  Zl = t.string(),
  _a = "get_browser_documentation";
function Gl(r) {
  return new i(_a, xa, r);
}
var kt = {};
d(kt, {
  PayloadSchema: () => Ca,
  ResultSchema: () => Hl,
  commandType: () => Sa,
  create: () => Jl,
});
var Ca = t.object({ url: t.string().url() }),
  Hl = Fe,
  Sa = "get_browser_for_url";
function Jl(r) {
  return new i(Sa, Ca, r);
}
var jt = {};
d(jt, {
  PayloadSchema: () => va,
  ResultSchema: () => Xl,
  commandType: () => Ta,
  create: () => Kl,
});
var va = t.object({ browser_id: t.string() }),
  Xl = t.object({ id: t.string() }),
  Ta = "create_tab";
function Kl(r) {
  return new i(Ta, va, r);
}
var ge = {};
d(ge, {
  PayloadSchema: () => Pa,
  ResultSchema: () => Yl,
  commandType: () => Ra,
  create: () => Ql,
});
var Pa = t.object({ browser_id: t.string(), tab_id: t.string() }),
  Yl = t.object({
    id: t.string(),
    title: t.string().optional(),
    url: t.string().optional(),
  }),
  Ra = "get_tab";
function Ql(r) {
  return new i(Ra, Pa, r);
}
var Ot = {};
d(Ot, {
  PayloadSchema: () => Aa,
  ResultSchema: () => ed,
  commandType: () => Ia,
  create: () => td,
});
var Aa = t.object({ browser_id: t.string(), tab_id: t.string() }),
  ed = t.object({}),
  Ia = "close_tab";
function td(r) {
  return new i(Ia, Aa, r);
}
var Bt = {};
d(Bt, {
  MarkTabStatusSchema: () => ka,
  PayloadSchema: () => ja,
  ResultSchema: () => rd,
  commandType: () => Oa,
  create: () => od,
});
var ka = t.enum(["handoff", "deliverable"]),
  ja = t.object({ browser_id: t.string(), tab_id: t.string(), status: ka }),
  rd = t.object({}),
  Oa = "mark_tab";
function od(r) {
  return new i(Oa, ja, r);
}
var zt = {};
d(zt, {
  PayloadSchema: () => za,
  ResultSchema: () => ad,
  commandType: () => Ba,
  create: () => nd,
});
var Ba = "tab_manual_handoff_request",
  za = t.object({ browser_id: t.string(), tab_id: t.string() }),
  ad = t.object({});
function nd(r) {
  return new i(Ba, za, r);
}
var Eo = {};
d(Eo, {
  PayloadSchema: () => Ea,
  ResultSchema: () => sd,
  commandType: () => Da,
  create: () => id,
});
var Ea = t.object({ browser_id: t.string() }),
  sd = t.object({
    display_truncate_max_chars: t.number().int().positive().optional(),
  }),
  Da = "runtime_config";
function id(r) {
  return new i(Da, Ea, r);
}
var Et = {};
d(Et, {
  PayloadSchema: () => Ma,
  ResultSchema: () => cd,
  commandType: () => La,
  create: () => ld,
});
var Ma = t.object({ browser_id: t.string(), name: t.string() }),
  cd = t.object({}),
  La = "name_session";
function ld(r) {
  return new i(La, Ma, r);
}
var Dt = {};
d(Dt, {
  PayloadSchema: () => Fa,
  ResultSchema: () => dd,
  commandType: () => Na,
  create: () => pd,
});
var Fa = t.object({ browser_id: t.string() }),
  dd = t.object({ id: t.string().optional() }),
  Na = "selected_tab";
function pd(r) {
  return new i(Na, Fa, r);
}
var Mt = {};
d(Mt, {
  PayloadSchema: () => $a,
  ResultSchema: () => md,
  commandType: () => qa,
  create: () => ud,
});
var $a = t.object({ browser_id: t.string() }),
  md = t.object({
    tabs: t.array(
      t.object({
        id: t.string(),
        providerTabId: t.string().optional(),
        url: t.string().optional(),
        title: t.string().optional(),
      }),
    ),
  }),
  qa = "list_tabs";
function ud(r) {
  return new i(qa, $a, r);
}
var Lt = {};
d(Lt, {
  ContentTypeSchema: () => Ua,
  PayloadSchema: () => Wa,
  ResultSchema: () => fd,
  commandType: () => Va,
  create: () => yd,
});
var Ua = t.enum(["html", "text", "domSnapshot"]),
  Wa = t.object({
    browser_id: t.string(),
    urls: t.array(t.string()),
    content_type: Ua,
    timeout_ms: t.number().int().positive().optional(),
  }),
  fd = t.object({
    results: t.array(
      t.object({
        url: t.string(),
        title: t.string().nullable(),
        content: t.string().nullable(),
      }),
    ),
  }),
  Va = "tabs_content";
function yd(r) {
  return new i(Va, Wa, r);
}
var Ne = {};
d(Ne, {
  PayloadSchema: () => Za,
  ResultSchema: () => gd,
  commandType: () => Ga,
  create: () => wd,
});
var hd = t.object({
    readOnlyHint: t.boolean().optional(),
    untrustedContentHint: t.boolean().optional(),
  }),
  bd = t.object({
    name: t.string(),
    registration_id: t.string(),
    title: t.string().optional(),
    description: t.string().optional(),
    input_schema: t.unknown(),
    annotations: hd.optional(),
    origin: t.string().optional(),
    pageUrl: t.string().optional(),
  }),
  Za = t.object({ browser_id: t.string(), tab_id: t.string() }),
  gd = t.object({ tools: t.array(bd) }),
  Ga = "webmcp_list_tools";
function wd(r) {
  return new i(Ga, Za, r);
}
var Ft = {};
d(Ft, {
  PayloadSchema: () => Ha,
  ResultSchema: () => xd,
  commandType: () => Ja,
  create: () => _d,
});
var Ha = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    tool_name: t.string(),
    tool_description: t.string().optional(),
    tool_title: t.string().optional(),
    registration_id: t.string(),
    input: t.unknown(),
    timeout_ms: t.number().int().positive().optional(),
  }),
  xd = t.object({ result: t.unknown() }),
  Ja = "webmcp_invoke_tool";
function _d(r) {
  return new i(Ja, Ha, r);
}
var Nt = {};
d(Nt, {
  PayloadSchema: () => Xa,
  ResultSchema: () => Cd,
  commandType: () => Ka,
  create: () => Sd,
});
var Xa = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    url: t.string(),
  }),
  Cd = t.object({}),
  Ka = "navigate_tab_url";
function Sd(r) {
  return new i(Ka, Xa, r);
}
var we = {};
d(we, {
  PayloadSchema: () => Ya,
  ResultSchema: () => vd,
  commandType: () => Qa,
  create: () => Td,
});
var Ya = t.object({ browser_id: t.string(), tab_id: t.string() }),
  vd = t.object({}),
  Qa = "navigate_tab_back";
function Td(r) {
  return new i(Qa, Ya, r);
}
var xe = {};
d(xe, {
  PayloadSchema: () => en,
  ResultSchema: () => Pd,
  commandType: () => tn,
  create: () => Rd,
});
var en = t.object({ browser_id: t.string(), tab_id: t.string() }),
  Pd = t.object({}),
  tn = "navigate_tab_forward";
function Rd(r) {
  return new i(tn, en, r);
}
var $t = {};
d($t, {
  PayloadSchema: () => rn,
  ResultSchema: () => Ad,
  commandType: () => on,
  create: () => Id,
});
var rn = t.object({ browser_id: t.string(), tab_id: t.string() }),
  Ad = t.object({}),
  on = "navigate_tab_reload";
function Id(r) {
  return new i(on, rn, r);
}
var qt = {};
d(qt, {
  PayloadSchema: () => an,
  ResultSchema: () => kd,
  commandType: () => nn,
  create: () => jd,
});
var an = t.object({ browser_id: t.string(), tab_id: t.string() }),
  kd = t.object({
    id: t.string(),
    title: t.string().optional(),
    url: t.string().optional(),
  }),
  nn = "browser_user_claim_tab";
function jd(r) {
  return new i(nn, an, r);
}
var Ut = {};
d(Ut, {
  PayloadSchema: () => sn,
  ResultSchema: () => Od,
  commandType: () => cn,
  create: () => Bd,
});
var sn = t.object({ browser_id: t.string() }),
  Od = t.object({
    tabs: t.array(
      t.object({
        id: t.string(),
        providerTabId: t.string().optional(),
        title: t.string().optional(),
        url: t.string().optional(),
        lastOpened: t.string().optional(),
        tabGroup: t.string().optional(),
      }),
    ),
  }),
  cn = "browser_user_open_tabs";
function Bd(r) {
  return new i(cn, sn, r);
}
var Wt = {};
d(Wt, {
  PayloadSchema: () => ln,
  ResultSchema: () => zd,
  commandType: () => dn,
  create: () => Ed,
});
var ln = t.object({
    browser_id: t.string(),
    queries: t.array(t.string()).min(1).optional(),
    limit: t.number().int().positive().optional(),
    from: t.string().optional(),
    to: t.string().optional(),
  }),
  zd = t.object({
    items: t.array(
      t.object({
        url: t.string(),
        title: t.string().optional(),
        dateVisited: t.string(),
      }),
    ),
  }),
  dn = "browser_user_history";
function Ed(r) {
  return new i(dn, ln, r);
}
var $e = {};
d($e, {
  LogLevelSchema: () => Do,
  PayloadSchema: () => pn,
  ResultSchema: () => Dd,
  commandType: () => mn,
  create: () => Md,
});
var Do = t.enum(["debug", "info", "log", "warn", "error"]),
  pn = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    filter: t.string().optional(),
    levels: t.array(Do).optional(),
    limit: t.number().int().positive().optional(),
  }),
  Dd = t.object({
    logs: t.array(
      t.object({
        level: Do,
        message: t.string(),
        timestamp: t.string(),
        url: t.string().optional(),
      }),
    ),
  }),
  mn = "tab_dev_logs";
function Md(r) {
  return new i(mn, pn, r);
}
var Vt = {};
d(Vt, {
  JsDialogTypeSchema: () => un,
  PayloadSchema: () => yn,
  ResultDialogSchema: () => fn,
  ResultSchema: () => Ld,
  commandType: () => hn,
  create: () => Fd,
});
var un = t.enum(["alert", "beforeunload", "confirm", "prompt"]),
  fn = t.object({ id: t.string(), type: un }),
  yn = t.object({ browser_id: t.string(), tab_id: t.string() }),
  Ld = t.object({ dialog: fn.nullable() }),
  hn = "tab_get_js_dialog";
function Fd(r) {
  return new i(hn, yn, r);
}
var _e = {};
d(_e, {
  PayloadSchema: () => bn,
  ResultSchema: () => Nd,
  commandType: () => gn,
  create: () => $d,
});
var bn = t.object({
    action: t.enum(["accept", "dismiss"]),
    browser_id: t.string(),
    dialog_id: t.string(),
    prompt_text: t.string().optional(),
    tab_id: t.string(),
  }),
  Nd = t.object({}),
  gn = "tab_handle_js_dialog";
function $d(r) {
  return new i(gn, bn, r);
}
var Zt = {};
d(Zt, {
  PayloadSchema: () => wn,
  ResultSchema: () => qd,
  commandType: () => xn,
  create: () => Ud,
});
var wn = t.object({ browser_id: t.string(), tab_id: t.string() }),
  qd = t.object({ path: t.string() }),
  xn = "tab_content_export";
function Ud(r) {
  return new i(xn, wn, r);
}
var Gt = {};
d(Gt, {
  GSuiteExportTypeSchema: () => _n,
  PayloadSchema: () => Cn,
  ResultSchema: () => Wd,
  commandType: () => Sn,
  create: () => Vd,
});
var _n = t.enum(["pdf", "md", "xlsx", "csv", "docx", "pptx"]),
  Cn = t.object({ browser_id: t.string(), tab_id: t.string(), format: _n }),
  Wd = t.object({ path: t.string() }),
  Sn = "tab_content_export_gsuite";
function Vd(r) {
  return new i(Sn, Cn, r);
}
var Ht = {};
d(Ht, {
  PayloadSchema: () => vn,
  ResultSchema: () => Zd,
  commandType: () => Tn,
  create: () => Gd,
});
var vn = t.object({ browser_id: t.string(), tab_id: t.string() }),
  Zd = t.object({ path: t.string() }),
  Tn = "tab_content_export_youtube_transcript";
function Gd(r) {
  return new i(Tn, vn, r);
}
var Jt = {};
d(Jt, {
  PayloadSchema: () => Pn,
  ResultSchema: () => Hd,
  commandType: () => Rn,
  create: () => Jd,
});
var Pn = t.object({ browser_id: t.string(), tab_id: t.string() }),
  Hd = t.object({ text: t.string() }),
  Rn = "tab_clipboard_read_text";
function Jd(r) {
  return new i(Rn, Pn, r);
}
var Xt = {};
d(Xt, {
  PayloadSchema: () => An,
  ResultSchema: () => Xd,
  commandType: () => In,
  create: () => Kd,
});
var An = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    text: t.string(),
  }),
  Xd = t.object({}),
  In = "tab_clipboard_write_text";
function Kd(r) {
  return new i(In, An, r);
}
var Kt = {};
d(Kt, {
  PayloadSchema: () => kn,
  ResultSchema: () => ep,
  commandType: () => jn,
  create: () => tp,
});
var Yd = t
    .object({
      mime_type: t.string(),
      text: t.string().optional(),
      base64: t.string().optional(),
    })
    .superRefine((r, e) => {
      let o = r.text !== void 0,
        a = r.base64 !== void 0;
      o === a &&
        e.addIssue({
          code: t.ZodIssueCode.custom,
          message: "Clipboard entries must set exactly one of text or base64",
        });
    }),
  Qd = t.object({
    entries: t.array(Yd),
    presentation_style: t
      .enum(["unspecified", "inline", "attachment"])
      .optional(),
  }),
  kn = t.object({ browser_id: t.string(), tab_id: t.string() }),
  ep = t.object({ items: t.array(Qd) }),
  jn = "tab_clipboard_read";
function tp(r) {
  return new i(jn, kn, r);
}
var Yt = {};
d(Yt, {
  PayloadSchema: () => On,
  ResultSchema: () => ap,
  commandType: () => Bn,
  create: () => np,
});
var rp = t
    .object({
      mime_type: t.string(),
      text: t.string().optional(),
      base64: t.string().optional(),
    })
    .superRefine((r, e) => {
      let o = r.text !== void 0,
        a = r.base64 !== void 0;
      o === a &&
        e.addIssue({
          code: t.ZodIssueCode.custom,
          message: "Clipboard entries must set exactly one of text or base64",
        });
    }),
  op = t.object({
    entries: t.array(rp),
    presentation_style: t
      .enum(["unspecified", "inline", "attachment"])
      .optional(),
  }),
  On = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    items: t.array(op),
  }),
  ap = t.object({}),
  Bn = "tab_clipboard_write";
function np(r) {
  return new i(Bn, On, r);
}
var Qt = {};
d(Qt, {
  PayloadSchema: () => zn,
  ResultSchema: () => sp,
  commandType: () => En,
  create: () => ip,
});
var zn = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    fullPage: t.boolean().optional(),
    cropX: t.number().optional(),
    cropY: t.number().optional(),
    cropWidth: t.number().optional(),
    cropHeight: t.number().optional(),
  }),
  sp = t.object({ data: t.string() }),
  En = "tab_screenshot";
function ip(r) {
  return new i(En, zn, r);
}
var er = {};
d(er, {
  PayloadSchema: () => Dn,
  ResultSchema: () => cp,
  commandType: () => Mn,
  create: () => lp,
});
var Dn = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    content: t.enum(["axState", "screenshot", "axStateAndScreenshot"]),
    disable_diffing: t.boolean().optional(),
  }),
  cp = t.object({
    state: t.string().optional(),
    data: t.string().optional(),
    screenshot_unavailable: t.string().optional(),
  }),
  Mn = "tab_ax_get_state";
function lp(r) {
  return new i(Mn, Dn, r);
}
var tr = {};
d(tr, {
  ActionSchema: () => Fn,
  PayloadSchema: () => Nn,
  ResultSchema: () => dp,
  commandType: () => $n,
  create: () => pp,
});
var Mo = t.tuple([t.number(), t.number()]),
  to = t.number().int().nonnegative(),
  Ln = t.union([to, Mo]),
  Fn = t.discriminatedUnion("kind", [
    t.object({
      kind: t.literal("click"),
      target: Ln,
      mouse_button: t
        .enum(["left", "right", "middle", "l", "r", "m"])
        .optional(),
      click_count: t.number().int().positive().optional(),
    }),
    t.object({ kind: t.literal("drag"), from: Mo, to: Mo }),
    t.object({
      kind: t.literal("perform_secondary_action"),
      element_index: to,
      action: t.string(),
    }),
    t.object({ kind: t.literal("press_key"), key: t.string() }),
    t.object({
      kind: t.literal("scroll"),
      target: Ln,
      direction: t.enum(["up", "down", "left", "right", "u", "d", "l", "r"]),
      pages: t.number().optional(),
    }),
    t.object({
      kind: t.literal("select_text"),
      element_index: to,
      text: t.string(),
      prefix: t.string().optional(),
      suffix: t.string().optional(),
      selection_type: t
        .enum(["text", "cursor_before", "cursor_after"])
        .optional(),
    }),
    t.object({
      kind: t.literal("set_value"),
      element_index: to,
      value: t.string(),
    }),
    t.object({ kind: t.literal("type_text"), text: t.string() }),
  ]),
  Nn = t.object({ browser_id: t.string(), tab_id: t.string(), action: Fn }),
  dp = t.object({}),
  $n = "tab_ax_action";
function pp(r) {
  return new i($n, Nn, r);
}
var Ce = {};
d(Ce, {
  PayloadSchema: () => qn,
  ResultSchema: () => mp,
  commandType: () => Un,
  create: () => up,
});
var qn = t.object({
    browser_id: t.string(),
    selector: t.string().optional(),
    selector_mode: t.literal("all").optional(),
    tab_id: t.string(),
    script: t.string(),
    timeout_ms: t.number().int().positive().optional(),
  }),
  mp = t.object({ value: t.unknown().optional() }),
  Un = "playwright_evaluate";
function up(r) {
  return new i(Un, qn, r);
}
var rr = {};
d(rr, {
  PayloadSchema: () => Wn,
  ResultSchema: () => fp,
  commandType: () => Vn,
  create: () => yp,
});
var Wn = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    x: t.number(),
    y: t.number(),
    button: t.number().optional(),
    keys: t.array(t.string()).optional(),
  }),
  fp = t.object({}),
  Vn = "cua_click";
function yp(r) {
  return new i(Vn, Wn, r);
}
var or = {};
d(or, {
  PayloadSchema: () => Zn,
  ResultSchema: () => hp,
  commandType: () => Gn,
  create: () => bp,
});
var Zn = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    x: t.number(),
    y: t.number(),
    timeout_ms: t.number().int().positive().optional(),
  }),
  hp = t.object({}),
  Gn = "cua_download_media";
function bp(r) {
  return new i(Gn, Zn, r);
}
var ar = {};
d(ar, {
  PayloadSchema: () => Hn,
  ResultSchema: () => gp,
  commandType: () => Jn,
  create: () => wp,
});
var Hn = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    x: t.number(),
    y: t.number(),
    keys: t.array(t.string()).optional(),
  }),
  gp = t.object({}),
  Jn = "cua_double_click";
function wp(r) {
  return new i(Jn, Hn, r);
}
var nr = {};
d(nr, {
  PayloadSchema: () => Xn,
  ResultSchema: () => xp,
  commandType: () => Kn,
  create: () => _p,
});
var Xn = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    keys: t.array(t.string()),
  }),
  xp = t.object({}),
  Kn = "cua_keypress";
function _p(r) {
  return new i(Kn, Xn, r);
}
var sr = {};
d(sr, {
  PayloadSchema: () => Yn,
  ResultSchema: () => Cp,
  commandType: () => Qn,
  create: () => Sp,
});
var Yn = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    path: t.array(t.object({ x: t.number(), y: t.number() })),
    keys: t.array(t.string()).optional(),
  }),
  Cp = t.object({}),
  Qn = "cua_drag";
function Sp(r) {
  return new i(Qn, Yn, r);
}
var ir = {};
d(ir, {
  PayloadSchema: () => es,
  ResultSchema: () => vp,
  commandType: () => ts,
  create: () => Tp,
});
var es = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    x: t.number(),
    y: t.number(),
    keys: t.array(t.string()).optional(),
  }),
  vp = t.object({}),
  ts = "cua_move";
function Tp(r) {
  return new i(ts, es, r);
}
var cr = {};
d(cr, {
  PayloadSchema: () => rs,
  ResultSchema: () => Pp,
  commandType: () => os,
  create: () => Rp,
});
var rs = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    x: t.number(),
    y: t.number(),
    scroll_x: t.number(),
    scroll_y: t.number(),
    keys: t.array(t.string()).optional(),
  }),
  Pp = t.object({}),
  os = "cua_scroll";
function Rp(r) {
  return new i(os, rs, r);
}
var lr = {};
d(lr, {
  PayloadSchema: () => as,
  ResultSchema: () => Ap,
  commandType: () => ns,
  create: () => Ip,
});
var as = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    text: t.string(),
  }),
  Ap = t.object({}),
  ns = "cua_type";
function Ip(r) {
  return new i(ns, as, r);
}
var qe = {};
d(qe, {
  DomNodeIdSchema: () => N,
  PayloadSchema: () => ss,
  ResultSchema: () => kp,
  commandType: () => is,
  create: () => jp,
});
var N = t
  .string({
    required_error: "node_id is required",
    invalid_type_error: "node_id must be a string",
  })
  .min(1, "node_id must not be empty");
var ss = t.object({ browser_id: t.string(), tab_id: t.string(), node_id: N }),
  kp = t.object({}),
  is = "dom_cua_click";
function jp(r) {
  return new i(is, ss, r);
}
var dr = {};
d(dr, {
  PayloadSchema: () => cs,
  ResultSchema: () => Op,
  commandType: () => ls,
  create: () => Bp,
});
var cs = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    node_id: N,
    timeout_ms: t.number().int().positive().optional(),
  }),
  Op = t.object({}),
  ls = "dom_cua_download_media";
function Bp(r) {
  return new i(ls, cs, r);
}
var pr = {};
d(pr, {
  PayloadSchema: () => ds,
  ResultSchema: () => zp,
  commandType: () => ps,
  create: () => Ep,
});
var ds = t.object({ browser_id: t.string(), tab_id: t.string(), node_id: N }),
  zp = t.object({}),
  ps = "dom_cua_double_click";
function Ep(r) {
  return new i(ps, ds, r);
}
var mr = {};
d(mr, {
  PayloadSchema: () => ms,
  ResultSchema: () => Dp,
  commandType: () => us,
  create: () => Mp,
});
var ms = t.object({ browser_id: t.string(), tab_id: t.string() }),
  Dp = t.unknown(),
  us = "dom_cua_get_visible_dom";
function Mp(r) {
  return new i(us, ms, r);
}
var ur = {};
d(ur, {
  PayloadSchema: () => fs,
  ResultSchema: () => Lp,
  commandType: () => ys,
  create: () => Fp,
});
var fs = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    keys: t.array(t.string()),
  }),
  Lp = t.object({}),
  ys = "dom_cua_keypress";
function Fp(r) {
  return new i(ys, fs, r);
}
var fr = {};
d(fr, {
  PayloadSchema: () => hs,
  ResultSchema: () => Np,
  commandType: () => bs,
  create: () => $p,
});
var hs = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    scroll_x: t.number(),
    scroll_y: t.number(),
    node_id: N.optional(),
  }),
  Np = t.object({}),
  bs = "dom_cua_scroll";
function $p(r) {
  return new i(bs, hs, r);
}
var yr = {};
d(yr, {
  PayloadSchema: () => gs,
  ResultSchema: () => qp,
  commandType: () => ws,
  create: () => Up,
});
var gs = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    text: t.string(),
  }),
  qp = t.object({}),
  ws = "dom_cua_type";
function Up(r) {
  return new i(ws, gs, r);
}
var hr = {};
d(hr, {
  PayloadSchema: () => xs,
  ResultSchema: () => Wp,
  commandType: () => _s,
  create: () => Vp,
});
var xs = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    selector: t.string(),
    modifiers: t
      .array(t.enum(["Alt", "Control", "ControlOrMeta", "Meta", "Shift"]))
      .optional(),
    button: t.enum(["left", "right", "middle"]).optional(),
    force: t.boolean().optional(),
    timeout_ms: t.number().int().positive().optional(),
  }),
  Wp = t.object({}),
  _s = "playwright_locator_click";
function Vp(r) {
  return new i(_s, xs, r);
}
var br = {};
d(br, {
  PayloadSchema: () => Cs,
  ResultSchema: () => Zp,
  commandType: () => Ss,
  create: () => Gp,
});
var Cs = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    selector: t.string(),
    modifiers: t
      .array(t.enum(["Alt", "Control", "ControlOrMeta", "Meta", "Shift"]))
      .optional(),
    button: t.enum(["left", "right", "middle"]).optional(),
    force: t.boolean().optional(),
    timeout_ms: t.number().int().positive().optional(),
  }),
  Zp = t.object({}),
  Ss = "playwright_locator_dblclick";
function Gp(r) {
  return new i(Ss, Cs, r);
}
var gr = {};
d(gr, {
  PayloadSchema: () => vs,
  ResultSchema: () => Hp,
  commandType: () => Ts,
  create: () => Jp,
});
var vs = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    selector: t.string(),
    timeout_ms: t.number().int().positive().optional(),
  }),
  Hp = t.object({}),
  Ts = "playwright_locator_download_media";
function Jp(r) {
  return new i(Ts, vs, r);
}
var Ue = {};
d(Ue, {
  PayloadSchema: () => Ps,
  ResultSchema: () => Xp,
  commandType: () => Rs,
  create: () => Kp,
});
var Ps = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    selector: t.string(),
    value: t.string(),
    replace: t.boolean(),
  }),
  Xp = t.object({}),
  Rs = "playwright_locator_fill";
function Kp(r) {
  return new i(Rs, Ps, r);
}
var wr = {};
d(wr, {
  PayloadSchema: () => As,
  ResultSchema: () => Yp,
  commandType: () => Is,
  create: () => Qp,
});
var As = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    selector: t.string(),
    value: t.string(),
  }),
  Yp = t.object({}),
  Is = "playwright_locator_press";
function Qp(r) {
  return new i(Is, As, r);
}
var xr = {};
d(xr, {
  PayloadSchema: () => ks,
  ResultSchema: () => em,
  commandType: () => js,
  create: () => tm,
});
var ks = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    selector: t.string(),
    value: t.string(),
    timeout_ms: t.number().int().positive().optional(),
  }),
  em = t.object({}),
  js = "playwright_locator_press_sequentially";
function tm(r) {
  return new i(js, ks, r);
}
var _r = {};
d(_r, {
  PayloadSchema: () => Os,
  ResultSchema: () => rm,
  commandType: () => Bs,
  create: () => om,
});
var Os = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    selector: t.string(),
    state: t.enum(["attached", "detached", "visible", "hidden"]),
    timeout_ms: t.number().int().positive().optional(),
  }),
  rm = t.object({}),
  Bs = "playwright_locator_wait_for";
function om(r) {
  return new i(Bs, Os, r);
}
var Cr = {};
d(Cr, {
  PayloadSchema: () => zs,
  ResultSchema: () => am,
  commandType: () => Es,
  create: () => nm,
});
var zs = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    selector: t.string(),
  }),
  am = t.object({ count: t.number().int() }),
  Es = "playwright_locator_count";
function nm(r) {
  return new i(Es, zs, r);
}
var Sr = {};
d(Sr, {
  PayloadSchema: () => Ms,
  ResultSchema: () => sm,
  SelectOptionSchema: () => Ds,
  commandType: () => Ls,
  create: () => im,
});
var Ds = t
    .object({
      value: t.string().optional(),
      label: t.string().optional(),
      index: t.number().int().nonnegative().optional(),
    })
    .superRefine((r, e) => {
      r.value === void 0 &&
        r.label === void 0 &&
        r.index === void 0 &&
        e.addIssue({
          code: t.ZodIssueCode.custom,
          message: "Select option requires value, label, or index",
        });
    }),
  Ms = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    selector: t.string(),
    selections: t.array(Ds).min(1),
    timeout_ms: t.number().int().positive().optional(),
  }),
  sm = t.object({}),
  Ls = "playwright_locator_select_option";
function im(r) {
  return new i(Ls, Ms, r);
}
var vr = {};
d(vr, {
  PayloadSchema: () => Fs,
  ResultSchema: () => cm,
  commandType: () => Ns,
  create: () => lm,
});
var Fs = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    selector: t.string(),
    checked: t.boolean(),
    force: t.boolean().optional(),
    timeout_ms: t.number().int().positive().optional(),
  }),
  cm = t.object({}),
  Ns = "playwright_locator_set_checked";
function lm(r) {
  return new i(Ns, Fs, r);
}
var Tr = {};
d(Tr, {
  PayloadSchema: () => $s,
  ResultSchema: () => dm,
  commandType: () => qs,
  create: () => pm,
});
var $s = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    selector: t.string(),
  }),
  dm = t.object({ value: t.boolean() }),
  qs = "playwright_locator_is_visible";
function pm(r) {
  return new i(qs, $s, r);
}
var Pr = {};
d(Pr, {
  PayloadSchema: () => Us,
  ResultSchema: () => mm,
  commandType: () => Ws,
  create: () => um,
});
var Us = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    selector: t.string(),
  }),
  mm = t.object({ value: t.boolean() }),
  Ws = "playwright_locator_is_enabled";
function um(r) {
  return new i(Ws, Us, r);
}
var Rr = {};
d(Rr, {
  PayloadSchema: () => Vs,
  ResultSchema: () => fm,
  commandType: () => Zs,
  create: () => ym,
});
var Vs = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    selector: t.string(),
    timeout_ms: t.number().int().positive().optional(),
  }),
  fm = t.object({ values: t.array(t.string()) }),
  Zs = "playwright_locator_all_text_contents";
function ym(r) {
  return new i(Zs, Vs, r);
}
var Ar = {};
d(Ar, {
  PayloadSchema: () => Gs,
  ResultSchema: () => hm,
  commandType: () => Hs,
  create: () => bm,
});
var Gs = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    selector: t.string(),
    timeout_ms: t.number().int().positive().optional(),
  }),
  hm = t.object({ value: t.string().nullable() }),
  Hs = "playwright_locator_text_content";
function bm(r) {
  return new i(Hs, Gs, r);
}
var Ir = {};
d(Ir, {
  PayloadSchema: () => Js,
  ResultSchema: () => gm,
  commandType: () => Xs,
  create: () => wm,
});
var Js = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    selector: t.string(),
    timeout_ms: t.number().int().positive().optional(),
  }),
  gm = t.object({ value: t.string() }),
  Xs = "playwright_locator_inner_text";
function wm(r) {
  return new i(Xs, Js, r);
}
var kr = {};
d(kr, {
  PayloadSchema: () => Ks,
  ResultSchema: () => xm,
  commandType: () => Ys,
  create: () => _m,
});
var Ks = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    selector: t.string(),
    name: t.string(),
    timeout_ms: t.number().int().positive().optional(),
  }),
  xm = t.object({ value: t.string().nullable() }),
  Ys = "playwright_locator_get_attribute";
function _m(r) {
  return new i(Ys, Ks, r);
}
var jr = {};
d(jr, {
  PayloadSchema: () => Qs,
  ResultSchema: () => Sm,
  commandType: () => ei,
  create: () => vm,
});
var Qs = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    selector: t.string(),
    relative_selector: t.string().optional(),
    timeout_ms: t.number().int().positive().optional(),
  }),
  Cm = t.object({
    attributes: t.record(t.string()),
    inner_text: t.string(),
    text_content: t.string().nullable(),
  }),
  Sm = t.object({ values: t.array(Cm.nullable()) }),
  ei = "playwright_locator_read_all";
function vm(r) {
  return new i(ei, Qs, r);
}
var Or = {};
d(Or, {
  PayloadSchema: () => ti,
  ResultSchema: () => Pm,
  commandType: () => ri,
  create: () => Rm,
});
var Tm = t.enum(["load", "domcontentloaded", "networkidle", "commit"]),
  ti = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    url: t.string(),
    wait_until: Tm.optional(),
    timeout_ms: t.number().int().positive().optional(),
  }),
  Pm = t.object({ url: t.string().optional() }),
  ri = "playwright_wait_for_url";
function Rm(r) {
  return new i(ri, ti, r);
}
var Br = {};
d(Br, {
  LoadStateSchema: () => oi,
  PayloadSchema: () => ai,
  ResultSchema: () => Am,
  commandType: () => ni,
  create: () => Im,
});
var oi = t.enum(["load", "domcontentloaded", "networkidle"]),
  ai = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    state: oi.optional(),
    timeout_ms: t.number().int().positive().optional(),
  }),
  Am = t.object({}),
  ni = "playwright_wait_for_load_state";
function Im(r) {
  return new i(ni, ai, r);
}
var zr = {};
d(zr, {
  PayloadSchema: () => si,
  ResultSchema: () => km,
  commandType: () => ii,
  create: () => jm,
});
var si = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    timeout_ms: t.number().int().nonnegative(),
  }),
  km = t.object({}),
  ii = "playwright_wait_for_timeout";
function jm(r) {
  return new i(ii, si, r);
}
var Er = {};
d(Er, {
  PayloadSchema: () => ci,
  ResultSchema: () => Om,
  commandType: () => li,
  create: () => Bm,
});
var ci = t.object({ browser_id: t.string(), tab_id: t.string() }),
  Om = t.object({ dom_snapshot: t.string() }),
  li = "playwright_dom_snapshot";
function Bm(r) {
  return new i(li, ci, r);
}
var Dr = {};
d(Dr, {
  PayloadSchema: () => di,
  ResultSchema: () => Mm,
  commandType: () => pi,
  create: () => Lm,
});
var zm = t.object({
    x: t.number(),
    y: t.number(),
    width: t.number(),
    height: t.number(),
  }),
  Em = t.object({
    primary: t.string().nullable().optional(),
    candidates: t.array(t.string()),
    frameSelectors: t.array(t.string()).optional(),
  }),
  Dm = t.object({
    nodeId: t.number().int().positive().nullable().optional(),
    tagName: t.string(),
    role: t.string().nullable().optional(),
    visibleText: t.string().nullable().optional(),
    ariaName: t.string().nullable().optional(),
    testId: t.string().nullable().optional(),
    boundingBox: zm.nullable().optional(),
    preview: t.string(),
    selector: Em,
  }),
  di = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    x: t.number(),
    y: t.number(),
    include_non_interactable: t.boolean().optional(),
  }),
  Mm = t.array(Dm),
  pi = "playwright_element_info";
function Lm(r) {
  return new i(pi, di, r);
}
var Mr = {};
d(Mr, {
  PayloadSchema: () => mi,
  ResultSchema: () => Fm,
  commandType: () => ui,
  create: () => Nm,
});
var mi = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    x: t.number(),
    y: t.number(),
    include_non_interactable: t.boolean().optional(),
  }),
  Fm = t.object({ data: t.string() }),
  ui = "playwright_element_screenshot";
function Nm(r) {
  return new i(ui, mi, r);
}
var Lr = {};
d(Lr, {
  PayloadSchema: () => fi,
  ResultSchema: () => $m,
  commandType: () => yi,
  create: () => qm,
});
var fi = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    timeout_ms: t.number().int().positive().optional(),
  }),
  $m = t.object({ download_id: t.string() }),
  yi = "playwright_wait_for_download";
function qm(r) {
  return new i(yi, fi, r);
}
var Fr = {};
d(Fr, {
  PayloadSchema: () => hi,
  ResultSchema: () => Um,
  commandType: () => bi,
  create: () => Wm,
});
var hi = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    download_id: t.string(),
    timeout_ms: t.number().int().positive().optional(),
  }),
  Um = t.object({ path: t.string().nullable() }),
  bi = "playwright_download_path";
function Wm(r) {
  return new i(bi, hi, r);
}
var Nr = {};
d(Nr, {
  PayloadSchema: () => gi,
  ResultSchema: () => Vm,
  commandType: () => wi,
  create: () => Zm,
});
var gi = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    timeout_ms: t.number().int().positive().optional(),
  }),
  Vm = t.object({ file_chooser_id: t.string(), is_multiple: t.boolean() }),
  wi = "playwright_wait_for_file_chooser";
function Zm(r) {
  return new i(wi, gi, r);
}
var $r = {};
d($r, {
  PayloadSchema: () => xi,
  ResultSchema: () => Gm,
  commandType: () => _i,
  create: () => Hm,
});
var xi = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    file_chooser_id: t.string(),
    files: t.array(t.string()),
    timeout_ms: t.number().int().positive().optional(),
  }),
  Gm = t.object({}),
  _i = "playwright_file_chooser_set_files";
function Hm(r) {
  return new i(_i, xi, r);
}
var We = class {
  #e;
  #t;
  #r;
  constructor({ createBrowser: e, onBrowserUsed: o, transport: a }) {
    ((this.#e = e), (this.#t = o), (this.#r = a));
  }
  async list() {
    return await this.#r.send({ command: Tt.create() });
  }
  async get(e) {
    if (!e) throw new Error("browsers.get requires a browser id");
    let o = await this.#r.send({ command: Pt.create({ id: e }) });
    return this.#o(o);
  }
  async getDefault() {
    let e = await this.#r.send({ command: Rt.create() });
    return this.#o(e);
  }
  async getForUrl(e) {
    let o = await this.#r.send({ command: kt.create({ url: e }) });
    return this.#o(o);
  }
  #o(e) {
    return (
      this.#t?.(e),
      this.#e({
        browserInfo: e,
        onBrowserUsed: () => this.#t?.(e),
        transport: this.#r,
      })
    );
  }
};
var Q = class {
  #e;
  constructor(e) {
    this.#e = e;
  }
  async get(e) {
    return await this.#e.send({ command: At.create({ name: e }) });
  }
};
var Ve = class {
  constructor({ createBrowser: e, onBrowserUsed: o, transport: a }) {
    if (!a) throw new Error("Agent requires a transport instance");
    ((this.documentation = new Q(a)),
      (this.browsers = new We({
        createBrowser: e,
        onBrowserUsed: o,
        transport: a,
      })));
  }
};
var Ze = class {
  #e;
  #t;
  #r;
  constructor({ browserId: e, createTab: o, transport: a }) {
    ((this.#e = e), (this.#t = o), (this.#r = a));
  }
  async new() {
    let e = await this.#r.send({ command: jt.create({ browser_id: this.#e }) });
    return this.#t(e);
  }
  async selected() {
    let e = await this.#r.send({ command: Dt.create({ browser_id: this.#e }) });
    if (e.id) return this.#t(e);
  }
  async list() {
    return (await this.#r.send({ command: Mt.create({ browser_id: this.#e }) }))
      .tabs;
  }
  async content(e) {
    let o = Lt.create({
      browser_id: this.#e,
      urls: e.urls,
      content_type: e.contentType,
      ...(e.timeoutMs === void 0 ? {} : { timeout_ms: e.timeoutMs }),
    });
    return (
      o.parse(),
      e.urls.length === 0 ? [] : (await this.#r.send({ command: o })).results
    );
  }
  async get(e) {
    if (!e) throw new Error("tabs.get requires a tab id");
    let o = await this.#r.send({
      command: ge.create({ browser_id: this.#e, tab_id: e }),
    });
    return this.#t(o);
  }
};
var Ge = class {
    #e;
    #t;
    #r;
    constructor({ browserId: e, createTab: o, transport: a }) {
      ((this.#e = e), (this.#t = o), (this.#r = a));
    }
    async openTabs() {
      return (
        await this.#r.send({ command: Ut.create({ browser_id: this.#e }) })
      ).tabs;
    }
    async claimTab(e) {
      let o = Jm(e),
        a = await this.#r.send({
          command: qt.create({ browser_id: this.#e, tab_id: o }),
        });
      return this.#t(a);
    }
  },
  Jm = (r) => {
    if (typeof r == "string") {
      if (r.length === 0)
        throw new Error("browser.user.claimTab received an empty tab id");
      return r;
    }
    if (r && typeof r == "object" && typeof r.id == "string") return r.id;
    throw new Error(
      "browser.user.claimTab expects a tab returned by browser.user.openTabs() or a tab id",
    );
  };
var Ci = Object.getOwnPropertyDescriptor(RegExp.prototype, "source")?.get;
function He(r) {
  if (r == null || typeof r != "object" || !Ci) return !1;
  try {
    return (Reflect.apply(Ci, r, []), !0);
  } catch {
    return !1;
  }
}
function ro(r) {
  if (r == null || typeof r != "object") return !1;
  let e = Object.getPrototypeOf(r);
  return e === null || Object.getPrototypeOf(e) === null;
}
function Si(r, e) {
  if (r == null || typeof r != "object") return !1;
  try {
    return "message" in r && r.message === e;
  } catch {
    return !1;
  }
}
var vi = () =>
    `req-${Date.now().toString(16)}-${Math.random().toString(16).slice(2)}`,
  Je = (r) => {
    let e = atob(r),
      o = new Uint8Array(e.length);
    for (let a = 0; a < e.length; a += 1) o[a] = e.charCodeAt(a);
    return o;
  };
function oo(r, e) {
  let o = r instanceof Error ? r.message : String(r),
    a = new Error(`${o}
${e}`);
  return (
    r instanceof Error &&
      r.stack &&
      (a.stack = `${a.name}: ${a.message}
${r.stack}`),
    a
  );
}
function Xm(r, e) {
  let o = r instanceof Error ? r.message : String(r);
  return /strict mode violation/i.test(o)
    ? "multiple_matches"
    : /intercept|receives pointer events/i.test(o)
      ? "intercepted"
      : e.matchCount === 0
        ? "no_matches"
        : e.visibleCount === 0
          ? "no_visible_match"
          : "action_failed";
}
function Km(r, e) {
  let o = Ai(e);
  if (typeof r == "string") {
    if (r.length === 0)
      throw new Error("playwright.evaluate requires a pageFunction");
    return `const arg = ${o};
return (${r});`;
  }
  if (typeof r == "function")
    return [
      `const arg = ${o};`,
      `const __playwrightEvaluate = (${r.toString()});`,
      "return await __playwrightEvaluate(arg);",
    ].join(`
`);
  throw new Error("playwright.evaluate requires a string or function");
}
function Ti(r, e, o) {
  let a = Ai(e),
    n = o === "all" ? "locator.evaluateAll" : "locator.evaluate";
  if (typeof r == "string") {
    if (r.length === 0) throw new Error(`${n} requires a pageFunction`);
    return `const arg = ${a};
return (${r});`;
  }
  if (typeof r == "function")
    return [
      `const arg = ${a};`,
      `const __playwrightEvaluate = (${r.toString()});`,
      `return await __playwrightEvaluate(${o === "all" ? "elements" : "element"}, arg);`,
    ].join(`
`);
  throw new Error(`${n} requires a string or function`);
}
function Ai(r) {
  if (r === void 0) return "undefined";
  try {
    let e = JSON.stringify(r);
    if (e !== void 0) return e;
  } catch (e) {
    throw oo(e, "playwright.evaluate arg must be JSON-serializable");
  }
  throw new Error("playwright.evaluate arg must be JSON-serializable");
}
function Pi(r, e) {
  let { x: o, y: a } = e;
  if (!Number.isFinite(o) || !Number.isFinite(a))
    throw new Error(`${r} requires numeric x and y coordinates`);
  return { x: o, y: a };
}
function Ym(r) {
  let e = Array.isArray(r) ? r : [r];
  if (!e.length)
    throw new Error("locator.selectOption requires at least one value");
  return e.map((o) => {
    if (typeof o == "string") return { value: o };
    if (!o || typeof o != "object")
      throw new Error(
        "locator.selectOption requires a string or { value?, label?, index? }",
      );
    let a = {};
    if (o.value !== void 0) {
      if (typeof o.value != "string")
        throw new Error("locator.selectOption value must be a string");
      a.value = o.value;
    }
    if (o.label !== void 0) {
      if (typeof o.label != "string")
        throw new Error("locator.selectOption label must be a string");
      a.label = o.label;
    }
    if (o.index !== void 0) {
      if (!Number.isInteger(o.index) || o.index < 0)
        throw new Error(
          "locator.selectOption index must be a non-negative integer",
        );
      a.index = o.index;
    }
    if (a.value === void 0 && a.label === void 0 && a.index === void 0)
      throw new Error(
        "locator.selectOption requires value, label, or index for each selection",
      );
    return a;
  });
}
var Ri = Symbol("browserAuthSelectorDetails"),
  H = class r {
    #e;
    #t;
    #r;
    #o;
    #a;
    #n;
    #s;
    constructor({
      browserId: e,
      tabId: o,
      selector: a,
      transport: n,
      collectionReadCache: s,
      collectionReadIndex: c,
      collectionRelativeSelector: l,
    }) {
      ((this.#e = e),
        (this.#t = o),
        (this.#r = a),
        (this.#o = n),
        (this.#a = s),
        (this.#n = c),
        (this.#s = l));
    }
    static browserAuthSelector(e) {
      return e[Ri]();
    }
    [Ri]() {
      return { browserId: this.#e, selector: this.#r, tabId: this.#t };
    }
    async click(e = {}) {
      try {
        (await this.#o.send({
          command: hr.create({
            browser_id: this.#e,
            tab_id: this.#t,
            selector: this.#r,
            modifiers: e.modifiers,
            button: e.button,
            force: e.force,
            timeout_ms: e.timeoutMs,
          }),
          timeoutMs: e.timeoutMs,
        }),
          this.#a?.clear());
      } catch (o) {
        throw await this.actionError(
          o,
          "click",
          `waiting on click for selector ${this.#r}`,
        );
      }
    }
    async dblclick(e = {}) {
      try {
        (await this.#o.send({
          command: br.create({
            browser_id: this.#e,
            tab_id: this.#t,
            selector: this.#r,
            modifiers: e.modifiers,
            button: e.button,
            force: e.force,
            timeout_ms: e.timeoutMs,
          }),
          timeoutMs: e.timeoutMs,
        }),
          this.#a?.clear());
      } catch (o) {
        throw await this.actionError(
          o,
          "dblclick",
          `waiting on dblclick for selector ${this.#r}`,
        );
      }
    }
    async selectOption(e, { timeoutMs: o } = {}) {
      let a = Ym(e);
      try {
        (await this.#o.send({
          command: Sr.create({
            browser_id: this.#e,
            tab_id: this.#t,
            selector: this.#r,
            selections: a,
            timeout_ms: o,
          }),
          timeoutMs: o,
        }),
          this.#a?.clear());
      } catch (n) {
        throw await this.actionError(
          n,
          "selectOption",
          `locator.selectOption failed for selector ${this.#r}`,
        );
      }
    }
    async fill(e, { timeoutMs: o } = {}) {
      if (e == null) throw new Error("locator.fill requires a value");
      try {
        (await this.#o.send({
          command: Ue.create({
            browser_id: this.#e,
            tab_id: this.#t,
            selector: this.#r,
            value: e,
            replace: !0,
          }),
          timeoutMs: o,
        }),
          this.#a?.clear());
      } catch (a) {
        throw await this.actionError(
          a,
          "fill",
          `locator.fill failed for selector ${this.#r}`,
        );
      }
    }
    async type(e, { timeoutMs: o } = {}) {
      if (e == null) throw new Error("locator.type requires a value");
      try {
        (await this.#o.send({
          command: Ue.create({
            browser_id: this.#e,
            tab_id: this.#t,
            selector: this.#r,
            value: e,
            replace: !1,
          }),
          timeoutMs: o,
        }),
          this.#a?.clear());
      } catch (a) {
        throw await this.actionError(
          a,
          "type",
          `locator.type failed for selector ${this.#r}`,
        );
      }
    }
    async pressSequentially(e, o = {}) {
      if (e == null)
        throw new Error("locator.pressSequentially requires a value");
      try {
        (await this.#o.send({
          command: xr.create({
            browser_id: this.#e,
            tab_id: this.#t,
            selector: this.#r,
            value: e,
            ...(o.timeoutMs == null ? {} : { timeout_ms: o.timeoutMs }),
          }),
          timeoutMs: o.timeoutMs,
        }),
          this.#a?.clear());
      } catch (a) {
        throw await this.actionError(
          a,
          "pressSequentially",
          `locator.pressSequentially failed for selector ${this.#r}`,
        );
      }
    }
    async press(e, { timeoutMs: o } = {}) {
      if (e == null) throw new Error("locator.press requires a value");
      try {
        (await this.#o.send({
          command: wr.create({
            browser_id: this.#e,
            tab_id: this.#t,
            selector: this.#r,
            value: e,
          }),
          timeoutMs: o,
        }),
          this.#a?.clear());
      } catch (a) {
        throw await this.actionError(
          a,
          "press",
          `locator.press failed for selector ${this.#r}`,
        );
      }
    }
    async setChecked(e, o = {}) {
      if (typeof e != "boolean")
        throw new Error("locator.setChecked requires a boolean");
      try {
        (await this.#o.send({
          command: vr.create({
            browser_id: this.#e,
            tab_id: this.#t,
            selector: this.#r,
            checked: e,
            force: o.force,
            timeout_ms: o.timeoutMs,
          }),
          timeoutMs: o.timeoutMs,
        }),
          this.#a?.clear());
      } catch (a) {
        throw await this.actionError(
          a,
          "setChecked",
          `locator.setChecked(${e}) failed for selector ${this.#r}`,
        );
      }
    }
    async check(e = {}) {
      await this.setChecked(!0, e);
    }
    async uncheck(e = {}) {
      await this.setChecked(!1, e);
    }
    async waitFor({ state: e, timeoutMs: o }) {
      if (!e) throw new Error("locator.waitFor requires a state");
      try {
        await this.#o.send({
          command: _r.create({
            browser_id: this.#e,
            tab_id: this.#t,
            selector: this.#r,
            state: e,
            timeout_ms: o,
          }),
          timeoutMs: o,
        });
      } catch (a) {
        throw await this.actionError(
          a,
          "waitFor",
          `locator.waitFor(${e}) timed out for selector ${this.#r}`,
        );
      }
    }
    async count() {
      return (
        await this.#o.send({
          command: Cr.create({
            browser_id: this.#e,
            tab_id: this.#t,
            selector: this.#r,
          }),
        })
      ).count;
    }
    async all() {
      let e = await this.count(),
        o = new Lo({
          browserId: this.#e,
          tabId: this.#t,
          selector: this.#r,
          transport: this.#o,
        });
      return Array.from(
        { length: e },
        (a, n) =>
          new r({
            browserId: this.#e,
            tabId: this.#t,
            selector: `${this.#r} >> nth=${n}`,
            transport: this.#o,
            collectionReadCache: o,
            collectionReadIndex: n,
          }),
      );
    }
    async textContent({ timeoutMs: e } = {}) {
      let o = await this.cachedRead(e);
      return o
        ? o.text_content
        : (
            await this.#o.send({
              command: Ar.create({
                browser_id: this.#e,
                tab_id: this.#t,
                selector: this.#r,
                timeout_ms: e,
              }),
              timeoutMs: e,
            })
          ).value;
    }
    async innerText({ timeoutMs: e } = {}) {
      let o = await this.cachedRead(e);
      return o
        ? o.inner_text
        : (
            await this.#o.send({
              command: Ir.create({
                browser_id: this.#e,
                tab_id: this.#t,
                selector: this.#r,
                timeout_ms: e,
              }),
              timeoutMs: e,
            })
          ).value;
    }
    async getAttribute(e, { timeoutMs: o } = {}) {
      if (!e) throw new Error("locator.getAttribute requires a name");
      let a = await this.cachedRead(o);
      return a
        ? Object.prototype.hasOwnProperty.call(a.attributes, e)
          ? (a.attributes[e] ?? null)
          : null
        : (
            await this.#o.send({
              command: kr.create({
                browser_id: this.#e,
                tab_id: this.#t,
                selector: this.#r,
                name: e,
                timeout_ms: o,
              }),
              timeoutMs: o,
            })
          ).value;
    }
    async isVisible() {
      return (
        await this.#o.send({
          command: Tr.create({
            browser_id: this.#e,
            tab_id: this.#t,
            selector: this.#r,
          }),
        })
      ).value;
    }
    async isEnabled() {
      return (
        await this.#o.send({
          command: Pr.create({
            browser_id: this.#e,
            tab_id: this.#t,
            selector: this.#r,
          }),
        })
      ).value;
    }
    async allTextContents({ timeoutMs: e } = {}) {
      return (
        await this.#o.send({
          command: Rr.create({
            browser_id: this.#e,
            tab_id: this.#t,
            selector: this.#r,
            timeout_ms: e,
          }),
          timeoutMs: e,
        })
      ).values;
    }
    async evaluate(e, o, a) {
      return (
        await this.#o.send({
          command: Ce.create({
            browser_id: this.#e,
            script: Ti(e, o),
            selector: this.#r,
            tab_id: this.#t,
            timeout_ms: a?.timeoutMs,
          }),
          timeoutMs: a?.timeoutMs,
        })
      ).value;
    }
    async evaluateAll(e, o, a) {
      return (
        await this.#o.send({
          command: Ce.create({
            browser_id: this.#e,
            script: Ti(e, o, "all"),
            selector: this.#r,
            selector_mode: "all",
            tab_id: this.#t,
            timeout_ms: a?.timeoutMs,
          }),
          timeoutMs: a?.timeoutMs,
        })
      ).value;
    }
    async actionError(e, o, a) {
      try {
        let n = await this.evaluateAll(
          (s) => ({
            matchCount: s.length,
            visibleCount: s.reduce(
              (c, l) => c + +(l.getClientRects().length > 0),
              0,
            ),
            matches: s
              .slice(0, 5)
              .map((c) => ({
                tag: c.tagName.toLowerCase(),
                role: c.getAttribute("role"),
                type: c.getAttribute("type"),
                ariaLabel: c.getAttribute("aria-label"),
                text: c.textContent?.trim().slice(0, 120) ?? "",
                visible: c.getClientRects().length > 0,
                disabled: "disabled" in c && !!c.disabled,
              })),
          }),
          void 0,
          { timeoutMs: 1e3 },
        );
        return oo(
          e,
          `${a}
Locator diagnostics: ${JSON.stringify({ kind: Xm(e, n), action: o, locator: this.#r, ...n, truncated: n.matchCount > n.matches.length })}`,
        );
      } catch {
        return oo(e, a);
      }
    }
    async downloadMedia({ timeoutMs: e } = {}) {
      try {
        (await this.#o.send({
          command: gr.create({
            browser_id: this.#e,
            tab_id: this.#t,
            selector: this.#r,
            timeout_ms: e,
          }),
          timeoutMs: e,
        }),
          this.#a?.clear());
      } catch (o) {
        throw await this.actionError(
          o,
          "downloadMedia",
          `locator.downloadMedia failed for selector ${this.#r}`,
        );
      }
    }
    locator(e, o = {}) {
      if (!e) throw new Error("locator.locator requires a selector");
      return new r({
        browserId: this.#e,
        tabId: this.#t,
        selector: `${this.#r} >> ${e}`,
        transport: this.#o,
        collectionReadCache: this.#a,
        collectionReadIndex: this.#n,
        collectionRelativeSelector:
          this.#a && this.#n !== void 0 ? $(this.#s, e) : void 0,
      }).filter(o);
    }
    first() {
      return new r({
        browserId: this.#e,
        tabId: this.#t,
        selector: `${this.#r} >> nth=0`,
        transport: this.#o,
        collectionReadCache: this.#a,
        collectionReadIndex: this.#n,
        collectionRelativeSelector:
          this.#a && this.#n !== void 0 ? $(this.#s, "nth=0") : void 0,
      });
    }
    last() {
      return new r({
        browserId: this.#e,
        tabId: this.#t,
        selector: `${this.#r} >> nth=-1`,
        transport: this.#o,
        collectionReadCache: this.#a,
        collectionReadIndex: this.#n,
        collectionRelativeSelector:
          this.#a && this.#n !== void 0 ? $(this.#s, "nth=-1") : void 0,
      });
    }
    nth(e) {
      if (typeof e != "number")
        throw new Error("locator.nth requires a numeric index");
      return new r({
        browserId: this.#e,
        tabId: this.#t,
        selector: `${this.#r} >> nth=${e}`,
        transport: this.#o,
        collectionReadCache: this.#a,
        collectionReadIndex: this.#n,
        collectionRelativeSelector:
          this.#a && this.#n !== void 0 ? $(this.#s, `nth=${e}`) : void 0,
      });
    }
    and(e) {
      if (!(e instanceof r))
        throw new Error("locator.and requires a PlaywrightLocator");
      if (e.#e !== this.#e || e.#t !== this.#t)
        throw new Error("Locators must belong to the same tab");
      return new r({
        browserId: this.#e,
        tabId: this.#t,
        selector: `${this.#r} >> internal:and=${JSON.stringify(e.#r)}`,
        transport: this.#o,
      });
    }
    or(e) {
      if (!(e instanceof r))
        throw new Error("locator.or requires a PlaywrightLocator");
      if (e.#e !== this.#e || e.#t !== this.#t)
        throw new Error("Locators must belong to the same tab");
      return new r({
        browserId: this.#e,
        tabId: this.#t,
        selector: `${this.#r} >> internal:or=${JSON.stringify(e.#r)}`,
        transport: this.#o,
      });
    }
    filter(e = {}) {
      let o = [this.#r];
      if (
        (e.hasText !== void 0 &&
          o.push(`internal:has-text=${ao(e.hasText, !1)}`),
        e.hasNotText !== void 0 &&
          o.push(`internal:has-not-text=${ao(e.hasNotText, !1)}`),
        e.has !== void 0 &&
          (this.assertCompatibleLocator(e.has, "locator.filter has"),
          o.push(`internal:has=${JSON.stringify(e.has.#r)}`)),
        e.hasNot !== void 0 &&
          (this.assertCompatibleLocator(e.hasNot, "locator.filter hasNot"),
          o.push(`internal:has-not=${JSON.stringify(e.hasNot.#r)}`)),
        e.visible !== void 0)
      ) {
        if (typeof e.visible != "boolean")
          throw new Error("locator.filter visible must be a boolean");
        o.push(`visible=${e.visible}`);
      }
      let a = o.slice(1).join(" >> "),
        n = this.#a && this.#n !== void 0 && (!a || this.#s !== void 0);
      return new r({
        browserId: this.#e,
        tabId: this.#t,
        selector: o.join(" >> "),
        transport: this.#o,
        collectionReadCache: n ? this.#a : void 0,
        collectionReadIndex: n ? this.#n : void 0,
        collectionRelativeSelector: n ? $(this.#s, a) : void 0,
      });
    }
    getByRole(e, o = {}) {
      let a = No(e, o);
      return new r({
        browserId: this.#e,
        tabId: this.#t,
        selector: `${this.#r} >> ${a}`,
        transport: this.#o,
        collectionReadCache: this.#a,
        collectionReadIndex: this.#n,
        collectionRelativeSelector:
          this.#a && this.#n !== void 0 ? $(this.#s, a) : void 0,
      });
    }
    getByText(e, o = {}) {
      let a = $o(e, o);
      return new r({
        browserId: this.#e,
        tabId: this.#t,
        selector: `${this.#r} >> ${a}`,
        transport: this.#o,
        collectionReadCache: this.#a,
        collectionReadIndex: this.#n,
        collectionRelativeSelector:
          this.#a && this.#n !== void 0 ? $(this.#s, a) : void 0,
      });
    }
    getByLabel(e, o = {}) {
      let a = qo(e, o);
      return new r({
        browserId: this.#e,
        tabId: this.#t,
        selector: `${this.#r} >> ${a}`,
        transport: this.#o,
        collectionReadCache: this.#a,
        collectionReadIndex: this.#n,
        collectionRelativeSelector:
          this.#a && this.#n !== void 0 ? $(this.#s, a) : void 0,
      });
    }
    getByPlaceholder(e, o = {}) {
      let a = Uo(e, o);
      return new r({
        browserId: this.#e,
        tabId: this.#t,
        selector: `${this.#r} >> ${a}`,
        transport: this.#o,
        collectionReadCache: this.#a,
        collectionReadIndex: this.#n,
        collectionRelativeSelector:
          this.#a && this.#n !== void 0 ? $(this.#s, a) : void 0,
      });
    }
    getByTestId(e) {
      let o = Wo(e);
      return new r({
        browserId: this.#e,
        tabId: this.#t,
        selector: `${this.#r} >> ${o}`,
        transport: this.#o,
        collectionReadCache: this.#a,
        collectionReadIndex: this.#n,
        collectionRelativeSelector:
          this.#a && this.#n !== void 0 ? $(this.#s, o) : void 0,
      });
    }
    async cachedRead(e) {
      return !this.#a || this.#n === void 0
        ? null
        : await this.#a.read({
            index: this.#n,
            relativeSelector: this.#s,
            timeoutMs: e,
          });
    }
    assertCompatibleLocator(e, o) {
      if (!(e instanceof r))
        throw new Error(`${o} requires a PlaywrightLocator`);
      if (e.#e !== this.#e || e.#t !== this.#t)
        throw new Error("Locators must belong to the same tab");
    }
  },
  Lo = class {
    #e;
    #t;
    #r;
    #o;
    #a = new Map();
    constructor({ browserId: e, tabId: o, selector: a, transport: n }) {
      ((this.#e = e), (this.#t = o), (this.#r = a), (this.#o = n));
    }
    clear() {
      this.#a.clear();
    }
    async read({ index: e, relativeSelector: o, timeoutMs: a }) {
      let n = o ?? "",
        s = this.#a.get(n);
      return (
        s ||
          ((s = this.#n(o, a).catch((c) => {
            throw (this.#a.delete(n), c);
          })),
          this.#a.set(n, s)),
        (await s)[e] ?? null
      );
    }
    async #n(e, o) {
      return (
        await this.#o.send({
          command: jr.create({
            browser_id: this.#e,
            tab_id: this.#t,
            selector: this.#r,
            relative_selector: e,
            timeout_ms: o,
          }),
          timeoutMs: o,
        })
      ).values;
    }
  };
function $(r, e) {
  return r ? (e ? `${r} >> ${e}` : r) : e || void 0;
}
function Ii(r) {
  let e = r;
  return r.unicode || e.unicodeSets
    ? String(r)
    : String(r)
        .replace(/(^|[^\\])(\\\\)*(["'`])/g, "$1$2\\\\$3")
        .replace(/>>/g, "\\>\\>");
}
function ao(r, e) {
  return typeof r != "string" ? Ii(r) : `${JSON.stringify(r)}${e ? "s" : "i"}`;
}
function Fo(r, e) {
  return typeof r != "string"
    ? Ii(r)
    : `"${r.replace(/\\/g, "\\\\").replace(/["]/g, '\\"')}"${e ? "s" : "i"}`;
}
function No(r, e) {
  if (!r) throw new Error("getByRole requires a role");
  let o = [];
  return (
    e.name !== void 0 && o.push(`[name=${Fo(e.name, !!e.exact)}]`),
    `internal:role=${r}${o.join("")}`
  );
}
function $o(r, e) {
  if (typeof r != "string" && !He(r))
    throw new Error("getByText requires a string or RegExp");
  return `internal:text=${ao(r, !!e.exact)}`;
}
function qo(r, e) {
  if (typeof r != "string" && !He(r))
    throw new Error("getByLabel requires a string or RegExp");
  return `internal:label=${ao(r, !!e.exact)}`;
}
function Uo(r, e) {
  if (typeof r != "string" && !He(r))
    throw new Error("getByPlaceholder requires a string or RegExp");
  return `internal:attr=[placeholder=${Fo(r, !!e.exact)}]`;
}
function Wo(r) {
  if (!r) throw new Error("getByTestId requires a testId");
  return `internal:testid=[data-testid=${Fo(r, !0)}]`;
}
var qr = class r {
    #e;
    #t;
    #r;
    #o;
    constructor({ browserId: e, tabId: o, frameSelector: a, transport: n }) {
      ((this.#e = e), (this.#t = o), (this.#r = a), (this.#o = n));
    }
    locator(e) {
      if (!e) throw new Error("frameLocator.locator requires a selector");
      let o = `${this.#r} >> internal:control=enter-frame >> ${e}`;
      return new H({
        browserId: this.#e,
        tabId: this.#t,
        selector: o,
        transport: this.#o,
      });
    }
    frameLocator(e) {
      if (!e) throw new Error("frameLocator.frameLocator requires a selector");
      return new r({
        browserId: this.#e,
        tabId: this.#t,
        frameSelector: `${this.#r} >> internal:control=enter-frame >> ${e}`,
        transport: this.#o,
      });
    }
    getByRole(e, o = {}) {
      return this.locator(No(e, o));
    }
    getByText(e, o = {}) {
      return this.locator($o(e, o));
    }
    getByLabel(e, o = {}) {
      return this.locator(qo(e, o));
    }
    getByPlaceholder(e, o = {}) {
      return this.locator(Uo(e, o));
    }
    getByTestId(e) {
      return this.locator(Wo(e));
    }
  },
  Ur = class {
    #e;
    #t;
    #r;
    #o;
    constructor({ browserId: e, tabId: o, downloadId: a, transport: n }) {
      ((this.#e = e), (this.#t = o), (this.#r = a), (this.#o = n));
    }
    async path({ timeoutMs: e } = {}) {
      return (
        (
          await this.#o.send({
            command: Fr.create({
              browser_id: this.#e,
              tab_id: this.#t,
              download_id: this.#r,
              timeout_ms: e,
            }),
            timeoutMs: e,
          })
        ).path ?? null
      );
    }
  },
  Wr = class {
    #e;
    #t;
    #r;
    #o;
    #a;
    constructor({
      browserId: e,
      tabId: o,
      fileChooserId: a,
      isMultiple: n,
      transport: s,
    }) {
      ((this.#e = e),
        (this.#t = o),
        (this.#r = a),
        (this.#o = n),
        (this.#a = s));
    }
    isMultiple() {
      return this.#o;
    }
    async setFiles(e, { timeoutMs: o } = {}) {
      if (e == null) throw new Error("fileChooser.setFiles requires files");
      let a = Array.isArray(e) ? e : [e];
      if (!a.length)
        throw new Error("fileChooser.setFiles requires at least one file");
      try {
        await this.#a.send({
          command: $r.create({
            browser_id: this.#e,
            tab_id: this.#t,
            file_chooser_id: this.#r,
            files: a,
            timeout_ms: o,
          }),
          timeoutMs: o,
        });
      } catch (n) {
        throw oo(n, "fileChooser.setFiles failed");
      }
    }
  },
  Xe = class {
    #e;
    #t;
    #r;
    constructor({ browserId: e, tabId: o, transport: a }) {
      ((this.#e = e), (this.#t = o), (this.#r = a));
    }
    async goBack() {
      await this.#r.send({
        command: we.create({ browser_id: this.#e, tab_id: this.#t }),
      });
    }
    async goForward() {
      await this.#r.send({
        command: xe.create({ browser_id: this.#e, tab_id: this.#t }),
      });
    }
    async evaluate(e, o, a) {
      return (
        await this.#r.send({
          command: Ce.create({
            browser_id: this.#e,
            script: Km(e, o),
            tab_id: this.#t,
            timeout_ms: a?.timeoutMs,
          }),
          timeoutMs: a?.timeoutMs,
        })
      ).value;
    }
    locator(e) {
      if (!e) throw new Error("playwright.locator requires a selector");
      return new H({
        browserId: this.#e,
        tabId: this.#t,
        selector: e,
        transport: this.#r,
      });
    }
    getByRole(e, o = {}) {
      return this.locator(No(e, o));
    }
    getByText(e, o = {}) {
      return this.locator($o(e, o));
    }
    getByLabel(e, o = {}) {
      return this.locator(qo(e, o));
    }
    getByPlaceholder(e, o = {}) {
      return this.locator(Uo(e, o));
    }
    getByTestId(e) {
      return this.locator(Wo(e));
    }
    frameLocator(e) {
      if (!e) throw new Error("playwright.frameLocator requires a selector");
      return new qr({
        browserId: this.#e,
        tabId: this.#t,
        frameSelector: e,
        transport: this.#r,
      });
    }
    async waitForURL(e, o = {}) {
      if (!e) throw new Error("playwright.waitForURL requires a url");
      await this.#r.send({
        command: Or.create({
          browser_id: this.#e,
          tab_id: this.#t,
          url: e,
          wait_until: o.waitUntil,
          timeout_ms: o.timeoutMs,
        }),
        timeoutMs: o.timeoutMs,
      });
    }
    async waitForLoadState(e = {}) {
      await this.#r.send({
        command: Br.create({
          browser_id: this.#e,
          tab_id: this.#t,
          state: e.state,
          timeout_ms: e.timeoutMs,
        }),
        timeoutMs: e.timeoutMs,
      });
    }
    async waitForTimeout(e) {
      if (!Number.isInteger(e) || e < 0)
        throw new Error(
          "playwright.waitForTimeout requires a non-negative integer",
        );
      await this.#r.send({
        command: zr.create({
          browser_id: this.#e,
          tab_id: this.#t,
          timeout_ms: e,
        }),
        timeoutMs: e + 2e3,
      });
    }
    async waitForEvent(e, o = {}) {
      if (e === "download") {
        let a = await this.#r.send({
          command: Lr.create({
            browser_id: this.#e,
            tab_id: this.#t,
            timeout_ms: o.timeoutMs,
          }),
          timeoutMs: o.timeoutMs,
        });
        return new Ur({
          browserId: this.#e,
          tabId: this.#t,
          downloadId: a.download_id,
          transport: this.#r,
        });
      }
      if (e === "filechooser") {
        let a = await this.#r.send({
          command: Nr.create({
            browser_id: this.#e,
            tab_id: this.#t,
            timeout_ms: o.timeoutMs,
          }),
          timeoutMs: o.timeoutMs,
        });
        return new Wr({
          browserId: this.#e,
          tabId: this.#t,
          fileChooserId: a.file_chooser_id,
          isMultiple: a.is_multiple,
          transport: this.#r,
        });
      }
      throw new Error(
        "playwright.waitForEvent only supports 'download' and 'filechooser'",
      );
    }
    async expectNavigation(e, o = {}) {
      let a = o.url
          ? this.waitForURL(o.url, {
              timeoutMs: o.timeoutMs,
              waitUntil: o.waitUntil,
            })
          : this.waitForLoadState({
              timeoutMs: o.timeoutMs,
              state: o.waitUntil,
            }),
        n = e(),
        [s] = await Promise.all([n, a]);
      return s;
    }
    async elementInfo(e) {
      let o = Pi("playwright.elementInfo", e);
      return await this.#r.send({
        command: Dr.create({
          browser_id: this.#e,
          tab_id: this.#t,
          x: o.x,
          y: o.y,
          include_non_interactable: e.includeNonInteractable,
        }),
      });
    }
    async elementScreenshot(e) {
      let o = Pi("playwright.elementScreenshot", e),
        a = await this.#r.send({
          command: Mr.create({
            browser_id: this.#e,
            tab_id: this.#t,
            x: o.x,
            y: o.y,
            include_non_interactable: e.includeNonInteractable,
          }),
        });
      return Je(a.data);
    }
    async domSnapshot() {
      return (
        await this.#r.send({
          command: Er.create({ browser_id: this.#e, tab_id: this.#t }),
        })
      ).dom_snapshot;
    }
  };
var Ke = class {
  #e;
  #t;
  #r;
  constructor({ browserId: e, tabId: o, transport: a }) {
    ((this.#e = e), (this.#t = o), (this.#r = a));
  }
  async get_visible_dom() {
    return this.#r.send({
      command: mr.create({ browser_id: this.#e, tab_id: this.#t }),
    });
  }
  async click(e) {
    let o = Vo(e?.node_id, "dom_cua.click");
    await this.#r.send({
      command: qe.create({ browser_id: this.#e, tab_id: this.#t, node_id: o }),
    });
  }
  async double_click(e) {
    let o = Vo(e?.node_id, "dom_cua.double_click");
    await this.#r.send({
      command: pr.create({ browser_id: this.#e, tab_id: this.#t, node_id: o }),
    });
  }
  async scroll({ node_id: e, x: o, y: a }) {
    if (typeof o != "number" || typeof a != "number")
      throw new Error("dom_cua.scroll requires x and y numbers");
    let n = e === void 0 ? void 0 : ki(e, "dom_cua.scroll");
    await this.#r.send({
      command: fr.create({
        browser_id: this.#e,
        tab_id: this.#t,
        node_id: n,
        scroll_x: o,
        scroll_y: a,
      }),
    });
  }
  async type({ text: e }) {
    if (typeof e != "string") throw new Error("dom_cua.type requires text");
    await this.#r.send({
      command: yr.create({ browser_id: this.#e, tab_id: this.#t, text: e }),
    });
  }
  async keypress(e) {
    if (!Array.isArray(e?.keys) || e.keys.length === 0)
      throw new Error("dom_cua.keypress requires a non-empty keys array");
    await this.#r.send({
      command: ur.create({
        browser_id: this.#e,
        tab_id: this.#t,
        keys: e.keys,
      }),
    });
  }
  async downloadMedia(e) {
    let o = Vo(e?.node_id, "dom_cua.downloadMedia");
    await this.#r.send({
      command: dr.create({
        browser_id: this.#e,
        tab_id: this.#t,
        node_id: o,
        timeout_ms: e.timeoutMs,
      }),
      timeoutMs: e.timeoutMs,
    });
  }
};
function Vo(r, e) {
  if (r === void 0) throw new Error(`${e} requires a node_id`);
  return ki(r, e);
}
function ki(r, e) {
  let o = qe.DomNodeIdSchema.safeParse(r);
  if (!o.success)
    throw new Error(
      `${e} ${o.error.issues[0]?.message ?? "has an invalid node_id"}`,
    );
  return o.data;
}
var Ye = class {
  #e;
  #t;
  #r;
  constructor({ browserId: e, tabId: o, transport: a }) {
    ((this.#e = e), (this.#t = o), (this.#r = a));
  }
  async click(e) {
    if (typeof e?.x != "number" || typeof e?.y != "number")
      throw new Error("cua.click requires x and y");
    await this.#r.send({
      command: rr.create({
        browser_id: this.#e,
        tab_id: this.#t,
        x: e.x,
        y: e.y,
        button: e.button,
        keys: e.keypress,
      }),
    });
  }
  async scroll(e) {
    if (
      typeof e?.x != "number" ||
      typeof e?.y != "number" ||
      typeof e?.scrollX != "number" ||
      typeof e?.scrollY != "number"
    )
      throw new Error("cua.scroll requires x, y, scrollX, and scrollY");
    await this.#r.send({
      command: cr.create({
        browser_id: this.#e,
        tab_id: this.#t,
        x: e.x,
        y: e.y,
        scroll_x: e.scrollX,
        scroll_y: e.scrollY,
        keys: e.keypress,
      }),
    });
  }
  async double_click(e) {
    if (typeof e?.x != "number" || typeof e?.y != "number")
      throw new Error("cua.double_click requires x and y");
    await this.#r.send({
      command: ar.create({
        browser_id: this.#e,
        tab_id: this.#t,
        x: e.x,
        y: e.y,
        keys: e.keypress,
      }),
    });
  }
  async type(e) {
    if (typeof e?.text != "string") throw new Error("cua.type requires text");
    await this.#r.send({
      command: lr.create({
        browser_id: this.#e,
        tab_id: this.#t,
        text: e.text,
      }),
    });
  }
  async keypress(e) {
    if (!Array.isArray(e?.keys) || e.keys.length === 0)
      throw new Error("cua.keypress requires a non-empty keys array");
    await this.#r.send({
      command: nr.create({
        browser_id: this.#e,
        tab_id: this.#t,
        keys: e.keys,
      }),
    });
  }
  async drag(e) {
    if (
      !Array.isArray(e?.path) ||
      e.path.length === 0 ||
      e.path.some((o) => typeof o?.x != "number" || typeof o?.y != "number")
    )
      throw new Error("cua.drag requires a non-empty path of {x, y} points");
    await this.#r.send({
      command: sr.create({
        browser_id: this.#e,
        tab_id: this.#t,
        path: e.path,
        keys: e.keys,
      }),
    });
  }
  async move(e) {
    if (typeof e?.x != "number" || typeof e?.y != "number")
      throw new Error("cua.move requires x and y");
    await this.#r.send({
      command: ir.create({
        browser_id: this.#e,
        tab_id: this.#t,
        x: e.x,
        y: e.y,
        keys: e.keys,
      }),
    });
  }
  async downloadMedia(e) {
    if (typeof e?.x != "number" || typeof e?.y != "number")
      throw new Error("cua.downloadMedia requires x and y");
    await this.#r.send({
      command: or.create({
        browser_id: this.#e,
        tab_id: this.#t,
        x: e.x,
        y: e.y,
        timeout_ms: e.timeoutMs,
      }),
      timeoutMs: e.timeoutMs,
    });
  }
};
var Qe = class {
  #e;
  #t;
  #r;
  constructor({ browserId: e, tabId: o, transport: a }) {
    ((this.#e = e), (this.#t = o), (this.#r = a));
  }
  async get(e = "state", o) {
    let a = await this.#o(e, o);
    switch (a.mode) {
      case "state":
        return a.state;
      case "screenshot":
        return a.screenshot;
      case "both":
        return a.screenshot === void 0
          ? { state: a.state }
          : { state: a.state, screenshot: a.screenshot };
    }
  }
  async write(e = "state", o) {
    let a = await this.#o(e, o);
    (a.mode !== "screenshot" && (await this.#r.display(a.state)),
      a.mode !== "state" &&
        a.screenshot !== void 0 &&
        (await this.#r.display(a.screenshot)));
  }
  async click(e, o) {
    await this.#a({
      kind: "click",
      target: e,
      ...(o?.mouseButton === void 0 ? {} : { mouse_button: o.mouseButton }),
      ...(o?.clickCount === void 0 ? {} : { click_count: o.clickCount }),
    });
  }
  async drag(e, o) {
    await this.#a({ kind: "drag", from: e, to: o });
  }
  async performSecondaryAction(e, o) {
    await this.#a({
      kind: "perform_secondary_action",
      element_index: e,
      action: o,
    });
  }
  async pressKey(e) {
    await this.#a({ kind: "press_key", key: e });
  }
  async scroll(e, o, a) {
    await this.#a({
      kind: "scroll",
      target: e,
      direction: o,
      ...(a === void 0 ? {} : { pages: a }),
    });
  }
  async selectText(e, o, a) {
    await this.#a({
      kind: "select_text",
      element_index: e,
      text: o,
      ...(a?.prefix === void 0 ? {} : { prefix: a.prefix }),
      ...(a?.suffix === void 0 ? {} : { suffix: a.suffix }),
      ...(a?.selectionType === void 0
        ? {}
        : { selection_type: a.selectionType }),
    });
  }
  async setValue(e, o) {
    await this.#a({ kind: "set_value", element_index: e, value: o });
  }
  async typeText(e) {
    await this.#a({ kind: "type_text", text: e });
  }
  async #o(e, o) {
    let a = await this.#r.send({
      command: er.create({
        browser_id: this.#e,
        tab_id: this.#t,
        content:
          e === "state"
            ? "axState"
            : e === "screenshot"
              ? "screenshot"
              : "axStateAndScreenshot",
        ...(o?.disableDiffing === void 0
          ? {}
          : { disable_diffing: o.disableDiffing }),
      }),
    });
    if (e === "state") {
      if (typeof a.state != "string")
        throw new Error("ax capture returned no accessibility state");
      return { mode: e, state: a.state };
    }
    if (typeof a.data != "string") {
      if (
        e === "both" &&
        typeof a.screenshot_unavailable == "string" &&
        typeof a.state == "string"
      )
        return { mode: e, state: a.state };
      throw new Error("ax capture returned no screenshot data");
    }
    let n = Je(a.data);
    if (e === "screenshot") return { mode: e, screenshot: n };
    if (typeof a.state != "string")
      throw new Error("ax capture returned no accessibility state");
    return { mode: e, state: a.state, screenshot: n };
  }
  async #a(e) {
    await this.#r.send({
      command: tr.create({ browser_id: this.#e, tab_id: this.#t, action: e }),
    });
  }
};
var et = class {
  #e;
  #t;
  #r;
  constructor({ browserId: e, tabId: o, transport: a }) {
    ((this.#e = e), (this.#t = o), (this.#r = a));
  }
  async export() {
    return (
      await this.#r.send({
        command: Zt.create({ browser_id: this.#e, tab_id: this.#t }),
      })
    ).path;
  }
  async exportGsuite(e) {
    return (
      await this.#r.send({
        command: Gt.create({ browser_id: this.#e, tab_id: this.#t, format: e }),
      })
    ).path;
  }
  async exportYouTubeTranscript() {
    return (
      await this.#r.send({
        command: Ht.create({ browser_id: this.#e, tab_id: this.#t }),
      })
    ).path;
  }
};
var tt = class {
  #e;
  #t;
  #r;
  constructor({ browserId: e, tabId: o, transport: a }) {
    ((this.#e = e), (this.#t = o), (this.#r = a));
  }
  async readText() {
    return (
      await this.#r.send({
        command: Jt.create({ browser_id: this.#e, tab_id: this.#t }),
      })
    ).text;
  }
  async writeText(e) {
    if (e == null) throw new Error("tab.clipboard.writeText requires text");
    await this.#r.send({
      command: Xt.create({ browser_id: this.#e, tab_id: this.#t, text: e }),
    });
  }
  async read() {
    return (
      await this.#r.send({
        command: Kt.create({ browser_id: this.#e, tab_id: this.#t }),
      })
    ).items.map((o) => ({
      entries: o.entries.map((a) => ({
        mimeType: a.mime_type,
        text: a.text,
        base64: a.base64,
      })),
      presentationStyle: o.presentation_style,
    }));
  }
  async write(e) {
    if (!Array.isArray(e) || e.length === 0)
      throw new Error(
        "tab.clipboard.write requires at least one clipboard item",
      );
    await this.#r.send({
      command: Yt.create({
        browser_id: this.#e,
        tab_id: this.#t,
        items: e.map((o) => ({
          presentation_style: o.presentationStyle,
          entries: o.entries.map((a) => ({
            mime_type: a.mimeType,
            text: a.text,
            base64: a.base64,
          })),
        })),
      }),
    });
  }
};
var rt = class {
    #e;
    #t;
    #r;
    constructor({ browserId: e, tabId: o, transport: a }) {
      ((this.#e = e), (this.#t = o), (this.#r = a));
    }
    async logs(e = {}) {
      let o = Qm(e.filter),
        a = eu(e.levels),
        n = tu(e.limit),
        s = {
          browser_id: this.#e,
          tab_id: this.#t,
          ...(o !== void 0 ? { filter: o } : {}),
          ...(a !== void 0 ? { levels: a } : {}),
          ...(n !== void 0 ? { limit: n } : {}),
        };
      return (await this.#r.send({ command: $e.create(s) })).logs;
    }
  },
  Qm = (r) => {
    if (r != null) {
      if (typeof r != "string")
        throw new Error("tab.dev.logs received an invalid filter");
      return r;
    }
  },
  eu = (r) => {
    if (r != null) {
      if (!Array.isArray(r) || r.length === 0)
        throw new Error("tab.dev.logs received invalid levels");
      return r.map((e) => {
        if (e === "warning") return "warn";
        if ($e.LogLevelSchema.options.includes(e)) return e;
        throw new Error(`tab.dev.logs received invalid level "${String(e)}"`);
      });
    }
  },
  tu = (r) => {
    if (r != null) {
      if (!Number.isInteger(r) || r <= 0)
        throw new Error("tab.dev.logs received an invalid limit");
      return r;
    }
  };
var ee = class {
    constructor(e, o, a, n) {
      this.transport = e;
      this.browserId = o;
      this.documentationApi = a;
      this.info = n;
    }
    get id() {
      return this.info.id;
    }
    async documentation() {
      return await this.documentationApi.get(`capabilities/browser/${this.id}`);
    }
  },
  B = class {
    constructor(e, o, a, n, s) {
      this.transport = e;
      this.browserId = o;
      this.tabId = a;
      this.documentationApi = n;
      this.info = s;
    }
    get id() {
      return this.info.id;
    }
    async documentation() {
      return await this.documentationApi.get(`capabilities/tab/${this.id}`);
    }
  };
function at({ capability: r, info: e, internalOnly: o }) {
  return {
    capability: r,
    create: (a) => new r({ ...a, info: e }),
    id: e.id,
    info: e,
    ...(o == null ? {} : { internalOnly: o }),
  };
}
function q({ capability: r, info: e, internalOnly: o }) {
  return {
    capability: r,
    create: (a) => new r({ ...a, info: e }),
    id: e.id,
    info: e,
    ...(o == null ? {} : { internalOnly: o }),
  };
}
var ot = class {
  #e;
  constructor(e) {
    this.#e = e;
  }
  async get(e) {
    let o = this.#e[e];
    if (!o) throw new Error(`Capability is not available: ${e}`);
    return o;
  }
  async list() {
    return Object.values(this.#e).map((e) => e.info);
  }
};
var no = {};
d(no, { Report: () => Se });
var Se = {};
d(Se, {
  PayloadSchema: () => Bi,
  ResultSchema: () => ou,
  commandType: () => Oi,
  create: () => au,
});
var ru = t.enum([
    "captcha_failed",
    "access_denied",
    "challenge_loop",
    "unexpected_bot_error",
  ]),
  ji = t.object({ reason: ru });
var Oi = "tab_bot_detection_report",
  Bi = ji.extend({ browser_id: t.string(), tab_id: t.string() }),
  ou = t.object({
    status: t.literal("reported"),
    hostname: t.string().nullable(),
  });
function au(r) {
  return new i(Oi, Bi, r);
}
var zi = {
  id: "botDetection",
  description:
    "Use when the current tab is blocked by bot detection, a failed CAPTCHA, a hard access denial, or a repeated challenge/login loop.",
};
var nt = class extends B {
    constructor({
      browserId: e,
      documentation: o,
      info: a,
      tabId: n,
      transport: s,
    }) {
      super(s, e, n, o, a);
    }
    async report(e) {
      let o = Se.PayloadSchema.parse({
          ...e,
          browser_id: this.browserId,
          tab_id: this.tabId,
        }),
        a = await this.transport.send({ command: Se.create(o) });
      return Se.ResultSchema.parse(a);
    }
  },
  so = q({ capability: nt, info: zi });
var co = {};
d(co, { Call: () => st, Events: () => it });
var st = {};
d(st, {
  PayloadSchema: () => Mi,
  ResultSchema: () => su,
  commandType: () => Di,
  create: () => iu,
});
var io = t
    .object({
      session_id: t.string().min(1).optional(),
      target_id: t.string().min(1).optional(),
    })
    .refine(
      (r) => (r.session_id == null) != (r.target_id == null),
      "CDP target must provide exactly one of session_id or target_id.",
    ),
  nu = t.object({
    extensionId: t.string().optional(),
    sessionId: t.string().optional(),
    tabId: t.number().int().optional(),
    targetId: t.string().optional(),
  }),
  Ei = t.object({
    sequence: t.number().int().positive(),
    source: nu,
    method: t.string().min(1),
    params: t.record(t.string(), t.unknown()).optional(),
  });
var Di = "tab_cdp_call",
  Mi = t.object({
    browser_id: t.string(),
    tab_id: t.string(),
    method: t.string().min(1),
    params: t.record(t.string(), t.unknown()).optional(),
    target: io.optional(),
    timeout_ms: t.number().int().positive().optional(),
  }),
  su = t.unknown();
function iu(r) {
  return new i(Di, Mi, r);
}
var it = {};
d(it, {
  PayloadSchema: () => Fi,
  ResultSchema: () => cu,
  commandType: () => Li,
  create: () => lu,
});
var Li = "tab_cdp_events",
  Fi = t.object({
    after_sequence: t.number().int().nonnegative().optional(),
    browser_id: t.string(),
    limit: t.number().int().positive().max(1e3).optional(),
    methods: t.array(t.string().min(1)).min(1).optional(),
    tab_id: t.string(),
    target: io.optional(),
    timeout_ms: t.number().int().nonnegative().optional(),
  }),
  cu = t.object({
    cursor: t.number().int().nonnegative(),
    events: t.array(Ei),
    hasMore: t.boolean(),
    truncated: t.boolean(),
  });
function lu(r) {
  return new i(Li, Fi, r);
}
var Ni = {
  id: "cdp",
  description:
    "Send raw Chrome DevTools Protocol commands and read debugger events through a supported tab for developer use cases.",
};
var du = 2e3,
  ct = class extends B {
    constructor({
      browserId: e,
      documentation: o,
      info: a,
      tabId: n,
      transport: s,
    }) {
      super(s, e, n, o, a);
    }
    async send(e, o, a) {
      let n = await this.transport.send({
        command: st.create({
          browser_id: this.browserId,
          method: e,
          params: o,
          tab_id: this.tabId,
          target: $i(a?.target),
          timeout_ms: a?.timeoutMs,
        }),
        timeoutMs: a?.timeoutMs,
      });
      return st.ResultSchema.parse(n);
    }
    async readEvents(e) {
      let o = await this.transport.send({
        command: it.create({
          after_sequence: e?.afterSequence,
          browser_id: this.browserId,
          limit: e?.limit,
          methods: e?.methods,
          tab_id: this.tabId,
          target: $i(e?.target),
          timeout_ms: e?.timeoutMs,
        }),
        timeoutMs: pu(e?.timeoutMs),
      });
      return it.ResultSchema.parse(o);
    }
  },
  lo = q({ capability: ct, info: Ni });
function $i(r) {
  if (r != null) return { session_id: r.sessionId, target_id: r.targetId };
}
function pu(r) {
  return r != null && r > 0 ? r + du : void 0;
}
var po = {};
d(po, { Handoff: () => lt });
var lt = {};
d(lt, {
  PayloadSchema: () => Zi,
  ResultSchema: () => yu,
  commandType: () => Vi,
  create: () => hu,
});
var qi = /^[A-Za-z0-9_-]{1,48}$/,
  Ui = new Set(["__proto__", "constructor", "prototype"]),
  mu = t.object({
    id: t
      .string()
      .regex(qi)
      .refine((r) => !Ui.has(r)),
    label: t.string(),
    type: t.string().min(1),
    autocomplete: t.string().nullable().optional(),
    required: t.boolean(),
    selector: t.string(),
  }),
  uu = t.object({
    selector: t.string(),
    action: t.enum(["click", "press_enter"]),
  }),
  fu = t
    .object({
      id: t
        .string()
        .regex(qi)
        .refine((r) => !Ui.has(r)),
      label: t.string().trim().min(1).max(120),
      selector: t.string().optional(),
      field_ids: t.array(t.string()).max(6).optional(),
    })
    .refine(
      (r) => (r.selector != null) != (r.field_ids?.length ?? 0) > 0,
      "browser auth options require either a selector or credential fields",
    ),
  Wi = t
    .object({
      origin: t.string(),
      fields: t.array(mu).max(6),
      options: t.array(fu).min(2).max(10).optional(),
      qr_code: t.literal(!0).optional(),
      submit: uu.optional(),
    })
    .refine(
      (r) => r.fields.length > 0 || r.options != null || r.qr_code === !0,
      "browser auth requests require credential fields, sign-in options, or a QR code",
    );
var Vi = "tab_browser_auth_handoff",
  Zi = Wi.and(t.object({ browser_id: t.string(), tab_id: t.string() })),
  yu = t.object({
    status: t.enum([
      "submitted",
      "declined",
      "cancelled",
      "unavailable",
      "expired",
      "origin_changed",
      "page_changed",
      "locator_invalid",
      "submission_failed",
    ]),
    locator_error: t
      .object({ field_id: t.string(), reason: t.literal("not_user_visible") })
      .optional(),
    selected_option: t.string().optional(),
    reason: t.literal("user_took_over").optional(),
  });
function hu(r) {
  return new i(Vi, Zi, r);
}
var Gi = {
  id: "browserAuth",
  description:
    "MUST read its documentation before the first interaction with any authentication or sign-in flow required for the task. Covers method selection, secure user-provided credentials, and post-authentication verification and recovery.",
};
var dt = class extends B {
    constructor({
      browserId: e,
      documentation: o,
      info: a,
      tabId: n,
      transport: s,
    }) {
      super(s, e, n, o, a);
    }
    async request(e) {
      let o = bu(e, { browserId: this.browserId, tabId: this.tabId }),
        a = lt.create({ ...o, browser_id: this.browserId, tab_id: this.tabId });
      a.parse();
      let n = await this.transport.send({ command: a });
      return lt.ResultSchema.parse(n);
    }
  },
  mo = q({ capability: dt, info: Gi });
function bu(r, e) {
  let o = {
    origin: r.origin,
    fields: r.fields.map((a) => {
      let n = {
        id: a.id,
        label: a.label,
        type: a.type,
        required: a.required,
        selector: Zo(a.selector, e),
      };
      return (
        a.autocomplete !== void 0 && (n.autocomplete = a.autocomplete),
        n
      );
    }),
    ...(r.qr_code === !0 ? { qr_code: !0 } : {}),
  };
  return (
    r.options != null &&
      (o.options = r.options.map((a) => ({
        id: a.id,
        label: a.label,
        ...(a.field_ids == null ? {} : { field_ids: a.field_ids }),
        ...(a.selector == null ? {} : { selector: Zo(a.selector, e) }),
      }))),
    r.submit != null &&
      (o.submit = {
        action: r.submit.action,
        selector: Zo(r.submit.selector, e),
      }),
    o
  );
}
function Zo(r, e) {
  if (typeof r == "string") return r;
  if (!(r instanceof H))
    throw new Error(
      "browserAuth selector must be a string or PlaywrightLocator",
    );
  let o = H.browserAuthSelector(r);
  if (o.browserId !== e.browserId || o.tabId !== e.tabId)
    throw new Error("browserAuth selector locator must belong to this tab");
  return o.selector;
}
var uo = {};
d(uo, { Bundle: () => pt, List: () => mt });
var pt = {};
d(pt, {
  PayloadSchema: () => ec,
  ResultSchema: () => Su,
  commandType: () => Qi,
  create: () => vu,
});
var Hi = t.enum(["font", "image", "script", "stylesheet", "video", "other"]),
  Ji = t.enum(["font", "image", "stylesheet", "video"]),
  gu = t.object({
    kind: t.enum(["attribute", "computedStyle", "resource"]),
    nodeId: t.number().int().positive().optional(),
    property: t.string().optional(),
  }),
  wu = t.object({
    id: t.string(),
    kind: Hi,
    name: t.string(),
    sources: t.array(gu),
    url: t.string(),
  }),
  xu = t.object({ id: t.string(), markup: t.string(), name: t.string() }),
  Xi = t.object({
    assets: t.array(wu),
    id: t.string(),
    inlineSvgs: t.array(xu),
    pageUrl: t.string().nullable(),
    summary: t.object({
      byKind: t.record(Hi, t.number().int().nonnegative()),
      inlineSvgCount: t.number().int().nonnegative(),
      totalCount: t.number().int().nonnegative(),
    }),
  }),
  Ki = t.object({
    assetIds: t.array(t.string()).optional(),
    inventoryId: t.string(),
    kinds: t.array(Ji).optional(),
  }),
  _u = t.object({
    contentType: t.string().nullable(),
    id: t.string(),
    kind: Ji,
    name: t.string(),
    path: t.string(),
    url: t.string(),
  }),
  Cu = t.object({
    contentType: t.string().nullable(),
    id: t.string(),
    name: t.string(),
    reason: t.string(),
    url: t.string(),
  }),
  Yi = t.object({
    assets: t.array(_u),
    directoryPath: t.string(),
    failures: t.array(Cu),
    manifestPath: t.string(),
    summary: t.object({
      downloadedCount: t.number().int().nonnegative(),
      elapsedMs: t.number().nonnegative(),
      failedCount: t.number().int().nonnegative(),
      requestedCount: t.number().int().nonnegative(),
    }),
  });
var Qi = "tab_page_assets_bundle",
  ec = Ki.extend({ browser_id: t.string(), tab_id: t.string() }),
  Su = Yi;
function vu(r) {
  return new i(Qi, ec, r);
}
var mt = {};
d(mt, {
  PayloadSchema: () => rc,
  ResultSchema: () => Tu,
  commandType: () => tc,
  create: () => Pu,
});
var tc = "tab_page_assets_list",
  rc = t.object({ browser_id: t.string(), tab_id: t.string() }),
  Tu = Xi;
function Pu(r) {
  return new i(tc, rc, r);
}
var oc = {
  id: "pageAssets",
  description:
    "List assets already observed in the current page state and bundle selected assets into a temporary local artifact.",
};
var ut = class extends B {
    constructor({
      browserId: e,
      documentation: o,
      info: a,
      tabId: n,
      transport: s,
    }) {
      super(s, e, n, o, a);
    }
    async list() {
      let e = await this.transport.send({
        command: mt.create({ browser_id: this.browserId, tab_id: this.tabId }),
      });
      return mt.ResultSchema.parse(e);
    }
    async bundle(e) {
      let o = await this.transport.send({
        command: pt.create({
          ...e,
          browser_id: this.browserId,
          tab_id: this.tabId,
        }),
      });
      return pt.ResultSchema.parse(o);
    }
  },
  fo = q({ capability: ut, info: oc });
var yo = {
  id: "webmcp",
  description:
    "Fetch page-defined WebMCP tools bound to the current document, then call them through the returned object.",
};
function ho(r) {
  return {
    name: r.name,
    ...(r.title == null ? {} : { title: r.title }),
    ...(r.description == null ? {} : { description: r.description }),
    inputSchema: r.input_schema,
    ...(r.annotations == null ? {} : { annotations: r.annotations }),
    ...(r.origin == null ? {} : { origin: r.origin }),
    ...(r.pageUrl == null ? {} : { pageUrl: r.pageUrl }),
  };
}
var ft = class extends B {
  constructor({
    browserId: e,
    documentation: o,
    info: a,
    tabId: n,
    transport: s,
  }) {
    super(s, e, n, o, a);
  }
  async fetchTools() {
    let e = Ne.ResultSchema.parse(
        await this.transport.send({
          command: Ne.create({
            browser_id: this.browserId,
            tab_id: this.tabId,
          }),
        }),
      ),
      o = e.tools.map(ho),
      a = new Map(e.tools.map((s) => [s.name, s])),
      n = Ru(o);
    return Object.freeze({
      description: () => n,
      call: async (s, c, l) => {
        let u = s.trim(),
          p = a.get(u);
        if (p == null)
          throw new Error(
            `WebMCP tool ${JSON.stringify(u)} is not available in this snapshot. Call fetchTools() again.`,
          );
        return (
          await this.transport.send({
            command: Ft.create({
              browser_id: this.browserId,
              tab_id: this.tabId,
              tool_name: u,
              tool_description: p.description,
              tool_title: p.title,
              registration_id: p.registration_id,
              input: c,
              ...(l?.timeoutMs == null ? {} : { timeout_ms: l.timeoutMs }),
            }),
          })
        ).result;
      },
    });
  }
};
function Ru(r) {
  return r.length === 0
    ? "No WebMCP tools are available in this document."
    : [
        "WebMCP tools available in this document:",
        JSON.stringify(r, null, 2),
        "Call tools.call(name, input) to invoke one.",
      ].join(`
`);
}
var bo = q({ capability: ft, info: yo, internalOnly: !0 });
var Go = [lo, so, mo, fo, bo],
  Au = Iu(Go),
  ac = ({
    browserId: r,
    capabilities: e,
    documentation: o,
    tabId: a,
    transport: n,
  }) => {
    let s = {};
    for (let c of e ?? []) {
      let l = Au.get(c.id);
      l != null &&
        Object.assign(s, {
          [c.id]: l({
            browserId: r,
            documentation: o,
            info: c,
            tabId: a,
            transport: n,
          }),
        });
    }
    return new ot(s);
  };
function Iu(r) {
  return new Map(r.map((e) => [e.id, e.create]));
}
var yt = class {
    constructor({
      browserId: e,
      dialogId: o,
      tabId: a,
      transport: n,
      type: s,
    }) {
      ((this.browserId = e),
        (this.dialogId = o),
        (this.tabId = a),
        (this.transport = n),
        (this.type = s));
    }
    async dismiss() {
      await this.transport.send({
        command: _e.create({
          action: "dismiss",
          browser_id: this.browserId,
          dialog_id: this.dialogId,
          tab_id: this.tabId,
        }),
      });
    }
  },
  Vr = class extends yt {
    constructor(e) {
      super({ ...e, type: "alert" });
    }
  },
  Zr = class extends yt {
    constructor(e) {
      super({ ...e, type: "beforeunload" });
    }
  },
  Gr = class extends yt {
    constructor(e) {
      super({ ...e, type: "confirm" });
    }
    async accept() {
      await this.transport.send({
        command: _e.create({
          action: "accept",
          browser_id: this.browserId,
          dialog_id: this.dialogId,
          tab_id: this.tabId,
        }),
      });
    }
  },
  Hr = class extends yt {
    constructor(e) {
      super({ ...e, type: "prompt" });
    }
    async accept(e) {
      if (typeof e != "string") throw new Error("prompt.accept requires text");
      await this.transport.send({
        command: _e.create({
          action: "accept",
          browser_id: this.browserId,
          dialog_id: this.dialogId,
          prompt_text: e,
          tab_id: this.tabId,
        }),
      });
    }
  },
  te = class {
    #e;
    #t;
    constructor({
      browserId: e,
      capabilities: o,
      documentation: a,
      transport: n,
      tabPayload: s = {},
    }) {
      if (!s.id) throw new Error("Tab requires an id");
      ((this.#e = e),
        (this.#t = n),
        (this.id = s.id),
        (this.playwright = new Xe({
          browserId: e,
          tabId: this.id,
          transport: this.#t,
        })),
        (this.dom_cua = new Ke({
          browserId: e,
          tabId: this.id,
          transport: this.#t,
        })),
        (this.cua = new Ye({
          browserId: e,
          tabId: this.id,
          transport: this.#t,
        })),
        (this.ax = new Qe({
          browserId: e,
          tabId: this.id,
          transport: this.#t,
        })),
        (this.content = new et({
          browserId: e,
          tabId: this.id,
          transport: this.#t,
        })),
        (this.clipboard = new tt({
          browserId: e,
          tabId: this.id,
          transport: this.#t,
        })),
        (this.dev = new rt({
          browserId: e,
          tabId: this.id,
          transport: this.#t,
        })),
        (this.capabilities = ac({
          browserId: this.#e,
          capabilities: o,
          documentation: a,
          tabId: this.id,
          transport: this.#t,
        })));
    }
    async goto(e) {
      if (!e) throw new Error("tab.goto requires a url");
      if ((this.#o(), !this.id)) throw new Error("tab.goto requires a tab id");
      await this.#t.send({
        command: Nt.create({ browser_id: this.#e, tab_id: this.id, url: e }),
      });
    }
    async markHandoff() {
      await this.#r("handoff");
    }
    async markDeliverable() {
      await this.#r("deliverable");
    }
    async requestManualHandoff() {
      await this.#t.send({
        command: zt.create({ browser_id: this.#e, tab_id: this.id }),
      });
    }
    async back() {
      (this.#o(),
        await this.#t.send({
          command: we.create({ browser_id: this.#e, tab_id: this.id }),
        }));
    }
    async forward() {
      (this.#o(),
        await this.#t.send({
          command: xe.create({ browser_id: this.#e, tab_id: this.id }),
        }));
    }
    async reload() {
      (this.#o(),
        await this.#t.send({
          command: $t.create({ browser_id: this.#e, tab_id: this.id }),
        }));
    }
    async close() {
      (this.#o(),
        await this.#t.send({
          command: Ot.create({ browser_id: this.#e, tab_id: this.id }),
        }));
    }
    async screenshot(e = {}) {
      this.#o();
      let o = { browser_id: this.#e, tab_id: this.id, fullPage: e.fullPage };
      if (e.clip) {
        let { x: n, y: s, width: c, height: l } = e.clip;
        if (
          typeof n != "number" ||
          typeof s != "number" ||
          typeof c != "number" ||
          typeof l != "number"
        )
          throw new Error(
            "tab.screenshot clip requires x, y, width, and height",
          );
        ((o.cropX = n), (o.cropY = s), (o.cropWidth = c), (o.cropHeight = l));
      }
      let a = await this.#t.send({ command: Qt.create(o) });
      return Je(a.data);
    }
    async title() {
      return (await this.#a()).title;
    }
    async url() {
      return (await this.#a()).url;
    }
    async getJsDialog() {
      this.#o();
      let o = (
        await this.#t.send({
          command: Vt.create({ browser_id: this.#e, tab_id: this.id }),
        })
      ).dialog;
      if (o != null)
        return ku({
          browserId: this.#e,
          dialogId: o.id,
          tabId: this.id,
          transport: this.#t,
          type: o.type,
        });
    }
    async #r(e) {
      (this.#o(),
        await this.#t.send({
          command: Bt.create({
            browser_id: this.#e,
            tab_id: this.id,
            status: e,
          }),
        }));
    }
    #o() {
      if (!this.#t) throw new Error("Tab is not bound to a transport");
    }
    async #a() {
      return (
        this.#o(),
        await this.#t.send({
          command: ge.create({ browser_id: this.#e, tab_id: this.id }),
        })
      );
    }
  };
function ku({ browserId: r, dialogId: e, tabId: o, transport: a, type: n }) {
  switch (n) {
    case "alert":
      return new Vr({ browserId: r, dialogId: e, tabId: o, transport: a });
    case "beforeunload":
      return new Zr({ browserId: r, dialogId: e, tabId: o, transport: a });
    case "confirm":
      return new Gr({ browserId: r, dialogId: e, tabId: o, transport: a });
    case "prompt":
      return new Hr({ browserId: r, dialogId: e, tabId: o, transport: a });
  }
}
var xo = {};
d(xo, { Call: () => ve, GetAuditTrail: () => Te, commands: () => Eu });
var ve = {};
d(ve, {
  PayloadSchema: () => Jr,
  ResultSchema: () => ju,
  commandType: () => go,
  create: () => Ou,
});
var go = "browser_management_call",
  Jr = t.object({
    args: t.array(t.unknown()),
    browser_id: t.string(),
    method: t.string(),
    namespace: t.string(),
  }),
  ju = t.object({ value: t.unknown() });
function Ou(r) {
  return new i(go, Jr, r);
}
var Te = {};
d(Te, {
  AuditChangeSchema: () => sc,
  PayloadSchema: () => nc,
  ResultSchema: () => Bu,
  commandType: () => wo,
  create: () => zu,
});
var wo = "browser_management_get_audit_trail",
  nc = Jr.pick({ browser_id: !0 }),
  sc = Jr.omit({ browser_id: !0 }).extend({
    before: t
      .object({
        bookmarks: t.array(
          t.object({
            id: t.string(),
            index: t.number().optional(),
            parentId: t.string().optional(),
            title: t.string(),
            url: t.string().optional(),
          }),
        ),
        tabLayout: t.object({
          groups: t.array(
            t.object({
              collapsed: t.boolean(),
              color: t.string(),
              id: t.number(),
              title: t.string().optional(),
              windowId: t.number(),
            }),
          ),
          tabs: t.array(
            t.object({
              autoDiscardable: t.boolean(),
              groupId: t.number(),
              id: t.number(),
              index: t.number(),
              pinned: t.boolean(),
              url: t.string().optional(),
              windowId: t.number(),
            }),
          ),
        }),
        windows: t.array(
          t.object({
            focused: t.boolean(),
            height: t.number().optional(),
            id: t.number(),
            left: t.number().optional(),
            state: t.string().optional(),
            top: t.number().optional(),
            width: t.number().optional(),
          }),
        ),
      })
      .partial(),
    createdAt: t.number(),
    result: t.union([t.number(), t.object({ id: t.string() })]).optional(),
  }),
  Bu = t.object({ changes: t.array(sc) });
function zu(r) {
  return new i(wo, nc, r);
}
var Eu = { [go]: ve, [wo]: Te };
var ic = {
  id: "management",
  description:
    "Organize windows, tabs, tab groups, and bookmarks. Use only for user-requested browser organization.",
};
var ht = class extends ee {
    constructor({ browserId: e, documentation: o, info: a, transport: n }) {
      (super(n, e, o, a),
        (this.windows = this.createNamespace("windows")),
        (this.tabs = this.createNamespace("tabs")),
        (this.tabGroups = this.createNamespace("tabGroups")),
        (this.bookmarks = this.createNamespace("bookmarks")));
    }
    async getAuditTrail() {
      let e = await this.transport.send({
        command: Te.create({ browser_id: this.browserId }),
      });
      return Te.ResultSchema.parse(e);
    }
    async invoke(e, o, ...a) {
      let n = await this.transport.send({
        command: ve.create({
          args: a,
          browser_id: this.browserId,
          method: o,
          namespace: e,
        }),
      });
      return ve.ResultSchema.parse(n).value;
    }
    createNamespace(e) {
      return new Proxy(
        {},
        {
          get: (o, a) => {
            if (
              !(
                typeof a != "string" ||
                a === "then" ||
                a === "toJSON" ||
                a in Object.prototype
              )
            )
              return (...n) => this.invoke(e, a, ...n);
          },
        },
      );
    }
  },
  _o = at({ capability: ht, info: ic });
var vo = {};
d(vo, { Get: () => Pe, Set: () => Re, commands: () => Nu });
var Pe = {};
d(Pe, {
  PayloadSchema: () => cc,
  ResultSchema: () => Du,
  commandType: () => Co,
  create: () => Mu,
});
var Co = "browser_visibility_get",
  cc = t.object({ browser_id: t.string() }),
  Du = t.object({ visible: t.boolean() });
function Mu(r) {
  return new i(Co, cc, r);
}
var Re = {};
d(Re, {
  PayloadSchema: () => lc,
  ResultSchema: () => Lu,
  commandType: () => So,
  create: () => Fu,
});
var So = "browser_visibility_set",
  lc = t.object({ browser_id: t.string(), visible: t.boolean() }),
  Lu = t.object({});
function Fu(r) {
  return new i(So, lc, r);
}
var Nu = { [Co]: Pe, [So]: Re };
var dc = {
  id: "visibility",
  description:
    "Use to show or hide the browser to the user, and to determine the browser's current visibility. Keep browser work in the background unless the user asks to see it or live viewing is useful. When the browser should be visible, call set(true).",
};
var bt = class extends ee {
    constructor({ browserId: e, documentation: o, info: a, transport: n }) {
      super(n, e, o, a);
    }
    async set(e) {
      let o = await this.transport.send({
        command: Re.create({ browser_id: this.browserId, visible: e }),
      });
      Re.ResultSchema.parse(o);
    }
    async get() {
      let e = await this.transport.send({
        command: Pe.create({ browser_id: this.browserId }),
      });
      return Pe.ResultSchema.parse(e).visible;
    }
  },
  To = at({ capability: bt, info: dc });
var Ao = {};
d(Ao, { Reset: () => Ae, Set: () => Ie, commands: () => Vu });
var Ae = {};
d(Ae, {
  PayloadSchema: () => pc,
  ResultSchema: () => $u,
  commandType: () => Po,
  create: () => qu,
});
var Po = "browser_viewport_reset",
  pc = t.object({ browser_id: t.string() }),
  $u = t.object({});
function qu(r) {
  return new i(Po, pc, r);
}
var Ie = {};
d(Ie, {
  PayloadSchema: () => mc,
  ResultSchema: () => Uu,
  commandType: () => Ro,
  create: () => Wu,
});
var Ro = "browser_viewport_set",
  mc = t.object({
    browser_id: t.string(),
    height: t.number().int().positive(),
    width: t.number().int().positive(),
  }),
  Uu = t.object({});
function Wu(r) {
  return new i(Ro, mc, r);
}
var Vu = { [Po]: Ae, [Ro]: Ie };
var uc = {
  id: "viewport",
  description:
    "Controls an explicit browser viewport override for responsive or device-size testing. Use it when a task calls for specific dimensions or breakpoint validation; otherwise leave it unset so the browser uses its normal viewport. Reset temporary overrides before finishing unless the user asked to keep them.",
};
var gt = class extends ee {
    constructor({ browserId: e, documentation: o, info: a, transport: n }) {
      super(n, e, o, a);
    }
    async set(e) {
      let o = await this.transport.send({
        command: Ie.create({ browser_id: this.browserId, ...e }),
      });
      Ie.ResultSchema.parse(o);
    }
    async reset() {
      let e = await this.transport.send({
        command: Ae.create({ browser_id: this.browserId }),
      });
      Ae.ResultSchema.parse(e);
    }
  },
  Io = at({ capability: gt, info: uc });
var Ho = [_o, To, Io],
  Zu = Gu(Ho),
  fc = ({ browserId: r, capabilities: e, documentation: o, transport: a }) => {
    let n = {};
    for (let s of e ?? []) {
      let c = Zu.get(s.id);
      c != null &&
        (n[s.id] = c({
          browserId: r,
          documentation: o,
          info: s,
          transport: a,
        }));
    }
    return new ot(n);
  };
function Gu(r) {
  return new Map(r.map((e) => [e.id, e.create]));
}
var wt = class {
    #e;
    #t;
    #r;
    constructor({
      capabilities: e,
      onBrowserUsed: o,
      transport: a,
      browserId: n,
    }) {
      ((this.#e = n), (this.#t = o), (this.#r = a), (this.browserId = n));
      let s = new Q(a);
      this.capabilities = fc({
        browserId: n,
        capabilities: e?.browser,
        documentation: s,
        transport: a,
      });
      let c = (l) =>
        new te({
          browserId: n,
          capabilities: e?.tab,
          documentation: s,
          transport: a,
          tabPayload: l,
        });
      ((this.tabs = new Ze({ browserId: n, createTab: c, transport: a })),
        (this.user = new Ge({ browserId: n, createTab: c, transport: a })));
    }
    async documentation() {
      return (
        this.#t?.(),
        await this.#r.send({
          command: It.create({ browser_id: this.browserId }),
        })
      );
    }
    async history(e = {}) {
      if (e === null || Array.isArray(e) || typeof e != "object")
        throw new Error("browser.history expects an options object");
      let o = { browser_id: this.#e },
        a = Hu(e.queries),
        n = Ju(e.limit),
        s = yc(e.from, "from"),
        c = yc(e.to, "to");
      return (
        a !== void 0 && (o.queries = a),
        n !== void 0 && (o.limit = n),
        s !== void 0 && (o.from = s),
        c !== void 0 && (o.to = c),
        (await this.#r.send({ command: Wt.create(o) })).items
      );
    }
    async nameSession(e) {
      let o = e.trim();
      if (!o) throw new Error("browser.nameSession requires a name");
      await this.#r.send({
        command: Et.create({ browser_id: this.browserId, name: o }),
      });
    }
  },
  yc = (r, e) => {
    if (r == null) return;
    let o = r instanceof Date ? r : new Date(r);
    if (Number.isNaN(o.getTime()))
      throw new Error(`browser.history received an invalid ${e} date`);
    return o.toISOString();
  },
  Hu = (r) => {
    if (r == null) return;
    if (!Array.isArray(r) || r.length === 0)
      throw new Error("browser.history received invalid queries");
    let e = [];
    for (let o of r) {
      if (typeof o != "string")
        throw new Error("browser.history received invalid queries");
      e.push(o);
    }
    return e;
  },
  Ju = (r) => {
    if (r != null) {
      if (!Number.isInteger(r) || r <= 0)
        throw new Error("browser.history received an invalid limit");
      return r;
    }
  };
var xt = class {
    constructor({ executeAgentCommand: e, displaySideEffect: o } = {}) {
      if (typeof e != "function")
        throw new Error(
          "FunctionAgentTransport requires an executeAgentCommand function",
        );
      ((this.executeAgentCommand = e), (this.displaySideEffect = o ?? Yu));
    }
    async display(e) {
      await this.displaySideEffect(e);
    }
    async send({ command: e, timeoutMs: o }) {
      let a = e.toJSON(),
        n = await this.executeAgentCommand({
          ...a,
          client_timeout_ms: typeof o == "number" && o > 0 ? o : void 0,
        }),
        s = await Xu(n, this.displaySideEffect);
      if (s == null) throw new Error("transport send returned empty response");
      return s;
    }
  },
  Xu = async (r, e) => {
    let o = Ku.safeParse(r);
    if (!o.success) return r;
    for (let s of o.data.side_effects) await e(s);
    let { side_effects: a, ...n } = o.data;
    return n;
  },
  Ku = t.object({ side_effects: t.array(t.string()) }).passthrough(),
  Yu = async (r) => {
    let e = globalThis.display;
    if (typeof e == "function") {
      await e(r);
      return;
    }
    console.log(r);
  };
function hc({ apiManifest: r, browserInfo: e, disabledMemberIds: o }) {
  let a = new Set(o);
  for (let [n, s] of Object.entries(r.interfaces))
    for (let [c, l] of Object.entries(s)) {
      let u = `${n}.${c}`;
      (e.apiSupportOverrides?.[u] ??
        l.unsupportedByDefaultIn?.includes(e.type) !== !0) ||
        a.add(u);
    }
  return a;
}
var ko = class {
  apiManifest;
  disabledMemberIds;
  view;
  constructor({ apiManifest: e, decorateTab: o, disabledMemberIds: a }) {
    ((this.apiManifest = e),
      (this.disabledMemberIds = a),
      (this.view = Qu(e, o)));
  }
  createBrowser = ({ browserInfo: e, onBrowserUsed: o, transport: a }) => {
    let n = hc({
      apiManifest: this.apiManifest,
      browserInfo: e,
      disabledMemberIds: this.disabledMemberIds,
    });
    return this.view(
      new wt({
        browserId: e.id,
        capabilities: e.capabilities,
        onBrowserUsed: o,
        transport: a,
      }),
      n,
    );
  };
  wrapAgent(e) {
    return this.view(e, this.disabledMemberIds);
  }
  createAgent({ displaySideEffect: e, executeAgentCommand: o }) {
    return this.wrapAgent(
      new Ve({
        createBrowser: this.createBrowser,
        transport: new xt({ displaySideEffect: e, executeAgentCommand: o }),
      }),
    );
  }
};
function Qu(r, e) {
  let o = new WeakMap(),
    a = Object.keys(r.interfaces).map((n) => {
      let s = Reflect.get(Jo, n);
      if (!ef(s))
        throw new Error(`Browser API interface has no runtime type: ${n}`);
      return [s, n];
    });
  return (n, s) => {
    let c = new WeakMap(),
      l = (p) => {
        if (p == null || typeof p != "object") return p;
        let w = o.get(p);
        return (
          w ??
          (Array.isArray(p)
            ? p.map(l)
            : ro(p)
              ? Object.fromEntries(Object.entries(p).map(([S, U]) => [S, l(U)]))
              : p)
        );
      },
      u = (p) => {
        if (p == null || typeof p != "object" || o.has(p)) return p;
        if (p instanceof Promise) return p.then(u);
        if (Array.isArray(p)) return p.map(u);
        let w = a.find(([R]) => p instanceof R)?.[1];
        if (w == null) return p;
        let S = c.get(p);
        if (S != null) return S;
        let U = (R) => typeof R != "string" || !s.has(`${w}.${R}`),
          ke = new Map(),
          _t = new Proxy(p, {
            get(R, z) {
              if (!U(z)) return;
              let D = Reflect.get(R, z, R);
              if (z === "constructor") return D;
              if (typeof D != "function") return u(D);
              let Xo = ke.get(z);
              if (Xo != null) return Xo;
              let Ko = (...gc) =>
                u(
                  Reflect.apply(
                    D,
                    R,
                    gc.map((wc) => l(wc)),
                  ),
                );
              return (ke.set(z, Ko), Ko);
            },
            getOwnPropertyDescriptor(R, z) {
              if (!U(z)) return;
              let D = Reflect.getOwnPropertyDescriptor(R, z);
              return D == null || !("value" in D)
                ? D
                : { ...D, value: u(D.value) };
            },
            has(R, z) {
              return U(z) && Reflect.has(R, z);
            },
            ownKeys(R) {
              return Reflect.ownKeys(R).filter(U);
            },
          });
        return (c.set(p, _t), o.set(_t, p), _t instanceof te && e?.(_t), _t);
      };
    return u(n);
  };
}
function ef(r) {
  return typeof r == "function" && r.prototype != null;
}
function bc({
  apiManifest: r,
  decorateTab: e,
  disabledMemberIds: o,
  displayBridge: a,
  executeAgentCommand: n,
}) {
  return new ko({
    apiManifest: r,
    decorateTab: e,
    disabledMemberIds: o,
  }).createAgent({
    displaySideEffect: Qo({ displayBridge: a, displayTruncateMaxChars: 1e5 }),
    executeAgentCommand: n,
  });
}
async function Yx(r = {}) {
  let e = globalThis.nodeRepl;
  if (e == null || typeof e.rpc != "function")
    throw new Error("Browser use requires a trusted Node REPL browser service");
  let o = e.rpc,
    a = {
      setup: (c) => o("browser", { method: "setup", params: c }),
      execute: (c) => o("browser", { method: "execute", params: c }),
    },
    { apiManifest: n, disabledMemberIds: s } = await a.setup({
      environment: r.environment ?? "codex-app",
      undocumentedApiMembers: r.undocumentedApiMembers,
      excludedDocumentation: r.excludedDocumentation,
    });
  return bc({
    apiManifest: n,
    decorateTab: r.decorateTab,
    disabledMemberIds: new Set(s),
    displayBridge: {
      displayImage: (c) => e.emitImage(c),
      displayValue: (c) => e.write(c.value),
    },
    executeAgentCommand: a.execute,
  });
}
export { Yx as setupBrowserRuntime };
