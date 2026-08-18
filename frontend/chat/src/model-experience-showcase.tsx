import { motion } from "motion/react";
import {
  ArrowLeft,
  ArrowRight,
  Check,
  ChevronDown,
  ChevronRight,
  Eye,
  EyeOff,
  KeyRound,
  Layers3,
  LogIn,
  Plus,
  RefreshCw,
  Search,
  SendHorizontal,
  Settings2,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Trash2,
  X,
  Zap,
} from "lucide-react";
import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import codexIcon from "./assets/provider-icons/codex.svg";
import deepseekIcon from "./assets/provider-icons/deepseek.svg";
import opencodeIcon from "./assets/provider-icons/opencode.svg";
import openrouterIcon from "./assets/provider-icons/openrouter.svg";
import "./model-experience-showcase.css";

type Surface = "chat" | "settings";
type VariantId = "dock" | "split" | "deck" | "command" | "rail";
type ModelOption = {
  id: string;
  name: string;
  source: string;
  icon: "deepseek" | "opencode" | "codex" | "openrouter";
  effort: string;
  detail: string;
};

const PROVIDER_ICONS: Record<ModelOption["icon"], string> = {
  codex: codexIcon,
  deepseek: deepseekIcon,
  opencode: opencodeIcon,
  openrouter: openrouterIcon,
};

const VARIANTS: Array<{ id: VariantId; number: string; name: string; note: string }> = [
  { id: "dock", number: "01", name: "悬浮坞", note: "选择层锚定输入框，配置用居中表单" },
  { id: "split", number: "02", name: "双段胶囊", note: "模型与推理强度并排，配置用侧滑表单" },
  { id: "deck", number: "03", name: "卡片叠层", note: "候选像卡片展开，配置用主从详情" },
  { id: "command", number: "04", name: "指令面板", note: "搜索优先，配置也采用快速查找" },
  { id: "rail", number: "05", name: "等宽上展", note: "唯一胶囊等宽向上展开，全供应商分组滑动" },
];

const MODELS: ModelOption[] = [
  {
    id: "oc-v4-flash",
    name: "deepseek-v4-flash",
    source: "OpenCode Go",
    icon: "opencode",
    effort: "高",
    detail: "主账号 · 128K",
  },
  {
    id: "ds-chat",
    name: "deepseek-chat",
    source: "DeepSeek 官方",
    icon: "deepseek",
    effort: "中",
    detail: "API Key · 64K",
  },
  {
    id: "codex-main",
    name: "gpt-5.2-codex",
    source: "Codex 登录",
    icon: "codex",
    effort: "高",
    detail: "OAuth · 400K",
  },
  {
    id: "router-v4",
    name: "deepseek-v4-flash",
    source: "OpenRouter 备用",
    icon: "openrouter",
    effort: "低",
    detail: "备用密钥 · 128K",
  },
];

const RAIL_MODELS: ModelOption[] = [
  ...MODELS,
  { id: "oc-kimi", name: "kimi-k2.5", source: "OpenCode Go", icon: "opencode", effort: "高", detail: "主账号 · 256K" },
  { id: "oc-glm", name: "glm-5", source: "OpenCode Go", icon: "opencode", effort: "中", detail: "主账号 · 200K" },
  { id: "ds-reasoner", name: "deepseek-reasoner", source: "DeepSeek 官方", icon: "deepseek", effort: "高", detail: "API Key · 64K" },
  { id: "codex-mini", name: "gpt-5.1-codex-mini", source: "Codex 登录", icon: "codex", effort: "中", detail: "OAuth · 400K" },
  { id: "router-claude", name: "claude-sonnet-4.5", source: "OpenRouter 备用", icon: "openrouter", effort: "高", detail: "备用密钥 · 200K" },
  { id: "router-gemini", name: "gemini-2.5-pro", source: "OpenRouter 备用", icon: "openrouter", effort: "中", detail: "备用密钥 · 1M" },
];

const RAIL_SOURCES = ["OpenCode Go", "DeepSeek 官方", "Codex 登录", "OpenRouter 备用"];

const CONNECTIONS = [
  { id: "codex", name: "Codex 登录", meta: "OAuth · 已连接", models: 6, icon: "codex", state: "就绪" },
  { id: "opencode", name: "OpenCode Go 主账号", meta: "登录凭据 · 4 个模型", models: 4, icon: "opencode", state: "就绪" },
  { id: "deepseek", name: "DeepSeek 官方", meta: "sk-••••••9Q · api.deepseek.com", models: 2, icon: "deepseek", state: "就绪" },
  { id: "openrouter", name: "OpenRouter 备用", meta: "sk-or-••••K2 · 仅故障切换", models: 3, icon: "openrouter", state: "备用" },
] as const;

type MemohProviderKind = "api" | "codex" | "opencode" | "custom";
type MemohModel = {
  id: string;
  name: string;
  metadata: string;
  enabled: boolean;
  effort: string;
};
type MemohProvider = {
  id: string;
  name: string;
  icon: string;
  kind: MemohProviderKind;
  clientType: string;
  description: string;
  status: "已连接" | "未配置";
  baseUrl?: string;
  secret?: string;
  account?: string;
  models: MemohModel[];
};

const MEMOH_PROVIDERS: MemohProvider[] = [
  {
    id: "deepseek-main",
    name: "DeepSeek 官方",
    icon: "deepseek",
    kind: "api",
    clientType: "OpenAI Compatible",
    description: "官方 API · 独立计费",
    status: "已连接",
    baseUrl: "https://api.deepseek.com/v1",
    secret: "已保存 · sk-••••••9Q",
    models: [
      { id: "deepseek-chat", name: "deepseek-chat", metadata: "64K · 工具调用", enabled: true, effort: "中" },
      { id: "deepseek-reasoner", name: "deepseek-reasoner", metadata: "64K · 深度推理", enabled: true, effort: "高" },
    ],
  },
  {
    id: "codex-account",
    name: "Codex 登录",
    icon: "codex",
    kind: "codex",
    clientType: "Codex OAuth",
    description: "huayue@example.com · 订阅账号",
    status: "已连接",
    account: "huayue@example.com",
    models: [
      { id: "gpt-5.2-codex", name: "gpt-5.2-codex", metadata: "400K · 视觉 · 工具", enabled: true, effort: "高" },
      { id: "gpt-5.1-codex-mini", name: "gpt-5.1-codex-mini", metadata: "400K · 快速", enabled: true, effort: "中" },
    ],
  },
  {
    id: "opencode-main",
    name: "OpenCode Go 主账号",
    icon: "opencode",
    kind: "opencode",
    clientType: "OpenCode Login",
    description: "登录凭据 · 自动同步模型",
    status: "已连接",
    account: "OpenCode Go · 主账号",
    models: [
      { id: "deepseek-v4-flash", name: "deepseek-v4-flash", metadata: "128K · 视觉 · 工具", enabled: true, effort: "高" },
      { id: "kimi-k2.5", name: "kimi-k2.5", metadata: "256K · 视觉", enabled: true, effort: "高" },
      { id: "glm-5", name: "glm-5", metadata: "200K · 工具调用", enabled: false, effort: "中" },
    ],
  },
];

const MEMOH_TEMPLATES: MemohProvider[] = [
  {
    id: "openrouter-template",
    name: "OpenRouter",
    icon: "openrouter",
    kind: "api",
    clientType: "OpenAI Compatible",
    description: "一个密钥连接多个模型",
    status: "未配置",
    baseUrl: "https://openrouter.ai/api/v1",
    models: [],
  },
  {
    id: "deepseek-template",
    name: "DeepSeek API",
    icon: "deepseek",
    kind: "api",
    clientType: "OpenAI Compatible",
    description: "预填官方 Base URL",
    status: "未配置",
    baseUrl: "https://api.deepseek.com/v1",
    models: [],
  },
  {
    id: "codex-template",
    name: "Codex 订阅",
    icon: "codex",
    kind: "codex",
    clientType: "Codex OAuth",
    description: "浏览器登录，无需 API Key",
    status: "未配置",
    models: [],
  },
  {
    id: "custom-template",
    name: "自定义兼容接口",
    icon: "custom",
    kind: "custom",
    clientType: "OpenAI Compatible",
    description: "Base URL + API Key",
    status: "未配置",
    models: [],
  },
];

function BrandIcon({ name, size = 20 }: { name: ModelOption["icon"] | string; size?: number }) {
  const icon = PROVIDER_ICONS[name as ModelOption["icon"]];
  return (
    <span className={`mx-brand mx-brand--${name}`} style={{ width: size + 12, height: size + 12 }} aria-hidden="true">
      {icon && <img src={icon} alt="" width={size} height={size} />}
      <span>{name.slice(0, 1).toUpperCase()}</span>
    </span>
  );
}

function ModelLabel({ model, compact = false }: { model: ModelOption; compact?: boolean }) {
  return (
    <span className="mx-model-label">
      <BrandIcon name={model.icon} size={compact ? 16 : 19} />
      <span className="mx-model-label__copy">
        <strong>{model.name}<i>：</i>{model.source}</strong>
        {!compact && <small>{model.detail} · 推理 {model.effort}</small>}
      </span>
    </span>
  );
}

function SelectionMenu({
  variant,
  selected,
  onSelect,
  onClose,
}: {
  variant: VariantId;
  selected: ModelOption;
  onSelect: (model: ModelOption) => void;
  onClose: () => void;
}) {
  const [query, setQuery] = useState("");
  const options = useMemo(
    () => MODELS.filter((model) => `${model.name} ${model.source}`.toLowerCase().includes(query.toLowerCase())),
    [query],
  );
  const showSearch = variant === "command" || variant === "dock";

  return (
    <motion.div
      className={`mx-model-menu mx-model-menu--${variant}`}
      role="listbox"
      aria-label="选择本轮模型"
      initial={{ opacity: 0, y: 12, scale: 0.96 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ type: "spring", duration: 0.3, bounce: 0 }}
    >
      <div className="mx-model-menu__head">
        <div>
          <strong>{variant === "command" ? "切换运行模型" : "本轮使用"}</strong>
          <span>切换只影响下一轮发送</span>
        </div>
        <button type="button" aria-label="关闭模型选择" onClick={onClose}><X size={18} /></button>
      </div>
      {showSearch && (
        <label className="mx-search">
          <Search size={17} aria-hidden="true" />
          <span className="sr-only">搜索模型或来源</span>
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索模型或来源" autoFocus />
          {variant === "command" && <kbd>⌘ K</kbd>}
        </label>
      )}
      <div className="mx-model-menu__options">
        {options.map((model, index) => (
          <motion.button
            type="button"
            role="option"
            aria-selected={selected.id === model.id}
            className="mx-model-option"
            key={model.id}
            onClick={() => onSelect(model)}
            initial={{ opacity: 0, y: variant === "deck" ? 12 : 4 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: Math.min(index * 0.035, 0.12), duration: 0.2 }}
            whileTap={{ scale: 0.96 }}
          >
            <ModelLabel model={model} />
            {selected.id === model.id ? <Check size={18} /> : <ArrowRight size={17} />}
          </motion.button>
        ))}
      </div>
      {variant === "split" && (
        <div className="mx-effort-row" aria-label="推理强度">
          <span><Sparkles size={16} />推理强度</span>
          {['低', '中', '高'].map((effort) => <button type="button" key={effort} className={effort === selected.effort ? "is-active" : ""}>{effort}</button>)}
        </div>
      )}
    </motion.div>
  );
}

function RailModelPicker({
  open,
  selected,
  onOpenChange,
  onSelect,
}: {
  open: boolean;
  selected: ModelOption;
  onOpenChange: (open: boolean) => void;
  onSelect: (model: ModelOption) => void;
}) {
  const selectedIndex = Math.max(0, RAIL_MODELS.findIndex((model) => model.id === selected.id));
  const [activeIndex, setActiveIndex] = useState(selectedIndex);
  const optionRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const listRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);

  // 必要 effect：打开时同步选中项并聚焦（focus 需 DOM 提交后执行），不可改为渲染期计算
  useEffect(() => {
    if (!open) return;
    setActiveIndex(selectedIndex);
    window.setTimeout(() => optionRefs.current[selectedIndex]?.focus({ preventScroll: true }), 0);
  }, [open, selectedIndex]);

  function moveFocus(nextIndex: number) {
    const normalized = (nextIndex + RAIL_MODELS.length) % RAIL_MODELS.length;
    setActiveIndex(normalized);
    const option = optionRefs.current[normalized];
    const list = listRef.current;
    option?.focus({ preventScroll: true });
    if (!option || !list) return;
    const optionRect = option.getBoundingClientRect();
    const listRect = list.getBoundingClientRect();
    if (optionRect.top < listRect.top + 38) list.scrollTop += optionRect.top - listRect.top - 38;
    if (optionRect.bottom > listRect.bottom - 8) list.scrollTop += optionRect.bottom - listRect.bottom + 8;
  }

  function closePicker() {
    onOpenChange(false);
    triggerRef.current?.focus({ preventScroll: true });
  }

  function selectModel(model: ModelOption) {
    onSelect(model);
    triggerRef.current?.focus({ preventScroll: true });
  }

  function onListKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === "Escape") {
      event.preventDefault();
      closePicker();
      return;
    }
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      moveFocus(activeIndex + (event.key === "ArrowDown" ? 1 : -1));
      return;
    }
    if ((event.key === "Enter" || event.key === " ") && document.activeElement === optionRefs.current[activeIndex]) {
      event.preventDefault();
      selectModel(RAIL_MODELS[activeIndex]);
    }
  }

  return (
    <div className="mx-rail-picker">
      <div className={`mx-rail-shell ${open ? "is-open" : ""}`}>
        <div className="mx-rail-panel" aria-hidden={!open}>
          <header className="mx-rail-panel__head">
            <div><span>所有供应商</span><strong>选择下一轮使用的模型</strong></div>
            <small>{RAIL_MODELS.length} 个可用模型</small>
          </header>
          <div
            id="mx-rail-model-list"
            ref={listRef}
            className="mx-rail-model-list"
            role="listbox"
            aria-label="所有供应商的模型"
            onKeyDown={onListKeyDown}
          >
            {RAIL_SOURCES.map((source) => {
              const sourceModels = RAIL_MODELS.filter((model) => model.source === source);
              const sourceIcon = sourceModels[0]?.icon;
              return (
                <section className="mx-rail-source" aria-label={source} key={source}>
                  <div className="mx-rail-source__head">
                    {sourceIcon && <BrandIcon name={sourceIcon} size={15} />}
                    <strong>{source}</strong>
                    <span>{sourceModels.length}</span>
                  </div>
                  <div className="mx-rail-source__models">
                    {sourceModels.map((model) => {
                      const index = RAIL_MODELS.findIndex((candidate) => candidate.id === model.id);
                      const isSelected = selected.id === model.id;
                      return (
                        <motion.button
                          ref={(node) => { optionRefs.current[index] = node; }}
                          type="button"
                          role="option"
                          aria-selected={isSelected}
                          tabIndex={open && activeIndex === index ? 0 : -1}
                          className="mx-rail-model"
                          key={model.id}
                          onFocus={() => setActiveIndex(index)}
                          onClick={() => selectModel(model)}
                          whileTap={{ scale: 0.96 }}
                        >
                          <span className="mx-rail-model__copy"><strong>{model.name}</strong><small>{model.detail}</small></span>
                          <span className="mx-rail-model__effort"><Sparkles size={13} />{model.effort}</span>
                          <span className={`mx-rail-model__state ${isSelected ? "is-selected" : ""}`} aria-hidden="true">
                            {isSelected ? <Check size={17} /> : <ArrowRight size={16} />}
                          </span>
                        </motion.button>
                      );
                    })}
                  </div>
                </section>
              );
            })}
          </div>
        </div>
        <motion.button
          ref={triggerRef}
          type="button"
          className="mx-rail-trigger"
          aria-haspopup="listbox"
          aria-controls="mx-rail-model-list"
          aria-expanded={open}
          onClick={() => onOpenChange(!open)}
          whileTap={{ scale: 0.96 }}
        >
          <ModelLabel model={selected} compact />
          <ChevronDown size={18} aria-hidden="true" />
        </motion.button>
      </div>
    </div>
  );
}

function ChatPrototype({ variant }: { variant: VariantId }) {
  const [selected, setSelected] = useState(MODELS[0]);
  const [open, setOpen] = useState(false);
  const isSplit = variant === "split";

  function choose(model: ModelOption) {
    setSelected(model);
    setOpen(false);
  }

  return (
    <section className={`mx-chat mx-chat--${variant}`} aria-label={`${variant} 对话选模原型`}>
      <div className="mx-chat__topline"><span>新对话</span><button type="button"><Settings2 size={18} />对话设置</button></div>
      <div className="mx-chat__empty">
        <span className="mx-akashic-mark">あ</span>
        <h2>今天想一起完成什么？</h2>
        <p>模型选择属于下一轮发送，不打断正在运行的这一轮。</p>
      </div>
      <div className="mx-composer-wrap">
        {variant !== "rail" && open && <SelectionMenu variant={variant} selected={selected} onSelect={choose} onClose={() => setOpen(false)} />}
        {variant === "rail" && <RailModelPicker open={open} selected={selected} onOpenChange={setOpen} onSelect={choose} />}
        <div className="mx-composer">
          <textarea aria-label="消息" placeholder="有问题，尽管问" />
          <div className="mx-composer__bar">
            <button type="button" className="mx-add" aria-label="添加附件"><Plus size={20} /></button>
            {variant !== "rail" && (
              <button
                type="button"
                className={`mx-model-trigger ${open ? "is-open" : ""}`}
                aria-haspopup="listbox"
                aria-expanded={open}
                onClick={() => setOpen((value) => !value)}
              >
                <ModelLabel model={selected} compact />
                {isSplit && <span className="mx-trigger-effort"><Sparkles size={14} />{selected.effort}</span>}
                <ChevronDown size={17} />
              </button>
            )}
            <motion.button type="button" className="mx-send" aria-label="发送" whileTap={{ scale: 0.96 }}>
              <SendHorizontal size={20} />
            </motion.button>
          </div>
        </div>
      </div>
    </section>
  );
}

function ConnectionRow({ connection, variant }: { connection: (typeof CONNECTIONS)[number]; variant: VariantId }) {
  return (
    <button type="button" className="mx-connection">
      <BrandIcon name={connection.icon} size={21} />
      <span className="mx-connection__copy"><strong>{connection.name}</strong><small>{connection.meta}</small></span>
      <span className="mx-connection__models">{connection.models} 模型</span>
      <span className={`mx-state mx-state--${connection.state === "备用" ? "standby" : "ready"}`}>{connection.state}</span>
      {variant === "deck" && <ArrowRight size={17} />}
    </button>
  );
}

function ConnectionForm({
  variant,
  onClose,
  onSaved,
}: {
  variant: VariantId;
  onClose: () => void;
  onSaved: () => void;
}) {
  const panelRef = useRef<HTMLDivElement>(null);
  const nameRef = useRef<HTMLInputElement>(null);
  const [showKey, setShowKey] = useState(false);
  const [source, setSource] = useState("api");

  useEffect(() => {
    nameRef.current?.focus();
    function onKeyDown(event: globalThis.KeyboardEvent) {
      if (event.key === "Escape") onClose();
      if (event.key !== "Tab" || !panelRef.current) return;
      const focusable = Array.from(panelRef.current.querySelectorAll<HTMLElement>("button, input, select"));
      const first = focusable[0];
      const last = focusable.at(-1);
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last?.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first?.focus();
      }
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  function submit(event: FormEvent) {
    event.preventDefault();
    onSaved();
  }

  return (
    <motion.div
      className="mx-scrim"
      role="presentation"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}
    >
      <motion.div
        className={`mx-connection-form mx-connection-form--${variant}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="mx-form-title"
        ref={panelRef}
        initial={{ opacity: 0, y: variant === "rail" ? 32 : 14, x: variant === "split" ? 32 : 0, scale: 0.97 }}
        animate={{ opacity: 1, y: 0, x: 0, scale: 1 }}
        transition={{ type: "spring", duration: 0.3, bounce: 0 }}
      >
        <header>
          <div><span className="mx-overline">新连接</span><h2 id="mx-form-title">接入一个模型来源</h2><p>一套登录或 API Key 是一个独立来源，可继续添加备用账号。</p></div>
          <button type="button" aria-label="关闭表单" onClick={onClose}><X size={20} /></button>
        </header>

        {variant === "rail" && <div className="mx-steps" aria-label="配置步骤"><span className="is-active">1 来源</span><span>2 凭据</span><span>3 模型</span></div>}

        <form onSubmit={submit}>
          <fieldset className="mx-source-choices">
            <legend>连接方式</legend>
            {[
              { id: "api", label: "API Key", icon: <KeyRound size={17} /> },
              { id: "codex", label: "Codex 登录", icon: <LogIn size={17} /> },
              { id: "opencode", label: "OpenCode", icon: <Layers3 size={17} /> },
            ].map((item) => (
              <label key={item.id} className={source === item.id ? "is-active" : ""}>
                <input type="radio" name="source" value={item.id} checked={source === item.id} onChange={() => setSource(item.id)} />
                {item.icon}<span>{item.label}</span><Check size={16} />
              </label>
            ))}
          </fieldset>

          <div className="mx-form-grid">
            <label><span>来源名称</span><input ref={nameRef} required defaultValue={source === "api" ? "DeepSeek 官方" : ""} placeholder="例如：OpenCode Go 主账号" /></label>
            {source === "api" ? (
              <>
                <label className="mx-secret"><span>API Key</span><input required type={showKey ? "text" : "password"} defaultValue="sk-demo-not-a-real-key" /><button type="button" aria-label={showKey ? "隐藏 API Key" : "显示 API Key"} onClick={() => setShowKey((value) => !value)}>{showKey ? <EyeOff size={18} /> : <Eye size={18} />}</button></label>
                <label><span>Base URL</span><input required type="url" defaultValue="https://api.deepseek.com/v1" /></label>
              </>
            ) : (
              <div className="mx-login-hint"><LogIn size={20} /><span><strong>保存后开始安全登录</strong><small>凭据由本机认证存储管理，不会显示完整 Token。</small></span></div>
            )}
            <label><span>模型名称</span><input required defaultValue={source === "api" ? "deepseek-chat" : ""} placeholder="例如：deepseek-v4-flash" /></label>
            <label><span>默认思考强度</span><select defaultValue="high"><option value="low">低</option><option value="medium">中</option><option value="high">高</option><option value="xhigh">极高</option></select></label>
          </div>
          <p className="mx-form-note">上下文长度、多模态与可选强度由模型注册表识别；识别不到时再显示高级覆盖项。</p>
          <footer><button type="button" className="mx-text-button" onClick={onClose}>取消</button><motion.button type="submit" className="mx-primary-button" whileTap={{ scale: 0.96 }}>保存并验证<ArrowRight size={17} /></motion.button></footer>
        </form>
      </motion.div>
    </motion.div>
  );
}

function MemohSwitch({ checked, label, onChange }: { checked: boolean; label: string; onChange: () => void }) {
  return (
    <button
      type="button"
      className="mx-memoh-switch"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      onClick={onChange}
    >
      <span />
    </button>
  );
}

function MemohProviderCard({
  provider,
  template = false,
  onOpen,
}: {
  provider: MemohProvider;
  template?: boolean;
  onOpen: () => void;
}) {
  return (
    <motion.button type="button" className="mx-memoh-provider-card" onClick={onOpen} whileTap={{ scale: 0.98 }}>
      <BrandIcon name={provider.icon} size={23} />
      <span className="mx-memoh-provider-card__copy">
        <strong>{provider.name}</strong>
        <small>{provider.description}</small>
      </span>
      {template ? (
        <span className="mx-memoh-provider-card__action">配置 <ChevronRight size={16} /></span>
      ) : (
        <span className="mx-memoh-provider-card__meta">
          <small>{provider.models.length} 个模型</small>
          <i><span />已连接</i>
        </span>
      )}
    </motion.button>
  );
}

function MemohModelRow({ model }: { model: MemohModel }) {
  const [enabled, setEnabled] = useState(model.enabled);
  const [effort, setEffort] = useState(model.effort);
  const [testing, setTesting] = useState(false);

  function testModel() {
    setTesting(true);
    window.setTimeout(() => setTesting(false), 900);
  }

  return (
    <div className="mx-memoh-model-row">
      <span className="mx-memoh-model-row__copy"><strong>{model.name}</strong><small>{model.metadata}</small></span>
      <label className="mx-memoh-effort">
        <span>思考强度</span>
        <select value={effort} onChange={(event) => setEffort(event.target.value)} aria-label={`${model.name} 默认思考强度`}>
          <option>低</option><option>中</option><option>高</option><option>极高</option>
        </select>
      </label>
      <button type="button" className={`mx-memoh-icon-button ${testing ? "is-testing" : ""}`} onClick={testModel} aria-label={`测试 ${model.name}`}>
        {testing ? <RefreshCw size={17} /> : <Zap size={17} />}
      </button>
      <MemohSwitch checked={enabled} label={`${enabled ? "停用" : "启用"} ${model.name}`} onChange={() => setEnabled((value) => !value)} />
    </div>
  );
}

function MemohProviderDetail({
  provider,
  onBack,
  onSaved,
}: {
  provider: MemohProvider;
  onBack: () => void;
  onSaved: (message: string) => void;
}) {
  const [enabled, setEnabled] = useState(provider.status === "已连接");
  const [showKey, setShowKey] = useState(false);
  const [query, setQuery] = useState("");
  const [refreshing, setRefreshing] = useState(false);
  const isLogin = provider.kind === "codex" || provider.kind === "opencode";
  const isDraft = provider.status === "未配置";
  const models = provider.models.filter((model) => model.name.toLowerCase().includes(query.toLowerCase()));

  function saveProvider(event: FormEvent) {
    event.preventDefault();
    onSaved(isDraft ? "连接已创建，正在读取模型目录。" : "配置已保存，新的请求将使用本次设置。");
  }

  function refreshModels() {
    setRefreshing(true);
    window.setTimeout(() => {
      setRefreshing(false);
      onSaved("模型目录已同步；能力信息来自注册表。 ");
    }, 900);
  }

  return (
    <motion.div className="mx-memoh-detail" initial={{ opacity: 0, x: 18 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.24 }}>
      <button type="button" className="mx-memoh-back" onClick={onBack}><ArrowLeft size={18} />全部连接</button>

      <section className="mx-memoh-identity">
        <BrandIcon name={provider.icon} size={28} />
        <span><small>{isDraft ? "新建连接" : provider.clientType}</small><h2>{provider.name}</h2></span>
        {!isDraft && <button type="button" className="mx-memoh-icon-button mx-memoh-delete" aria-label={`删除 ${provider.name}`}><Trash2 size={18} /></button>}
        <span className="mx-memoh-enabled"><small>{enabled ? "已启用" : "已停用"}</small><MemohSwitch checked={enabled} label={`${enabled ? "停用" : "启用"}连接`} onChange={() => setEnabled((value) => !value)} /></span>
      </section>

      <form className="mx-memoh-section" onSubmit={saveProvider}>
        <header><div><h3>连接配置</h3><p>{isLogin ? "账号授权由本机安全存储管理。" : "密钥保存后只显示模糊状态，不会重新回填明文。"}</p></div></header>
        <div className="mx-memoh-config-grid">
          <label><span>连接名称</span><input required defaultValue={provider.name} /></label>
          {isLogin ? (
            <div className="mx-memoh-account">
              <span className="mx-memoh-account__icon"><ShieldCheck size={20} /></span>
              <span><strong>{provider.account ?? "尚未登录"}</strong><small>{isDraft ? "登录后自动同步可用模型" : "授权有效 · 凭据不会显示在网页中"}</small></span>
              <button type="button" onClick={() => onSaved(isDraft ? "已打开安全登录流程。" : "已准备重新授权。")}>{isDraft ? "开始登录" : "重新连接"}</button>
            </div>
          ) : (
            <>
              <label><span>兼容协议</span><select defaultValue={provider.clientType}><option>OpenAI Compatible</option><option>Anthropic</option><option>Google Generative AI</option></select></label>
              <label className="mx-memoh-field-wide"><span>Base URL</span><input required type="url" defaultValue={provider.baseUrl} placeholder="https://api.example.com/v1" /></label>
              <label className="mx-secret mx-memoh-field-wide"><span>API Key</span><input required type={showKey ? "text" : "password"} defaultValue={provider.secret} placeholder="sk-…" /><button type="button" aria-label={showKey ? "隐藏 API Key" : "显示 API Key"} onClick={() => setShowKey((value) => !value)}>{showKey ? <EyeOff size={18} /> : <Eye size={18} />}</button></label>
            </>
          )}
        </div>
        <div className="mx-memoh-form-footer">
          <span><ShieldCheck size={16} />凭据仅保存在本机</span>
          <motion.button type="submit" className="mx-primary-button" whileTap={{ scale: 0.96 }}>{isDraft && isLogin ? "继续登录" : isDraft ? "保存并检测" : "保存更改"}</motion.button>
        </div>
      </form>

      <section className="mx-memoh-section mx-memoh-models">
        <header>
          <div><h3>模型目录</h3><p>从 Provider 拉取模型，再由注册表补全上下文、视觉和推理能力。</p></div>
          <button type="button" className="mx-memoh-secondary-button" onClick={refreshModels}><RefreshCw size={16} className={refreshing ? "is-spinning" : ""} />{refreshing ? "同步中" : "刷新模型"}</button>
        </header>
        {provider.models.length > 0 ? (
          <>
            <label className="mx-memoh-search"><Search size={17} /><span className="sr-only">搜索此连接的模型</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索此连接的模型" /></label>
            <div className="mx-memoh-model-list">{models.map((model) => <MemohModelRow key={model.id} model={model} />)}</div>
          </>
        ) : (
          <div className="mx-memoh-empty"><RefreshCw size={21} /><strong>保存后自动发现模型</strong><span>如果服务不支持模型列表，再允许手动填写 model name。</span></div>
        )}
        <p className="mx-memoh-capability-note">识别失败时才显示“高级覆盖”，默认不要求填写上下文长度或是否多模态。</p>
      </section>
    </motion.div>
  );
}

function MemohAddProviderDialog({ onClose, onSaved }: { onClose: () => void; onSaved: (message: string) => void }) {
  const panelRef = useRef<HTMLDivElement>(null);
  const nameRef = useRef<HTMLInputElement>(null);
  const [kind, setKind] = useState<MemohProviderKind>("api");
  const [showKey, setShowKey] = useState(false);
  const [manualModel, setManualModel] = useState(false);
  const isLogin = kind === "codex" || kind === "opencode";

  useEffect(() => {
    nameRef.current?.focus();
    function onKeyDown(event: globalThis.KeyboardEvent) {
      if (event.key === "Escape") onClose();
      if (event.key !== "Tab" || !panelRef.current) return;
      const focusable = Array.from(panelRef.current.querySelectorAll<HTMLElement>("button, input, select"));
      const first = focusable[0];
      const last = focusable.at(-1);
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last?.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first?.focus();
      }
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  function submit(event: FormEvent) {
    event.preventDefault();
    onSaved(isLogin ? "连接已保存，接下来完成安全登录。" : "连接已保存，API Key 已加密并开始发现模型。 ");
  }

  return (
    <motion.div className="mx-scrim" role="presentation" initial={{ opacity: 0 }} animate={{ opacity: 1 }} onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
      <motion.div ref={panelRef} className="mx-memoh-dialog" role="dialog" aria-modal="true" aria-labelledby="mx-memoh-add-title" initial={{ opacity: 0, y: 18, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }} transition={{ type: "spring", duration: 0.3, bounce: 0 }}>
        <header><div><span className="mx-overline">添加 Provider</span><h2 id="mx-memoh-add-title">新建模型连接</h2><p>先选择认证方式，只填写这一类连接真正需要的字段。</p></div><button type="button" className="mx-memoh-icon-button" onClick={onClose} aria-label="关闭"><X size={20} /></button></header>
        <form onSubmit={submit}>
          <fieldset className="mx-memoh-kind-picker">
            <legend>认证方式</legend>
            {[
              { id: "api", label: "API Key", detail: "Base URL + 密钥", icon: <KeyRound size={18} /> },
              { id: "codex", label: "Codex 登录", detail: "订阅账号授权", icon: <LogIn size={18} /> },
              { id: "opencode", label: "OpenCode", detail: "账号登录", icon: <Layers3 size={18} /> },
              { id: "custom", label: "自定义", detail: "选择兼容协议", icon: <Settings2 size={18} /> },
            ].map((item) => (
              <label key={item.id} className={kind === item.id ? "is-active" : ""}>
                <input type="radio" name="memoh-kind" value={item.id} checked={kind === item.id} onChange={() => { setKind(item.id as MemohProviderKind); setManualModel(false); }} />
                {item.icon}<span><strong>{item.label}</strong><small>{item.detail}</small></span><Check size={16} />
              </label>
            ))}
          </fieldset>

          <div className="mx-memoh-config-grid">
            <label className="mx-memoh-field-wide"><span>连接名称</span><input ref={nameRef} required placeholder={kind === "codex" ? "Codex 订阅" : kind === "opencode" ? "OpenCode Go 主账号" : "例如：DeepSeek 官方"} /></label>
            {isLogin ? (
              <div className="mx-memoh-account mx-memoh-field-wide"><span className="mx-memoh-account__icon"><ShieldCheck size={20} /></span><span><strong>无需填写 API Key</strong><small>保存后打开官方授权；完成后自动同步模型。</small></span></div>
            ) : (
              <>
                {kind === "custom" && <label><span>兼容协议</span><select defaultValue="OpenAI Compatible"><option>OpenAI Compatible</option><option>Anthropic</option><option>Google Generative AI</option></select></label>}
                <label className={kind === "api" ? "mx-memoh-field-wide" : ""}><span>Base URL</span><input required type="url" placeholder="https://api.example.com/v1" /></label>
                <label className="mx-secret mx-memoh-field-wide"><span>API Key</span><input required type={showKey ? "text" : "password"} placeholder="sk-…" /><button type="button" aria-label={showKey ? "隐藏 API Key" : "显示 API Key"} onClick={() => setShowKey((value) => !value)}>{showKey ? <EyeOff size={18} /> : <Eye size={18} />}</button></label>
                <label className="mx-memoh-discovery mx-memoh-field-wide"><input type="checkbox" checked={manualModel} onChange={(event) => setManualModel(event.target.checked)} /><span><strong>服务不支持模型发现</strong><small>仅此时手动填写 model name。</small></span></label>
                {manualModel && <><label><span>Model name</span><input required placeholder="deepseek-chat" /></label><label><span>默认思考强度</span><select defaultValue="high"><option value="low">低</option><option value="medium">中</option><option value="high">高</option><option value="xhigh">极高</option></select></label></>}
              </>
            )}
          </div>
          <p className="mx-form-note">上下文窗口、多模态和缓存统计字段由注册表及真实响应归一化，不进入首次配置表单。</p>
          <footer><button type="button" className="mx-text-button" onClick={onClose}>取消</button><motion.button type="submit" className="mx-primary-button" whileTap={{ scale: 0.96 }}>{isLogin ? "保存并登录" : "保存并检测"}<ArrowRight size={17} /></motion.button></footer>
        </form>
      </motion.div>
    </motion.div>
  );
}

function MemohSettingsPrototype() {
  const [selected, setSelected] = useState<MemohProvider | null>(null);
  const [query, setQuery] = useState("");
  const [dialogOpen, setDialogOpen] = useState(false);
  const [toast, setToast] = useState("");
  const addButtonRef = useRef<HTMLButtonElement>(null);
  const normalizedQuery = query.trim().toLowerCase();
  const connected = MEMOH_PROVIDERS.filter((provider) => `${provider.name} ${provider.description}`.toLowerCase().includes(normalizedQuery));
  const templates = MEMOH_TEMPLATES.filter((provider) => `${provider.name} ${provider.description}`.toLowerCase().includes(normalizedQuery));

  function showToast(message: string) {
    setDialogOpen(false);
    setToast(message);
    window.setTimeout(() => setToast(""), 2800);
    window.setTimeout(() => addButtonRef.current?.focus(), 0);
  }

  if (selected) {
    return (
      <section className="mx-settings mx-settings--dock mx-memoh-settings" aria-label="Memoh 风格模型配置原型">
        <MemohProviderDetail provider={selected} onBack={() => setSelected(null)} onSaved={showToast} />
        {createPortal(<div className="mx-toast-region" aria-live="polite" aria-atomic="true">{toast && <motion.div className="mx-toast" role="status" initial={{ opacity: 0, y: 16, scale: 0.96 }} animate={{ opacity: 1, y: 0, scale: 1 }}><Check size={18} /><span><strong>操作成功</strong><small>{toast}</small></span></motion.div>}</div>, document.body)}
      </section>
    );
  }

  return (
    <section className="mx-settings mx-settings--dock mx-memoh-settings" aria-label="Memoh 风格模型配置原型">
      <header className="mx-memoh-page-header">
        <div><span className="mx-overline">设置 · Provider</span><h2>模型连接</h2><p>像 Memoh 一样按 Provider 实例管理账号、密钥与模型目录。</p></div>
        <motion.button ref={addButtonRef} type="button" className="mx-primary-button" onClick={() => setDialogOpen(true)} whileTap={{ scale: 0.96 }}><Plus size={18} />添加连接</motion.button>
      </header>
      <label className="mx-memoh-search mx-memoh-search--page"><Search size={18} /><span className="sr-only">搜索 Provider</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索连接或模板" /></label>

      <section className="mx-memoh-gallery-section">
        <header><div><h3>已连接</h3><p>每套凭据都是独立实例，同一供应商可以添加多个账号。</p></div><span>{connected.length} 个</span></header>
        <div className="mx-memoh-gallery">{connected.map((provider) => <MemohProviderCard key={provider.id} provider={provider} onOpen={() => setSelected(provider)} />)}</div>
      </section>
      <section className="mx-memoh-gallery-section mx-memoh-gallery-section--templates">
        <header><div><h3>添加 Provider</h3><p>选择模板后预填协议与 Base URL，再完成认证。</p></div></header>
        <div className="mx-memoh-gallery">{templates.map((provider) => <MemohProviderCard key={provider.id} provider={provider} template onOpen={() => setSelected(provider)} />)}</div>
      </section>

      {createPortal(
        <>
          {dialogOpen && <MemohAddProviderDialog onClose={() => { setDialogOpen(false); window.setTimeout(() => addButtonRef.current?.focus(), 0); }} onSaved={showToast} />}
          <div className="mx-toast-region" aria-live="polite" aria-atomic="true">{toast && <motion.div className="mx-toast" role="status" initial={{ opacity: 0, y: 16, scale: 0.96 }} animate={{ opacity: 1, y: 0, scale: 1 }}><Check size={18} /><span><strong>连接已保存</strong><small>{toast}</small></span></motion.div>}</div>
        </>,
        document.body,
      )}
    </section>
  );
}

function GenericSettingsPrototype({ variant }: { variant: VariantId }) {
  const [formOpen, setFormOpen] = useState(false);
  const [toast, setToast] = useState(false);
  const title = variant === "command" ? "搜索与管理来源" : variant === "deck" ? "模型来源详情" : "模型与连接";

  function saved() {
    setFormOpen(false);
    setToast(true);
    window.setTimeout(() => setToast(false), 2600);
  }

  return (
    <section className={`mx-settings mx-settings--${variant}`} aria-label={`${variant} 模型配置原型`}>
      <header className="mx-settings__header">
        <div><span className="mx-overline">设置 · 模型</span><h2>{title}</h2><p>每套凭据是一个具名来源；同一 Provider 可以添加多个账号。</p></div>
        <motion.button type="button" className="mx-primary-button" onClick={() => setFormOpen(true)} whileTap={{ scale: 0.96 }}><Plus size={18} />添加来源</motion.button>
      </header>
      {variant === "command" && <label className="mx-settings-search"><Search size={18} /><span className="sr-only">搜索连接</span><input placeholder="搜索来源、模型或凭据类型" /><kbd>⌘ K</kbd></label>}
      {variant === "deck" && <aside className="mx-source-index"><strong>来源</strong><button type="button" className="is-active">全部 <span>4</span></button><button type="button">登录 <span>2</span></button><button type="button">API Key <span>2</span></button></aside>}
      <div className="mx-connection-list">
        {CONNECTIONS.map((connection) => <ConnectionRow key={connection.id} connection={connection} variant={variant} />)}
      </div>
      {variant === "rail" && <div className="mx-role-bindings"><SlidersHorizontal size={19} /><span><strong>角色默认值</strong><small>默认 · gpt-5.2-codex；快速 · deepseek-v4-flash</small></span><button type="button">调整</button></div>}

      {createPortal(
        <>
          {formOpen && <ConnectionForm variant={variant} onClose={() => setFormOpen(false)} onSaved={saved} />}
          <div className="mx-toast-region" aria-live="polite" aria-atomic="true">
            {toast && <motion.div className="mx-toast" role="status" initial={{ opacity: 0, y: 16, scale: 0.96 }} animate={{ opacity: 1, y: 0, scale: 1 }}><Check size={18} /><span><strong>来源已保存</strong><small>连接验证通过，模型信息已识别。</small></span></motion.div>}
          </div>
        </>,
        document.body,
      )}
    </section>
  );
}

function SettingsPrototype({ variant }: { variant: VariantId }) {
  return variant === "dock" ? <MemohSettingsPrototype /> : <GenericSettingsPrototype variant={variant} />;
}

/** Present five equally weighted interaction directions for model setup and per-turn selection. */
export function ModelExperienceShowcase() {
  const [variant, setVariant] = useState<VariantId>("dock");
  const [surface, setSurface] = useState<Surface>("chat");
  const active = VARIANTS.find((item) => item.id === variant) ?? VARIANTS[0];

  function onVariantKeyDown(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const next = (index + (event.key === "ArrowRight" ? 1 : -1) + VARIANTS.length) % VARIANTS.length;
    setVariant(VARIANTS[next].id);
    document.getElementById(`mx-variant-${VARIANTS[next].id}`)?.focus();
  }

  return (
    <main className="model-experience-showcase">
      <header className="mx-showcase-header">
        <div><span className="mx-overline">AKASHIC · INTERACTION STUDY</span><h1>模型体验，五个等权方向</h1><p>每版都覆盖多凭据配置、模糊表单、保存 Toast，以及聊天框上方的动态模型选择。</p></div>
        <div className="mx-surface-switch" role="tablist" aria-label="预览页面">
          <button type="button" role="tab" aria-selected={surface === "chat"} onClick={() => setSurface("chat")}>对话选模</button>
          <button type="button" role="tab" aria-selected={surface === "settings"} onClick={() => setSurface("settings")}>模型配置</button>
        </div>
      </header>
      <nav className="mx-variant-tabs" role="tablist" aria-label="五个等权设计版本">
        {VARIANTS.map((item, index) => (
          <button
            id={`mx-variant-${item.id}`}
            type="button"
            role="tab"
            aria-selected={variant === item.id}
            key={item.id}
            onClick={() => setVariant(item.id)}
            onKeyDown={(event) => onVariantKeyDown(event, index)}
          >
            <span>{item.number}</span><strong>{item.name}</strong><small>{item.note}</small>
          </button>
        ))}
      </nav>
      <div className="mx-version-caption"><span>方案 {active.number} / 05</span><strong>{active.name}</strong><p>{active.note}</p></div>
      <motion.div key={`${variant}-${surface}`} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.24 }}>
        {surface === "chat" ? <ChatPrototype variant={variant} /> : <SettingsPrototype variant={variant} />}
      </motion.div>
      <p className="mx-prototype-note">交互原型 · 使用演示数据，不保存或发送任何凭据</p>
    </main>
  );
}
