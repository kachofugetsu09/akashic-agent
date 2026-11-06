import { Eye, EyeOff } from "lucide-react";
import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import { HeaderMarkerActions, MarqueeText, render, type PluginUiDefinition } from "@akashic/chat-ui-v1";
import { acquirePrivacyMaskEngine, type PrivacyMaskEngine } from "./privacy-engine";
import "./plugin-ui.css";

function PrivacyMaskControl({ engine }: { engine: PrivacyMaskEngine }) {
  const snapshot = useSyncExternalStore(engine.subscribe, engine.getSnapshot, engine.getSnapshot);
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape" || event.defaultPrevented) return;
      setOpen(false);
      const active = document.activeElement;
      if (active === document.body || rootRef.current?.contains(active)) {
        triggerRef.current?.focus();
      }
    };
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  const activeLabel = snapshot.enabled
    ? snapshot.pickMode
      ? "点选打码中"
      : "已打码"
    : "截屏打码";

  const statusText = snapshot.enabled
    ? `词面 ${snapshot.piiCount} 处${snapshot.pickedCount > 0 ? ` · 点选 ${snapshot.pickedCount} 块` : ""}`
    : "按 Alt+M 一键模糊隐私内容";

  return (
    <div className="header-marker privacy-mask-marker" ref={rootRef} data-privacy-ui="true">
      <button
        ref={triggerRef}
        type="button"
        className="header-marker__trigger"
        data-active={snapshot.enabled || undefined}
        aria-expanded={open}
        aria-label={`截屏隐私打码：${snapshot.enabled ? "已开启" : "未开启"}（Alt+M）`}
        title={`截屏隐私打码（Alt+M）`}
        onClick={() => setOpen((value) => !value)}
      >
        {snapshot.enabled ? <EyeOff size={14} aria-hidden="true" /> : <Eye size={14} aria-hidden="true" />}
        <span>
          <MarqueeText still>{activeLabel}</MarqueeText>
        </span>
      </button>

      {open ? (
        <div
          className="header-marker__panel privacy-mask-panel"
          role="group"
          aria-label="截屏隐私打码"
          data-escape-layer=""
          data-privacy-ui="true"
        >
          <div className="privacy-mask-panel__head">
            <span className="privacy-mask-panel__title">截屏隐私打码</span>
            <kbd className="privacy-mask-panel__kbd">Alt+M</kbd>
          </div>

          <div className="privacy-mask-panel__list">
            <button
              type="button"
              role="switch"
              aria-checked={snapshot.enabled && snapshot.redactTitles}
              className="privacy-mask-scope"
              onClick={() =>
                engine.setScope({
                  redactTitles: !(snapshot.enabled && snapshot.redactTitles),
                })
              }
            >
              <span className="privacy-mask-scope__copy">
                <span className="privacy-mask-scope__label">导航与会话标题</span>
                <span className="privacy-mask-scope__desc">侧栏项目名、会话标题与顶栏化为骨架条</span>
              </span>
              <span className="privacy-mask-switch" aria-hidden="true">
                <span className="privacy-mask-switch__thumb" />
              </span>
            </button>

            <button
              type="button"
              role="switch"
              aria-checked={snapshot.enabled && snapshot.blurPii}
              className="privacy-mask-scope"
              onClick={() =>
                engine.setScope({
                  blurPii: !(snapshot.enabled && snapshot.blurPii),
                })
              }
            >
              <span className="privacy-mask-scope__copy">
                <span className="privacy-mask-scope__label">
                  敏感词面模糊
                  {snapshot.enabled && snapshot.blurPii && snapshot.piiCount > 0 ? (
                    <span className="privacy-mask-scope__count">{snapshot.piiCount} 处</span>
                  ) : null}
                </span>
                <span className="privacy-mask-scope__desc">邮箱 · 本机路径 · IP · 手机号 · 密钥</span>
              </span>
              <span className="privacy-mask-switch" aria-hidden="true">
                <span className="privacy-mask-switch__thumb" />
              </span>
            </button>

            <button
              type="button"
              role="switch"
              aria-checked={snapshot.enabled && snapshot.pickMode}
              className="privacy-mask-scope"
              onClick={() =>
                engine.setScope({
                  pickMode: !(snapshot.enabled && snapshot.pickMode),
                })
              }
            >
              <span className="privacy-mask-scope__copy">
                <span className="privacy-mask-scope__label">
                  点选区块打码
                  {snapshot.pickedCount > 0 ? (
                    <span className="privacy-mask-scope__count">已选 {snapshot.pickedCount} 块</span>
                  ) : null}
                </span>
                <span className="privacy-mask-scope__desc">单击任意段落、代码块或消息气泡单独模糊</span>
              </span>
              <span className="privacy-mask-switch" aria-hidden="true">
                <span className="privacy-mask-switch__thumb" />
              </span>
            </button>
          </div>

          <div className="privacy-mask-panel__footer">
            <HeaderMarkerActions>
              <button
                type="button"
                data-testid="privacy-mask-toggle-btn"
                onClick={() => engine.toggle()}
              >
                {snapshot.enabled ? "恢复正常显示" : "开启截屏打码"}
              </button>
              {snapshot.pickedCount > 0 ? (
                <button
                  type="button"
                  data-testid="privacy-mask-clear-picked-btn"
                  onClick={() => engine.clearPicked()}
                >
                  清空点选 ({snapshot.pickedCount})
                </button>
              ) : null}
            </HeaderMarkerActions>
            <span className="privacy-mask-panel__status">{statusText}</span>
          </div>
        </div>
      ) : null}
    </div>
  );
}

const definition: PluginUiDefinition = {
  slots: {
    "header.end": {
      mount(host) {
        const engine = acquirePrivacyMaskEngine();
        const unmount = render(host, <PrivacyMaskControl engine={engine} />);
        return () => {
          unmount();
          engine.release();
        };
      },
    },
  },
};

export default definition;
