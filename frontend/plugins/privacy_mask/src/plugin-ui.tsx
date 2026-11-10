import { Eye, EyeOff } from "lucide-react";
import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import { HeaderMarkerActions, MarqueeText, render, type PluginUiDefinition } from "@akashic/chat-ui-v1";
import { acquirePrivacyMaskEngine, type PrivacyMaskEngine } from "./privacy-engine";
import "./plugin-ui.css";

function PrivacyMaskControl({ engine }: { engine: PrivacyMaskEngine }) {
  const snapshot = useSyncExternalStore(engine.subscribe, engine.getSnapshot, engine.getSnapshot);
  const [open, setOpen] = useState(false);
  // Esc 把焦点还给触发按钮；未开启时按钮要等焦点离开后再卸载，否则键盘焦点掉回 body。
  const [lingering, setLingering] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape" || event.defaultPrevented) return;
      const active = document.activeElement;
      const returnFocus = active === document.body || rootRef.current?.contains(active);
      if (returnFocus) setLingering(true);
      setOpen(false);
      if (returnFocus) triggerRef.current?.focus();
    };
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  const activeLabel = snapshot.enabled ? "已打码" : "截屏打码";
  // 未开启时顶栏不常驻：开关在侧栏底栏与 Alt+M；开启后才显示状态与关闭入口。
  if (!snapshot.enabled && !open && !lingering) return null;

  const statusText = snapshot.enabled
    ? snapshot.redactTitles && snapshot.blurContent
      ? "标题与内容均已打码"
      : snapshot.redactTitles
        ? "已打码导航与标题"
        : "已模糊全部对话内容"
    : "按 Alt+M 一键打码";

  return (
    <div className="header-marker privacy-mask-marker" ref={rootRef} data-privacy-ui="true">
      <button
        ref={triggerRef}
        type="button"
        className="header-marker__trigger"
        data-active={snapshot.enabled || undefined}
        aria-expanded={open}
        aria-label={`截屏隐私打码：${snapshot.enabled ? "已开启" : "未开启"}（Alt+M）`}
        title="截屏隐私打码（Alt+M）"
        onClick={() => setOpen((value) => !value)}
        onBlur={() => setLingering(false)}
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
                <span className="privacy-mask-scope__desc">侧栏项目名、会话名与顶栏标题化为骨架条</span>
              </span>
              <span className="privacy-mask-switch" aria-hidden="true">
                <span className="privacy-mask-switch__thumb" />
              </span>
            </button>

            <button
              type="button"
              role="switch"
              aria-checked={snapshot.enabled && snapshot.blurContent}
              className="privacy-mask-scope"
              onClick={() =>
                engine.setScope({
                  blurContent: !(snapshot.enabled && snapshot.blurContent),
                })
              }
            >
              <span className="privacy-mask-scope__copy">
                <span className="privacy-mask-scope__label">全部内容模糊</span>
                <span className="privacy-mask-scope__desc">消息、代码块与工具过程整体模糊</span>
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
