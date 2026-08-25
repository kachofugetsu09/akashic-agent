import { BookOpenText, Bot, Gauge, Palette, SlidersHorizontal } from "lucide-react";
import { akashicBrandIcon } from "./akashic-brand";

export interface ChatProductBandProps {
  surface: "chat" | "runtime";
  chatReady: boolean;
  themeLabel: string;
  onOpenRuntime: () => void;
  onCycleTheme: () => void;
}

/** Standalone `/chat` top product band — mirrors dashboard L-shape destinations. */
export function ChatProductBand({
  surface,
  chatReady,
  themeLabel,
  onOpenRuntime,
  onCycleTheme,
}: ChatProductBandProps) {
  const dashboardHref = chatReady ? "/" : undefined;

  return (
    <header className="chat-product-band" aria-label="Akashic 主导航">
      <div className="chat-product-band__brand" title="Akashic">
        <span
          className="chat-product-band__mark"
          style={{ WebkitMaskImage: `url(${akashicBrandIcon})`, maskImage: `url(${akashicBrandIcon})` }}
          aria-hidden="true"
        />
        <strong>Akashic</strong>
      </div>
      <nav className="chat-product-band__nav" aria-label="主要功能">
        <a
          className={`chat-product-band__item ${surface === "chat" ? "is-active" : ""}`}
          href="/chat"
          aria-current={surface === "chat" ? "page" : undefined}
        >
          <Bot size={16} aria-hidden="true" />
          <span>对话</span>
        </a>
        {dashboardHref ? (
          <a className="chat-product-band__item" href={dashboardHref}>
            <Gauge size={16} aria-hidden="true" />
            <span>工作台</span>
          </a>
        ) : (
          <span className="chat-product-band__item is-disabled" aria-disabled="true">
            <Gauge size={16} aria-hidden="true" />
            <span>工作台</span>
          </span>
        )}
        <button
          type="button"
          className={`chat-product-band__item ${surface === "runtime" ? "is-active" : ""}`}
          aria-current={surface === "runtime" ? "page" : undefined}
          onClick={onOpenRuntime}
        >
          <BookOpenText size={16} aria-hidden="true" />
          <span>知识与运行</span>
        </button>
        <a className="chat-product-band__item" href="/settings">
          <SlidersHorizontal size={16} aria-hidden="true" />
          <span>模型</span>
        </a>
      </nav>
      <div className="chat-product-band__footer">
        <button type="button" className="chat-product-band__item" onClick={onCycleTheme} title={`主题 · ${themeLabel}`}>
          <Palette size={16} aria-hidden="true" />
          <span className="chat-product-band__theme-label">{themeLabel}</span>
        </button>
      </div>
    </header>
  );
}
