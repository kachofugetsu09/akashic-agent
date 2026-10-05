import { Cpu, Gauge, MessageSquare, Palette } from "lucide-react";
import type { KeyboardEvent, ReactNode } from "react";
import { akashicBrandIcon } from "./akashic-brand";

export interface ChatProductBandProps {
  chatReady: boolean;
  themeLabel: string;
  onCycleTheme: () => void;
}

interface BandItem {
  id: string;
  label: string;
  icon: ReactNode;
  href?: string;
  disabled?: boolean;
}

/** 顶部目的地：普通横排链接导航，视觉与工作台顶栏共用 theme/src/product-band.css。 */
export function ChatProductBand({ chatReady, themeLabel, onCycleTheme }: ChatProductBandProps) {
  const items: BandItem[] = [
    { id: "chat", label: "对话", icon: <MessageSquare size={16} aria-hidden="true" />, href: "/chat" },
    {
      id: "workbench",
      label: "工作台",
      icon: <Gauge size={16} aria-hidden="true" />,
      href: chatReady ? "/" : undefined,
      disabled: !chatReady,
    },
    { id: "models", label: "模型", icon: <Cpu size={16} aria-hidden="true" />, href: "/#models" },
  ];

  const onKeyDown = (event: KeyboardEvent<HTMLElement>) => {
    if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
    const links = [...event.currentTarget.querySelectorAll<HTMLElement>(".product-band__item:not(.is-disabled)")];
    const current = links.indexOf(event.target as HTMLElement);
    if (current < 0) return;
    event.preventDefault();
    const next = (current + (event.key === "ArrowRight" ? 1 : -1) + links.length) % links.length;
    links[next]?.focus();
  };

  return (
    <header className="product-band" aria-label="Akashic 主导航">
      <div className="product-band__brand" title="Akashic">
        <span
          className="product-band__mark"
          style={{ WebkitMaskImage: `url(${akashicBrandIcon})`, maskImage: `url(${akashicBrandIcon})` }}
          aria-hidden="true"
        />
        <strong>Akashic</strong>
      </div>
      <nav className="product-band__nav" aria-label="主要功能" onKeyDown={onKeyDown}>
        <div className="product-band__track">
          {items.map((item) => {
            const current = item.id === "chat";
            const className = `product-band__item${item.disabled ? " is-disabled" : ""}`;
            if (item.disabled || !item.href) {
              return (
                <span key={item.id} className={className} aria-disabled={item.disabled || undefined}>
                  {item.icon}
                  <span>{item.label}</span>
                </span>
              );
            }
            return (
              <a
                key={item.id}
                className={className}
                href={item.href}
                tabIndex={current ? 0 : -1}
                aria-current={current ? "page" : undefined}
              >
                {item.icon}
                <span>{item.label}</span>
              </a>
            );
          })}
        </div>
      </nav>
      <div className="product-band__footer">
        <button type="button" className="product-band__action" onClick={onCycleTheme} title={`主题 · ${themeLabel}`}>
          <Palette size={16} aria-hidden="true" />
          <span className="product-band__theme-label">{themeLabel}</span>
        </button>
      </div>
    </header>
  );
}
