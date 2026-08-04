/// <reference path="../../types/akashic-dashboard.d.ts" />
import { useEffect, useState, type ReactElement } from "react";
import { api } from "@akashic/dashboard-ui";

interface MeterData {
  session_key: string;
  hazard_after: number;
  preference_pressure: number;
  threshold: number;
  evidence: number;
  rate: number;
  driver_item_id: string;
  candidate_count: number;
  unread_count: number;
  should_wake: number;
  evaluated_at: string | null;
  last_action: string | null;
  last_action_at: string | null;
}

function shortTime(value: unknown): string {
  if (!value) return "—";
  const date = new Date(String(value));
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(date);
}

function meterStatus(data: MeterData): { label: string; detail: string } {
  if (data.should_wake) {
    return { label: "已冲破", detail: "信息压力已越线，进入 LLM 最终判断。" };
  }
  const ratio = data.threshold > 0
    ? (data.hazard_after + data.preference_pressure) / data.threshold
    : 0;
  if (ratio >= 0.75) {
    return { label: "接近阈值", detail: "再出现一条强相关信息就可能进入最终判断。" };
  }
  if (ratio >= 0.35) {
    return { label: "正在累积", detail: "相关信息正在提高主动唤醒压力。" };
  }
  return { label: "低压稳定", detail: "当前信息不足以打扰用户。" };
}

function percent(value: number, threshold: number): number {
  if (threshold <= 0) return 0;
  return Math.min(100, Math.max(0, value / (threshold * 1.25) * 100));
}

function MeterPage(): ReactElement {
  const [data, setData] = useState<MeterData | null>(null);
  useEffect(() => {
    let active = true;
    const refresh = async () => {
      const next = await api<MeterData>("/api/dashboard/wake-proactive/meter");
      if (active) setData(next);
    };
    void refresh();
    const timer = window.setInterval(() => void refresh(), 15_000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  if (!data) {
    return <div className="meter-loading">正在读取压力传感器…</div>;
  }

  const accumulated = percent(data.hazard_after, data.threshold);
  const pressure = Math.min(
    100 - accumulated,
    percent(data.preference_pressure, data.threshold),
  );
  const total = data.hazard_after + data.preference_pressure;
  const status = meterStatus(data);
  const crossed = Boolean(data.should_wake);
  const ratio = data.threshold > 0 ? total / data.threshold : 0;

  return (
    <main className={`excitement-console${crossed ? " is-crossed" : ""}`} aria-labelledby="meter-title">
      <header className="meter-header">
        <div>
          <span>主动唤醒压力</span>
          <h2 id="meter-title">是否值得现在打扰用户</h2>
        </div>
        <div className="meter-state" aria-live="polite">
          <i aria-hidden="true" />
          {status.label}
        </div>
      </header>

      <section className="meter-machine" aria-label="当前压力与阈值">
        <div className="meter-readout">
          <span>当前压力</span>
          <strong>{total.toFixed(2)}</strong>
          <em>/ {data.threshold.toFixed(2)}</em>
          <p>{status.detail}</p>
        </div>
        <div className="pressure-visual">
          <div className="pressure-scale"><span>0</span><span>阈值 {data.threshold.toFixed(2)}</span><span>125%</span></div>
          <div className="pressure-track" role="img" aria-label={`当前压力 ${total.toFixed(2)}，阈值 ${data.threshold.toFixed(2)}`}>
            <i className="pressure-segment is-accumulated" style={{ width: `${accumulated}%` }} />
            <i className="pressure-segment is-instant" style={{ left: `${accumulated}%`, width: `${pressure}%` }} />
            <b className="pressure-threshold" />
          </div>
          <div className="pressure-legend">
            <span><i className="is-accumulated" />持续积累 {data.hazard_after.toFixed(3)}</span>
            <span><i className="is-instant" />瞬时兴趣 {data.preference_pressure.toFixed(3)}</span>
            <strong>{Math.round(ratio * 100)}%</strong>
          </div>
        </div>
      </section>

      <div className="meter-telemetry">
        <div>
          <span><i className="tone-cobalt" aria-hidden="true" />持续蓄积</span>
          <strong>{data.hazard_after.toFixed(3)}</strong>
        </div>
        <div>
          <span><i className="tone-amber" aria-hidden="true" />瞬时兴趣推力</span>
          <strong>{data.preference_pressure.toFixed(3)}</strong>
        </div>
        <div>
          <span>未读内容</span>
          <strong>{data.unread_count}</strong>
          <small>{data.candidate_count} 条参与本轮</small>
        </div>
        <div>
          <span>最近计算</span>
          <strong title={String(data.evaluated_at || "")}>{shortTime(data.evaluated_at)}</strong>
          <small>{data.candidate_count} 条已计算</small>
        </div>
        <div>
          <span>最近 LLM 判断</span>
          <strong>{data.last_action || "尚未触发"}</strong>
          <small title={String(data.last_action_at || "")}>{shortTime(data.last_action_at)}</small>
        </div>
      </div>

      <footer className="meter-footnote">
        <span>越线只代表允许唤醒 LLM 判断，不等于一定推送。</span>
        <code title={data.driver_item_id || "NO ACTIVE DRIVER"}>
          {data.driver_item_id || "NO ACTIVE DRIVER"}
        </code>
      </footer>
    </main>
  );
}

window.AkashicDashboard.registerPlugin({
  id: "wake-meter",
  label: "兴奋阈值",
  viewLabel: "兴奋阈值",
  layout: "workbench",
  rowKey: "id",
  columns: [],
  async getCount() {
    const data = await api<MeterData>("/api/dashboard/wake-proactive/meter");
    return data.unread_count;
  },
  async fetchPage() {
    return { items: [], total: 0 };
  },
  Main: MeterPage,
});
