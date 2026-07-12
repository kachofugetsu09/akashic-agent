/// <reference path="../../types/akashic-dashboard.d.ts" />
import { useEffect, useState, type ReactElement } from "react";
import { Panel, Stack, api } from "@akashic/dashboard-ui";

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
  const date = new Date(String(value || ""));
  return Number.isNaN(date.getTime())
    ? String(value || "-")
    : `${date.getMonth() + 1}-${String(date.getDate()).padStart(2, "0")} ${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`;
}

function meterStatus(data: MeterData): { label: string; detail: string } {
  if (data.should_wake) {
    return { label: "已冲破", detail: "信息压力已越线，进入 LLM 最终判断" };
  }
  const ratio = data.threshold > 0
    ? (data.hazard_after + data.preference_pressure) / data.threshold
    : 0;
  if (ratio >= 0.75) return { label: "临界蓄压", detail: "再出现一条强相关信息就可能越线" };
  if (ratio >= 0.35) return { label: "持续积累", detail: "兴趣证据正在蓄水池中累积" };
  return { label: "低压稳定", detail: "当前信息不足以打扰用户" };
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

  if (!data) return <div className="meter-loading">正在读取压力传感器…</div>;
  const accumulated = percent(data.hazard_after, data.threshold);
  const pressure = Math.min(100 - accumulated, percent(data.preference_pressure, data.threshold));
  const total = data.hazard_after + data.preference_pressure;
  const status = meterStatus(data);
  const crossed = Boolean(data.should_wake);
  return <Stack className={`excitement-console ${crossed ? "is-crossed" : ""}`}>
    <header className="meter-header">
      <div><span>WAKE RESERVOIR</span><h2>兴奋水位</h2></div>
      <div className="meter-state"><i />{status.label}</div>
    </header>
    <Panel className="meter-machine">
      <div className="meter-chamber-wrap" role="img" aria-label={`当前水位 ${total.toFixed(2)}，阈值 ${data.threshold.toFixed(2)}`}>
        <div className="meter-burst"><i /><i /><i /><i /><i /></div>
        <div className="tank-rim" />
        <div className="meter-chamber">
          <div className="tank-zone overflow-zone"><span>冲破区</span></div>
          <div className="tank-zone warm-zone" />
          <div className="meter-threshold"><span>兴奋阈值 {data.threshold.toFixed(2)}</span></div>
          <div className="meter-fluid accumulated" style={{ height: `${accumulated}%` }} />
          <div className="meter-fluid pressure" style={{ bottom: `${accumulated}%`, height: `${pressure}%` }} />
          <div
            className={`water-surface ${data.preference_pressure > 0 ? "pressure-surface" : ""}`}
            style={{ bottom: `${Math.min(98, accumulated + pressure)}%` }}
          >
            <svg viewBox="0 0 180 18" preserveAspectRatio="none" aria-hidden="true">
              <path d="M0 10 Q22 2 45 10 T90 10 T135 10 T180 10 V18 H0 Z" />
            </svg>
          </div>
          <div className="water-bubbles"><i /><i /><i /></div>
        </div>
        <div className="meter-scale"><span>125%</span><span>阈值</span><span>50%</span><span>0</span></div>
      </div>
      <div className="meter-readout">
        <span>当前水位</span>
        <strong>{total.toFixed(2)}</strong>
        <em>/ {data.threshold.toFixed(2)}</em>
        <p>{status.detail}</p>
      </div>
    </Panel>
    <Stack className="meter-telemetry">
      <div><span><i className="tone-cobalt" />持续蓄积</span><strong>{data.hazard_after.toFixed(3)}</strong></div>
      <div><span><i className="tone-amber" />瞬时兴趣推力</span><strong>{data.preference_pressure.toFixed(3)}</strong></div>
      <div><span>未读内容</span><strong>{data.unread_count}</strong><small>{data.candidate_count} 条参与本轮</small></div>
      <div><span>最近判断</span><strong>{data.last_action || "尚未触发"}</strong><small>{shortTime(data.evaluated_at)}</small></div>
    </Stack>
    <footer className="meter-footnote">
      <span>越线只代表允许唤醒 LLM 判断，不等于一定推送。</span>
      <code>{data.driver_item_id || "NO ACTIVE DRIVER"}</code>
    </footer>
  </Stack>;
}

window.AkashicDashboard.registerPlugin({
  id: "wake-meter",
  label: "兴奋阈值",
  viewLabel: "wake pressure",
  layout: "workbench",
  rowKey: "id",
  columns: [],
  async getCount() {
    const data = await api<MeterData>("/api/dashboard/wake-proactive/meter");
    return data.unread_count;
  },
  async fetchPage() { return { items: [], total: 0 }; },
  Main: MeterPage,
});
