/// <reference path="../../types/akashic-dashboard.d.ts" />

interface GraphNode {
  id?: string;
  x: number;
  y: number;
  r: number;
  c: string;
  t: string;
  g: number;
}

interface GraphEdgeObject {
  s: number;
  t: number;
  w: number;
  cc: number;
  sim: number;
}

type GraphEdge = [number, number, number, number, number];

interface GraphLegend {
  c: string;
  size: number;
  label: string;
}

interface GraphPayload {
  nodes: GraphNode[];
  edges: GraphEdgeObject[];
  legend: GraphLegend[];
  meta?: {
    missing?: boolean;
    stale?: boolean;
    rebuilding?: boolean;
    version?: string;
    elapsed_ms?: number;
  };
}

function ghEscape(value: string): string {
  return escapeHtml(String(value || ""));
}

function ghEdges(edges: GraphEdgeObject[]): GraphEdge[] {
  return edges.map((edge) => [edge.s, edge.t, edge.w, edge.cc, edge.sim]);
}

function renderAkashaGraph(container: HTMLElement): void {
  const previous = (container as HTMLElement & { __agDispose?: () => void }).__agDispose;
  if (previous) previous();

  container.innerHTML = `
    <div class="ag-html">
      <canvas id="c"></canvas>
      <div id="hud">
        <b>Akasha 真实记忆图</b><br>telegram 子图 · 数据透明版<br>
        <span id="stat">布局计算中...</span>
        <div style="position:relative;">
          <input id="search" placeholder="搜索记忆正文…">
          <div id="search_results"></div>
        </div>
        <div id="slider-container">
          <input type="range" id="cc_slider" min="1" max="10" value="2">
          <span style="font-size:11px">共现频次 &ge; <span id="cc_val" style="color:#fff;font-weight:bold;font-size:13px">2</span></span>
        </div>
        <div class="hint">拖拽平移 · 滚轮缩放 · 悬停看邻居 · 调滑块看引力</div>
      </div>
      <div id="leg"></div>
      <div id="tip"></div>
      <div id="node_detail"></div>
    </div>
  `;

  let disposed = false;
  (container as HTMLElement & { __agDispose?: () => void }).__agDispose = () => {
    disposed = true;
    if (pollTimer !== undefined) window.clearInterval(pollTimer);
    window.removeEventListener("resize", resize);
    window.removeEventListener("mouseup", onMouseUp);
    document.removeEventListener("click", onDocumentClick);
  };

  const root = container.querySelector<HTMLElement>(".ag-html")!;
  const cv = root.querySelector<HTMLCanvasElement>("#c")!;
  const ctx = cv.getContext("2d")!;
  const tip = root.querySelector<HTMLElement>("#tip")!;
  const stat = root.querySelector<HTMLElement>("#stat")!;
  const legEl = root.querySelector<HTMLElement>("#leg")!;
  const detailPanel = root.querySelector<HTMLElement>("#node_detail")!;
  const searchEl = root.querySelector<HTMLInputElement>("#search")!;
  const resEl = root.querySelector<HTMLElement>("#search_results")!;
  const slider = root.querySelector<HTMLInputElement>("#cc_slider")!;
  const ccVal = root.querySelector<HTMLElement>("#cc_val")!;

  let NODES: GraphNode[] = [];
  let EDGES: GraphEdge[] = [];
  let LEG: GraphLegend[] = [];
  let W = 1;
  let H = 1;
  let DPR = 1;
  let scale = 0.6;
  let tx = 0;
  let ty = 0;
  let ccThreshold = 2;
  let adj: number[][] = [];
  let hover = -1;
  let pinned = -1;
  let filter = "";
  let drag = false;
  let lx = 0;
  let ly = 0;
  let moved = false;
  let hlColor: string | null = null;
  let currentVersion = "";
  // eslint-disable-next-line prefer-const
  let pollTimer: number | undefined;
  let tweenFrame: number | undefined;
  let lockedColor: string | null = null;

  function flyTo(targetScale: number, targetTx: number, targetTy: number) {
    if (tweenFrame) cancelAnimationFrame(tweenFrame);
    const startScale = scale, startTx = tx, startTy = ty;
    let progress = 0;
    function step() {
      progress += 0.08;
      if (progress >= 1) {
        scale = targetScale; tx = targetTx; ty = targetTy;
        draw();
        return;
      }
      const ease = 1 - Math.pow(1 - progress, 3);
      scale = startScale + (targetScale - startScale) * ease;
      tx = startTx + (targetTx - startTx) * ease;
      ty = startTy + (targetTy - startTy) * ease;
      draw();
      tweenFrame = requestAnimationFrame(step);
    }
    step();
  }

  function flyToCommunity(c: string): void {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const n of NODES) {
      if (n.c === c) {
        if (n.x < minX) minX = n.x;
        if (n.x > maxX) maxX = n.x;
        if (n.y < minY) minY = n.y;
        if (n.y > maxY) maxY = n.y;
      }
    }
    if (minX === Infinity) return;
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const w = maxX - minX;
    const h = maxY - minY;
    
    let targetScale = scale;
    if (w > 0 && h > 0) {
      const s = Math.min((W - 120) / w, (H - 120) / h);
      targetScale = Math.max(0.2, Math.min(s, 2.5));
    } else {
      targetScale = Math.max(scale, 1.2);
    }
    
    const targetTx = W / 2 - cx * targetScale;
    const targetTy = H / 2 - cy * targetScale;
    flyTo(targetScale, targetTx, targetTy);
  }

  function resize(): void {
    if (disposed) return;
    const rect = root.getBoundingClientRect();
    W = Math.max(320, rect.width);
    H = Math.max(320, rect.height);
    DPR = window.devicePixelRatio || 1;
    cv.width = W * DPR;
    cv.height = H * DPR;
    cv.style.width = `${W}px`;
    cv.style.height = `${H}px`;
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    draw();
  }

  function fit(): void {
    const m = 60;
    const s = Math.min(W - 2 * m, H - 2 * m) / 1000;
    scale = Math.max(0.2, s);
    tx = (W - 1000 * scale) / 2;
    ty = (H - 1000 * scale) / 2;
  }

  function X(x: number): number { return x * scale + tx; }
  function Y(y: number): number { return y * scale + ty; }
  function invX(px: number): number { return (px - tx) / scale; }
  function invY(py: number): number { return (py - ty) / scale; }
  function activeId(): number { return hover >= 0 ? hover : pinned; }

  function recomputeAdj(): void {
    adj = NODES.map(() => []);
    for (const [a, b, , cc] of EDGES) {
      if (cc >= ccThreshold) {
        adj[a].push(b);
        adj[b].push(a);
      }
    }
  }

  function drawBase(): void {
    ctx.clearRect(0, 0, W, H);
    const act = activeId();
    const hl = new Set<number>();
    if (act >= 0) {
      hl.add(act);
      for (const n of adj[act] || []) hl.add(n);
    }
    const fil = filter.trim().toLowerCase();
    const match = fil
      ? new Set(NODES.map((n, i) => n.t.toLowerCase().includes(fil) ? i : -1).filter((i) => i >= 0))
      : null;

    ctx.lineWidth = Math.max(0.3, 0.5 * scale);
    for (const [a, b, w, cc] of EDGES) {
      if (cc < ccThreshold) continue;
      const on = act >= 0 && (a === act || b === act);
      if (act >= 0 && !on) continue;
      if (on) {
        const cross = NODES[a].g !== NODES[b].g;
        ctx.strokeStyle = cross ? "rgba(255,90,120,0.85)" : "rgba(150,200,255,.6)";
        ctx.lineWidth = cross ? Math.max(1.5, 2 * scale) : Math.max(0.6, scale);
        if (cross) {
          ctx.shadowBlur = 6 * scale;
          ctx.shadowColor = "rgba(255,90,120,0.9)";
        }
      } else {
        ctx.strokeStyle = `rgba(120,130,150,${Math.min(.22, .05 + w)})`;
        ctx.lineWidth = Math.max(0.3, 0.5 * scale);
        ctx.shadowBlur = 0;
      }
      ctx.beginPath();
      ctx.moveTo(X(NODES[a].x), Y(NODES[a].y));
      ctx.lineTo(X(NODES[b].x), Y(NODES[b].y));
      ctx.stroke();
      ctx.shadowBlur = 0;
    }

    if (act < 0) {
      ctx.strokeStyle = "rgba(110,120,140,.10)";
      ctx.lineWidth = Math.max(0.25, 0.35 * scale);
      ctx.beginPath();
      for (const [a, b, , cc] of EDGES) {
        if (cc < ccThreshold) continue;
        ctx.moveTo(X(NODES[a].x), Y(NODES[a].y));
        ctx.lineTo(X(NODES[b].x), Y(NODES[b].y));
      }
      ctx.stroke();
    }

    for (let i = 0; i < NODES.length; i += 1) {
      const n = NODES[i];
      let alpha = 1;
      let r = n.r * Math.max(0.6, Math.sqrt(scale));
      if ((adj[i]?.length || 0) === 0 && ccThreshold > 1 && !match) alpha = 0.08;
      if (act >= 0 && !hl.has(i)) alpha = 0.12;
      if (match && !match.has(i)) alpha = 0.06;
      if (match && match.has(i)) r *= 1.5;
      ctx.globalAlpha = alpha;
      ctx.fillStyle = n.c;
      if ((act >= 0 && hl.has(i)) || (match && match.has(i))) {
        ctx.shadowBlur = Math.max(6, r * 1.5);
        ctx.shadowColor = n.c;
      } else {
        ctx.shadowBlur = 0;
      }
      ctx.beginPath();
      ctx.arc(X(n.x), Y(n.y), r, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
      if ((act >= 0 && hl.has(i)) || (match && match.has(i))) {
        ctx.globalAlpha = 1;
        ctx.lineWidth = 1;
        ctx.strokeStyle = (act >= 0 && i !== act && NODES[i].g !== NODES[act].g) ? "#ff5a78" : "#fff";
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;

    if (act >= 0) {
      const n = NODES[act];
      ctx.fillStyle = "#fff";
      ctx.font = "bold 14px sans-serif";
      const text = n.t.length > 25 ? `${n.t.slice(0, 25)}...` : n.t;
      ctx.fillText(text, X(n.x) + 12, Y(n.y) - 12);
    }
  }

  function draw(): void {
    if (hlColor) {
      ctx.clearRect(0, 0, W, H);
      ctx.lineWidth = Math.max(0.6, scale);
      for (const [a, b, w, cc] of EDGES) {
        if (cc < ccThreshold) continue;
        const aOn = NODES[a].c === hlColor;
        const bOn = NODES[b].c === hlColor;
        if (!aOn && !bOn) continue;
        
        const cross = aOn !== bOn;
        if (cross) {
            ctx.strokeStyle = "rgba(255,90,120,0.6)";
            ctx.shadowBlur = 4 * scale;
            ctx.shadowColor = "rgba(255,90,120,0.8)";
        } else {
            ctx.strokeStyle = "rgba(150,200,255,0.4)";
            ctx.shadowBlur = 2 * scale;
            ctx.shadowColor = "rgba(150,200,255,0.6)";
        }
        ctx.beginPath();
        ctx.moveTo(X(NODES[a].x), Y(NODES[a].y));
        ctx.lineTo(X(NODES[b].x), Y(NODES[b].y));
        ctx.stroke();
        ctx.shadowBlur = 0;
      }
      for (let i = 0; i < NODES.length; i += 1) {
        const n = NODES[i];
        const on = n.c === hlColor;
        ctx.globalAlpha = on ? 1 : ((adj[i]?.length || 0) === 0 ? 0.03 : 0.08);
        ctx.fillStyle = n.c;
        if (on) {
          ctx.shadowBlur = n.r * 2 * Math.sqrt(scale);
          ctx.shadowColor = n.c;
        } else {
          ctx.shadowBlur = 0;
        }
        ctx.beginPath();
        ctx.arc(X(n.x), Y(n.y), n.r * Math.max(0.6, Math.sqrt(scale)) * (on ? 1.3 : 1), 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
      }
      ctx.globalAlpha = 1;
      return;
    }
    drawBase();
  }

  function badgeHTML(t: GraphNode & { w: number; cc: number; sim: number }): string {
    const simClass = t.sim > 0.65 ? "ag-badge-success" : (t.sim > 0.45 ? "ag-badge-warning" : "ag-badge-danger");
    const simText = t.sim > 0.65 ? "同义" : (t.sim > 0.45 ? "相关" : "潜意识跳跃");
    const simPct = `${(t.sim * 100).toFixed(0)}%`;
    return `<div style="display:flex;flex-wrap:wrap;margin-top:2px;">
      <span class="ag-badge ag-badge-outline">同框:${t.cc}次</span>
      <span class="ag-badge ag-badge-outline">引力:${t.w.toFixed(2)}</span>
      <span class="ag-badge ${simClass}">语义:${simPct} (${simText})</span>
    </div>`;
  }

  function updateDetailPanel(): void {
    if (pinned < 0) {
      detailPanel.style.display = "none";
      return;
    }
    const n = NODES[pinned];
    const neighbors: Array<{ id: number; w: number; cc: number; sim: number }> = [];
    for (const [a, b, w, cc, sim] of EDGES) {
      if (cc < ccThreshold) continue;
      if (a === pinned) neighbors.push({ id: b, w, cc, sim });
      if (b === pinned) neighbors.push({ id: a, w, cc, sim });
    }
    neighbors.sort((x, y) => y.w - x.w);
    const internal: Array<GraphNode & { w: number; cc: number; sim: number }> = [];
    const external: Array<GraphNode & { w: number; cc: number; sim: number }> = [];
    for (const nb of neighbors) {
      const target = { ...NODES[nb.id], w: nb.w, cc: nb.cc, sim: nb.sim };
      if (target.g === n.g) internal.push(target);
      else external.push(target);
    }

    let html = '<div style="font-size:13px;color:#8b9eb5;margin-bottom:6px;">选中记忆切片</div>';
    html += `<div style="font-size:14px;padding:12px;background:linear-gradient(135deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.02) 100%); border: 1px solid rgba(255,255,255,0.05); border-radius:8px; margin-bottom:16px;">${ghEscape(n.t)}</div>`;
    if (external.length > 0) {
      html += `<div style="color:#ff5a78;font-weight:bold;margin-bottom:4px;border-bottom:1px solid rgba(255,90,120,0.3);padding-bottom:6px;">思想跳跃 / 跨界走神 (${external.length})</div>`;
      html += '<div style="font-size:11px;color:#737a88;margin-bottom:12px;line-height:1.4;">溯源：分属不同的话题岛屿，但在特定时间点被你跨界关联。</div>';
      for (const t of external) {
        html += `<div class="detail-item" style="display:flex;gap:8px;align-items:flex-start;background:linear-gradient(90deg, rgba(255,255,255,0.05) 0%, transparent 100%); border-left: 2px solid ${t.c}; padding: 8px; margin-bottom: 8px;">
          <div style="display:flex;flex-direction:column;flex:1;"><span style="opacity:0.95">${ghEscape(t.t)}</span>${badgeHTML(t)}</div>
        </div>`;
      }
    }
    if (internal.length > 0) {
      html += `<div style="color:#96c8ff;font-weight:bold;margin-bottom:4px;margin-top:20px;border-bottom:1px solid rgba(150,200,255,0.3);padding-bottom:6px;">核心圈层 (${internal.length})</div>`;
      html += '<div style="font-size:11px;color:#737a88;margin-bottom:12px;line-height:1.4;">溯源：基于模块度算法，这些话题形成了高频同框的内聚孤岛。</div>';
      for (const t of internal) {
        html += `<div class="detail-item" style="display:flex;gap:8px;align-items:flex-start;background:linear-gradient(90deg, rgba(255,255,255,0.05) 0%, transparent 100%); border-left: 2px solid ${t.c}; padding: 8px; margin-bottom: 8px;">
          <div style="display:flex;flex-direction:column;flex:1;"><span>${ghEscape(t.t)}</span>${badgeHTML(t)}</div>
        </div>`;
      }
    }
    detailPanel.innerHTML = html;
    detailPanel.style.display = "block";
  }

  function pick(px: number, py: number): number {
    let best = -1;
    let bd = Number.POSITIVE_INFINITY;
    for (let i = 0; i < NODES.length; i += 1) {
      if ((adj[i]?.length || 0) === 0 && !filter) continue;
      const dx = X(NODES[i].x) - px;
      const dy = Y(NODES[i].y) - py;
      const d = dx * dx + dy * dy;
      const rr = Math.max(6, NODES[i].r * Math.sqrt(scale) + 4);
      if (d < rr * rr && d < bd) {
        bd = d;
        best = i;
      }
    }
    return best;
  }

  function onMouseUp(): void {
    drag = false;
    cv.classList.remove("drag");
  }

  function onDocumentClick(event: MouseEvent): void {
    const target = event.target as Node;
    if (!searchEl.contains(target) && !resEl.contains(target)) {
      resEl.style.display = "none";
    }
  }

  cv.addEventListener("mousedown", (event) => {
    if (tweenFrame) cancelAnimationFrame(tweenFrame);
    drag = true;
    moved = false;
    lx = event.clientX;
    ly = event.clientY;
    cv.classList.add("drag");
  });
  window.addEventListener("mouseup", onMouseUp);
  cv.addEventListener("mousemove", (event) => {
    const rect = cv.getBoundingClientRect();
    const mx = event.clientX - rect.left;
    const my = event.clientY - rect.top;
    if (drag) {
      tx += event.clientX - lx;
      ty += event.clientY - ly;
      lx = event.clientX;
      ly = event.clientY;
      moved = true;
      draw();
      tip.style.display = "none";
      return;
    }
    const h = pick(mx, my);
    if (h !== hover) {
      hover = h;
      draw();
    }
    if (h >= 0) {
      tip.style.display = "block";
      tip.style.left = `${mx + 14}px`;
      tip.style.top = `${my + 14}px`;
      tip.textContent = NODES[h].t.length > 40 ? `${NODES[h].t.slice(0, 40)}...` : NODES[h].t;
    } else {
      tip.style.display = "none";
    }
  });
  cv.addEventListener("click", (event) => {
    if (moved) return;
    if (lockedColor) {
      lockedColor = null;
      hlColor = null;
      legEl.querySelectorAll<HTMLElement>(".row").forEach(r => r.classList.remove("selected"));
    }
    const rect = cv.getBoundingClientRect();
    const h = pick(event.clientX - rect.left, event.clientY - rect.top);
    pinned = h === pinned ? -1 : h;
    if (pinned >= 0) {
      const n = NODES[pinned];
      const targetScale = Math.max(scale, 1.2);
      const targetTx = W / 2 - n.x * targetScale;
      const targetTy = H / 2 - n.y * targetScale;
      flyTo(targetScale, targetTx, targetTy);
    } else {
      draw();
    }
    updateDetailPanel();
  });
  cv.addEventListener("wheel", (event) => {
    if (tweenFrame) cancelAnimationFrame(tweenFrame);
    event.preventDefault();
    const rect = cv.getBoundingClientRect();
    const mx = event.clientX - rect.left;
    const my = event.clientY - rect.top;
    const f = event.deltaY < 0 ? 1.15 : 1 / 1.15;
    const wx = invX(mx);
    const wy = invY(my);
    scale *= f;
    tx = mx - wx * scale;
    ty = my - wy * scale;
    draw();
  }, { passive: false });

  slider.addEventListener("input", () => {
    ccThreshold = Number(slider.value);
    ccVal.textContent = String(ccThreshold);
    recomputeAdj();
    draw();
    updateDetailPanel();
  });

  searchEl.addEventListener("input", () => {
    filter = searchEl.value.trim().toLowerCase();
    if (!filter) {
      resEl.style.display = "none";
      draw();
      return;
    }
    const matches: number[] = [];
    for (let i = 0; i < NODES.length; i += 1) {
      if (NODES[i].t.toLowerCase().includes(filter)) matches.push(i);
    }
    if (matches.length > 0) {
      resEl.innerHTML = matches.slice(0, 30).map((i) => {
        const txt = NODES[i].t.length > 30 ? `${NODES[i].t.slice(0, 30)}...` : NODES[i].t;
        return `<div class="res-item" data-node="${i}">${ghEscape(txt)}</div>`;
      }).join("");
      resEl.querySelectorAll<HTMLElement>(".res-item").forEach((item) => {
        item.onclick = () => selectNode(Number(item.dataset.node));
      });
      resEl.style.display = "block";
    } else {
      resEl.innerHTML = '<div style="padding:8px 10px;color:#737a88;">无匹配项</div>';
      resEl.style.display = "block";
    }
    draw();
  });
  document.addEventListener("click", onDocumentClick);

  function selectNode(i: number): void {
    if (lockedColor) {
      lockedColor = null;
      hlColor = null;
      legEl.querySelectorAll<HTMLElement>(".row").forEach(r => r.classList.remove("selected"));
    }
    pinned = i;
    hover = -1;
    const n = NODES[i];
    const targetScale = Math.max(scale, 1.2);
    const targetTx = W / 2 - n.x * targetScale;
    const targetTy = H / 2 - n.y * targetScale;
    flyTo(targetScale, targetTx, targetTy);
    searchEl.value = "";
    filter = "";
    resEl.style.display = "none";
    updateDetailPanel();
  }

  function renderLegend(): void {
    legEl.innerHTML = '<div class="grab">社区</div><div class="content" style="padding-top:4px;"><div style="margin-bottom:12px;"><b style="color:#fff;font-size:13px;">社区主题</b> <span style="color:#737a88;">(悬停或点击锁定)</span></div>'
      + LEG.map((l) => `<div class="row" data-c="${ghEscape(l.c)}"><span class="dot" style="background:${ghEscape(l.c)}"></span><span><span style="color:#9aa3b5">[${l.size}]</span> ${ghEscape(l.label)}</span></div>`).join("")
      + "</div>";
    legEl.querySelectorAll<HTMLElement>(".row").forEach((row) => {
      row.addEventListener("mouseenter", () => {
        if (!lockedColor) {
          filter = "";
          hlColor = row.dataset.c || null;
          draw();
        }
      });
      row.addEventListener("mouseleave", () => {
        if (!lockedColor) {
          hlColor = null;
          draw();
        }
      });
      row.addEventListener("click", () => {
        const c = row.dataset.c || null;
        if (lockedColor === c) {
          lockedColor = null;
          hlColor = c;
        } else {
          lockedColor = c;
          hlColor = c;
          filter = "";
          pinned = -1;
          updateDetailPanel();
          if (c) flyToCommunity(c);
        }
        legEl.querySelectorAll<HTMLElement>(".row").forEach(r => r.classList.remove("selected"));
        if (lockedColor) row.classList.add("selected");
        draw();
      });
    });
  }

  function applyPayload(payload: GraphPayload, refit: boolean): void {
    if (payload.meta?.missing) {
      stat.textContent = payload.meta.rebuilding ? "快照后台生成中..." : "等待快照生成...";
      return;
    }
    const nextVersion = payload.meta?.version || "";
    if (!refit && nextVersion && nextVersion === currentVersion) {
      stat.textContent = `${NODES.length} 节点 · 共 ${EDGES.length} 候选边${payload.meta?.stale ? " · 后台刷新中" : ""}`;
      return;
    }
    currentVersion = nextVersion;
    NODES = payload.nodes || [];
    EDGES = ghEdges(payload.edges || []);
    LEG = payload.legend || [];
    ccThreshold = 2;
    slider.value = "2";
    ccVal.textContent = "2";
    recomputeAdj();
    renderLegend();
    stat.textContent = `${NODES.length} 节点 · 共 ${EDGES.length} 候选边${payload.meta?.stale ? " · 后台刷新中" : ""}`;
    if (refit) fit();
    resize();
  }

  async function load(refit: boolean): Promise<void> {
    if (refit) stat.textContent = "加载全景快照...";
    const payload = await api<GraphPayload>("/api/dashboard/akasha-graph/global");
    if (disposed) return;
    applyPayload(payload, refit);
  }

  window.addEventListener("resize", resize);
  resize();
  void load(true).catch((error) => {
    stat.textContent = error instanceof Error ? error.message : String(error);
  });
  pollTimer = window.setInterval(() => {
    void load(false).catch(() => undefined);
  }, 5000);
}

window.AkashicDashboard.registerPlugin({
  id: "akasha_graph",
  label: "Akasha Graph",
  viewLabel: "akasha graph",
  layout: "workbench",
  pageSize: 1,
  rowKey: "id",
  columns: [{ key: "id", label: "Graph", flex: true }],
  async getCount(): Promise<number | null> {
    try {
      const data = await api<GraphPayload>("/api/dashboard/akasha-graph/global");
      return data.nodes.length;
    } catch {
      return null;
    }
  },
  async fetchPage(): Promise<FetchPageResult> {
    return { items: [], total: 0 };
  },
  renderMain(container: HTMLElement): void {
    renderAkashaGraph(container);
  },
});
