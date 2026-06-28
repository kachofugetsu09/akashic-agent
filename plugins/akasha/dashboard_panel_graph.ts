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

interface CommunityView {
  g: number;
  c: string;
  size: number;
  label: string;
  x: number;
  y: number;
  r: number;
}

interface SubIslandView {
  id: number;
  parent: number;
  c: string;
  size: number;
  x: number;
  y: number;
  r: number;
}

interface IslandLink {
  ax: number;
  ay: number;
  bx: number;
  by: number;
  color: string;
  strength: number;
  cross: boolean;
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
        <div class="hint">远景看大岛 · 中景看子岛 · 近景看联想根系</div>
      </div>
      <div id="leg"></div>
      <div id="loading_card" class="ag-loading-card">
        <div class="ag-loading-kicker">Akasha Graph</div>
        <div class="ag-loading-title">正在生成全量记忆图</div>
        <div class="ag-loading-copy">第一次进入需要构建 snapshot，之后会直接读取缓存。</div>
        <div class="ag-loading-track"><div id="loading_bar"></div></div>
        <div id="loading_note" class="ag-loading-note">启动后台布局任务...</div>
      </div>
      <div id="tip"></div>
      <div id="node_detail"></div>
    </div>
  `;

  let disposed = false;
  (container as HTMLElement & { __agDispose?: () => void }).__agDispose = () => {
    disposed = true;
    if (pollTimer !== undefined) window.clearInterval(pollTimer);
    if (tweenFrame !== undefined) cancelAnimationFrame(tweenFrame);
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
  const loadingCard = root.querySelector<HTMLElement>("#loading_card")!;
  const loadingBar = root.querySelector<HTMLElement>("#loading_bar")!;
  const loadingNote = root.querySelector<HTMLElement>("#loading_note")!;
  const detailPanel = root.querySelector<HTMLElement>("#node_detail")!;
  const searchEl = root.querySelector<HTMLInputElement>("#search")!;
  const resEl = root.querySelector<HTMLElement>("#search_results")!;
  const slider = root.querySelector<HTMLInputElement>("#cc_slider")!;
  const ccVal = root.querySelector<HTMLElement>("#cc_val")!;

  let NODES: GraphNode[] = [];
  let EDGES: GraphEdge[] = [];
  let LEG: GraphLegend[] = [];
  let COMMUNITIES: CommunityView[] = [];
  let SUB_ISLANDS: SubIslandView[] = [];
  let childByNode: number[] = [];
  let nodeIsRepresentative: boolean[] = [];
  let parentLinks: IslandLink[] = [];
  let childLinks: IslandLink[] = [];
  let W = 1;
  let H = 1;
  let DPR = 1;
  let scale = 0.6;
  let tx = 0;
  let ty = 0;
  let ccThreshold = 2;
  let adj: number[][] = [];
  let adjEdges: GraphEdge[][] = [];
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
  let hlInternalPath = new Path2D();
  let hlCrossPath = new Path2D();
  const globalEdgeScaleLimit = 1.2;
  const loadingStartedAt = Date.now();
  let loadingTick = 0;

  function updateHlPaths() {
    hlInternalPath = new Path2D();
    hlCrossPath = new Path2D();
    if (!hlColor) return;
    for (const [a, b, , cc] of EDGES) {
      if (cc < ccThreshold) continue;
      const nA = NODES[a], nB = NODES[b];
      const aOn = nA.c === hlColor;
      const bOn = nB.c === hlColor;
      if (!aOn && !bOn) continue;
      if (aOn !== bOn) {
        hlCrossPath.moveTo(nA.x, nA.y);
        hlCrossPath.lineTo(nB.x, nB.y);
      } else {
        hlInternalPath.moveTo(nA.x, nA.y);
        hlInternalPath.lineTo(nB.x, nB.y);
      }
    }
  }

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

    const targetScale = w > 0 && h > 0
      ? Math.max(0.2, Math.min(Math.min((W - 120) / w, (H - 120) / h), 2.5))
      : Math.max(scale, 1.2);

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
  function activeId(): number { return pinned; }

  function colorAlpha(color: string, alpha: number): string {
    const safeAlpha = Math.max(0, Math.min(1, alpha));
    if (color.startsWith("hsl(")) return color.replace("hsl(", "hsla(").replace(")", `,${safeAlpha})`);
    const hex = color.startsWith("#") ? color.slice(1) : "";
    if (hex.length === 6) {
      const r = parseInt(hex.slice(0, 2), 16);
      const g = parseInt(hex.slice(2, 4), 16);
      const b = parseInt(hex.slice(4, 6), 16);
      return `rgba(${r},${g},${b},${safeAlpha})`;
    }
    return color;
  }

  function rebalanceFractalSpacing(): void {
    const groups = new Map<number, { count: number; sx: number; sy: number }>();
    for (const n of NODES) {
      const old = groups.get(n.g);
      if (old) {
        old.count += 1;
        old.sx += n.x;
        old.sy += n.y;
      } else {
        groups.set(n.g, { count: 1, sx: n.x, sy: n.y });
      }
    }
    const centers = new Map<number, { x: number; y: number; spread: number }>();
    for (const [g, item] of groups.entries()) {
      centers.set(g, {
        x: item.sx / item.count,
        y: item.sy / item.count,
        spread: item.count >= 180 ? 1.55 : (item.count >= 70 ? 1.42 : 1.26),
      });
    }
    const world = { x: 500, y: 500 };
    NODES = NODES.map((node) => {
      const center = centers.get(node.g);
      if (!center) return node;
      const cx = world.x + (center.x - world.x) * 0.74;
      const cy = world.y + (center.y - world.y) * 0.74;
      return {
        ...node,
        x: cx + (node.x - center.x) * center.spread,
        y: cy + (node.y - center.y) * center.spread,
      };
    });
  }

  function recomputeAdj(): void {
    adj = NODES.map(() => []);
    adjEdges = NODES.map(() => []);
    for (const edge of EDGES) {
      const [a, b, , cc] = edge;
      if (cc >= ccThreshold) {
        adj[a].push(b);
        adj[b].push(a);
        adjEdges[a].push(edge);
        adjEdges[b].push(edge);
      }
    }
    recomputeLinks();
  }

  function recomputeCommunities(): void {
    const groups = new Map<number, {
      c: string;
      label: string;
      count: number;
      sx: number;
      sy: number;
      minX: number;
      minY: number;
      maxX: number;
      maxY: number;
    }>();
    const nodesByGroup = new Map<number, number[]>();
    for (let i = 0; i < NODES.length; i += 1) {
      const n = NODES[i];
      const bucket = nodesByGroup.get(n.g);
      if (bucket) bucket.push(i);
      else nodesByGroup.set(n.g, [i]);
      const old = groups.get(n.g);
      const label = LEG[n.g]?.label || `社区 ${n.g}`;
      if (!old) {
        groups.set(n.g, {
          c: n.c,
          label,
          count: 1,
          sx: n.x,
          sy: n.y,
          minX: n.x,
          minY: n.y,
          maxX: n.x,
          maxY: n.y,
        });
        continue;
      }
      old.count += 1;
      old.sx += n.x;
      old.sy += n.y;
      old.minX = Math.min(old.minX, n.x);
      old.minY = Math.min(old.minY, n.y);
      old.maxX = Math.max(old.maxX, n.x);
      old.maxY = Math.max(old.maxY, n.y);
    }
    COMMUNITIES = [...groups.entries()].map(([g, item]) => {
      const x = item.sx / item.count;
      const y = item.sy / item.count;
      const dx = Math.max(x - item.minX, item.maxX - x);
      const dy = Math.max(y - item.minY, item.maxY - y);
      return {
        g,
        c: item.c,
        size: item.count,
        label: item.label,
        x,
        y,
        r: Math.max(18, Math.hypot(dx, dy) + 10 + Math.sqrt(item.count) * 1.8),
      };
    }).sort((a, b) => b.size - a.size);
    recomputeSubIslands(nodesByGroup);
  }

  function recomputeSubIslands(nodesByGroup: Map<number, number[]>): void {
    SUB_ISLANDS = [];
    childByNode = NODES.map(() => -1);
    nodeIsRepresentative = NODES.map(() => false);
    for (const comm of COMMUNITIES) {
      const indexes = nodesByGroup.get(comm.g) || [];
      const parts = splitSubIslands(indexes, comm);
      for (const part of parts) {
        const id = SUB_ISLANDS.length;
        let sx = 0, sy = 0;
        for (const nodeIndex of part) {
          sx += NODES[nodeIndex].x;
          sy += NODES[nodeIndex].y;
        }
        const x = sx / part.length;
        const y = sy / part.length;
        let far = 0;
        for (const nodeIndex of part) {
          const n = NODES[nodeIndex];
          far = Math.max(far, Math.hypot(n.x - x, n.y - y));
          childByNode[nodeIndex] = id;
        }
        const sub = {
          id,
          parent: comm.g,
          c: comm.c,
          size: part.length,
          x,
          y,
          r: Math.max(7, far + 4 + Math.sqrt(part.length) * 0.9),
        };
        SUB_ISLANDS.push(sub);
        markRepresentatives(part, sub);
      }
    }
  }

  function splitSubIslands(indexes: number[], comm: CommunityView): number[][] {
    if (indexes.length < 32) return [indexes];
    const k = Math.max(2, Math.min(8, Math.round(Math.sqrt(indexes.length) / 2.7)));
    const centers: Array<{ x: number; y: number }> = [];
    let first = indexes[0];
    let far = -1;
    for (const nodeIndex of indexes) {
      const n = NODES[nodeIndex];
      const d = Math.hypot(n.x - comm.x, n.y - comm.y);
      if (d > far) {
        far = d;
        first = nodeIndex;
      }
    }
    centers.push({ x: NODES[first].x, y: NODES[first].y });
    while (centers.length < k) {
      let next = indexes[0];
      let best = -1;
      for (const nodeIndex of indexes) {
        const n = NODES[nodeIndex];
        let nearest = Infinity;
        for (const c of centers) nearest = Math.min(nearest, Math.hypot(n.x - c.x, n.y - c.y));
        if (nearest > best) {
          best = nearest;
          next = nodeIndex;
        }
      }
      centers.push({ x: NODES[next].x, y: NODES[next].y });
    }
    let buckets: number[][] = [];
    for (let iter = 0; iter < 5; iter += 1) {
      buckets = Array.from({ length: centers.length }, () => []);
      for (const nodeIndex of indexes) {
        const n = NODES[nodeIndex];
        let pick = 0;
        let best = Infinity;
        for (let i = 0; i < centers.length; i += 1) {
          const c = centers[i];
          const d = Math.hypot(n.x - c.x, n.y - c.y);
          if (d < best) {
            best = d;
            pick = i;
          }
        }
        buckets[pick].push(nodeIndex);
      }
      for (let i = 0; i < buckets.length; i += 1) {
        if (!buckets[i].length) continue;
        let sx = 0, sy = 0;
        for (const nodeIndex of buckets[i]) {
          sx += NODES[nodeIndex].x;
          sy += NODES[nodeIndex].y;
        }
        centers[i] = { x: sx / buckets[i].length, y: sy / buckets[i].length };
      }
    }
    return buckets.filter((bucket) => bucket.length > 0);
  }

  function markRepresentatives(indexes: number[], sub: SubIslandView): void {
    const limit = Math.min(5, Math.max(1, Math.round(Math.sqrt(indexes.length) / 3)));
    const ranked = [...indexes].sort((a, b) => {
      const na = NODES[a], nb = NODES[b];
      return Math.hypot(na.x - sub.x, na.y - sub.y) - Math.hypot(nb.x - sub.x, nb.y - sub.y);
    });
    for (const nodeIndex of ranked.slice(0, limit)) nodeIsRepresentative[nodeIndex] = true;
  }

  function recomputeLinks(): void {
    const commByGroup = new Map(COMMUNITIES.map((comm) => [comm.g, comm]));
    const parentAgg = new Map<string, { a: number; b: number; cc: number; w: number }>();
    const childAgg = new Map<string, { a: number; b: number; cc: number; w: number }>();
    for (const [a, b, w, cc] of EDGES) {
      if (cc < ccThreshold) continue;
      const na = NODES[a], nb = NODES[b];
      if (na.g !== nb.g && cc >= 5) {
        addAgg(parentAgg, na.g, nb.g, cc, w);
      }
      const ca = childByNode[a], cb = childByNode[b];
      if (ca >= 0 && cb >= 0 && ca !== cb && cc >= 3) {
        const sameParent = SUB_ISLANDS[ca]?.parent === SUB_ISLANDS[cb]?.parent;
        if (sameParent || cc >= 8) addAgg(childAgg, ca, cb, cc, w);
      }
    }
    parentLinks = [...parentAgg.values()]
      .map((item) => {
        const a = commByGroup.get(item.a);
        const b = commByGroup.get(item.b);
        if (!a || !b) return null;
        return linkFrom(a, b, item.cc, item.w, true);
      })
      .filter((item): item is IslandLink => item !== null)
      .sort((a, b) => b.strength - a.strength)
      .slice(0, 240);
    childLinks = [...childAgg.values()]
      .map((item) => {
        const a = SUB_ISLANDS[item.a];
        const b = SUB_ISLANDS[item.b];
        if (!a || !b) return null;
        return linkFrom(a, b, item.cc, item.w, a.parent !== b.parent);
      })
      .filter((item): item is IslandLink => item !== null)
      .sort((a, b) => b.strength - a.strength)
      .slice(0, 1100);
  }

  function addAgg(
    map: Map<string, { a: number; b: number; cc: number; w: number }>,
    a: number,
    b: number,
    cc: number,
    w: number,
  ): void {
    const x = Math.min(a, b);
    const y = Math.max(a, b);
    const key = `${x}:${y}`;
    const old = map.get(key);
    if (old) {
      old.cc += cc;
      old.w = Math.max(old.w, w);
    } else {
      map.set(key, { a: x, b: y, cc, w });
    }
  }

  function linkFrom(
    a: CommunityView | SubIslandView,
    b: CommunityView | SubIslandView,
    cc: number,
    w: number,
    cross: boolean,
  ): IslandLink {
    return {
      ax: a.x,
      ay: a.y,
      bx: b.x,
      by: b.y,
      color: cross ? "#d7e4ff" : a.c,
      strength: Math.sqrt(cc) * Math.max(0.04, w),
      cross,
    };
  }

  function drawIslands(act: number): void {
    ctx.save();
    ctx.translate(tx, ty);
    ctx.scale(scale, scale);
    for (let i = COMMUNITIES.length - 1; i >= 0; i -= 1) {
      const comm = COMMUNITIES[i];
      const active = act >= 0 && NODES[act]?.g === comm.g;
      const alpha = active ? 0.22 : (scale < 0.9 ? 0.12 : 0.06);
      const radius = comm.r + (scale < 0.9 ? 10 : 4);
      const gradient = ctx.createRadialGradient(
        comm.x,
        comm.y,
        Math.max(4, radius * 0.12),
        comm.x,
        comm.y,
        radius,
      );
      gradient.addColorStop(0, colorAlpha(comm.c, alpha));
      gradient.addColorStop(0.72, colorAlpha(comm.c, alpha * 0.32));
      gradient.addColorStop(1, colorAlpha(comm.c, 0));
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(comm.x, comm.y, radius, 0, Math.PI * 2);
      ctx.fill();
    }
    if (scale > 0.68) {
      for (const sub of SUB_ISLANDS) {
        const active = act >= 0 && childByNode[act] === sub.id;
        const alpha = active ? 0.28 : (scale < 1.18 ? 0.15 : 0.08);
        const radius = sub.r + (scale < 1.1 ? 3 : 1.5);
        const gradient = ctx.createRadialGradient(sub.x, sub.y, 1, sub.x, sub.y, radius);
        gradient.addColorStop(0, colorAlpha(sub.c, alpha));
        gradient.addColorStop(0.68, colorAlpha(sub.c, alpha * 0.28));
        gradient.addColorStop(1, colorAlpha(sub.c, 0));
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(sub.x, sub.y, radius, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    if (scale < 0.95) {
      ctx.textBaseline = "middle";
      ctx.font = `${Math.max(9 / scale, 7)}px sans-serif`;
      for (const comm of COMMUNITIES) {
        if (comm.size < 60) continue;
        const label = comm.label.split(" · ")[0] || `社区 ${comm.g}`;
        ctx.fillStyle = getComputedStyle(document.body).getPropertyValue("--color-muted");
        ctx.fillText(`[${comm.size}] ${label.slice(0, 16)}`, comm.x - comm.r * 0.32, comm.y - comm.r - 8 / scale);
      }
    }
    ctx.restore();
  }

  function drawGlobalEdges(): void {
    drawIslandLinks(parentLinks, scale < 0.75 ? 0.42 : 0.24, 1.35);
    if (scale > 0.72) drawIslandLinks(childLinks, scale < 1.08 ? 0.22 : 0.13, 0.78);
  }

  function drawIslandLinks(links: IslandLink[], alphaBase: number, widthBase: number): void {
    ctx.save();
    ctx.translate(tx, ty);
    ctx.scale(scale, scale);
    for (const link of links) {
      const dx = link.bx - link.ax;
      const dy = link.by - link.ay;
      const len = Math.max(1, Math.hypot(dx, dy));
      const bend = Math.min(60, len * 0.12) * (link.cross ? 1 : -0.5);
      const cx = (link.ax + link.bx) / 2 - dy / len * bend;
      const cy = (link.ay + link.by) / 2 + dx / len * bend;
      const alpha = Math.min(0.62, alphaBase * (0.65 + Math.log1p(link.strength) * 0.42));
      ctx.strokeStyle = link.cross ? `rgba(210,224,255,${alpha})` : colorAlpha(link.color, alpha);
      ctx.lineWidth = Math.max(0.45 / scale, widthBase * (0.55 + Math.log1p(link.strength) * 0.34) / scale);
      ctx.beginPath();
      ctx.moveTo(link.ax, link.ay);
      ctx.quadraticCurveTo(cx, cy, link.bx, link.by);
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawActiveEdges(act: number): void {
    const n0 = NODES[act];
    const items = [...(adjEdges[act] || [])]
      .sort((a, b) => (b[3] - a[3]) || (b[2] - a[2]))
      .slice(0, 180);
    for (let i = 0; i < items.length; i += 1) {
      const [a, b, w, cc] = items[i];
      const target = a === act ? b : a;
      const n1 = NODES[target];
      const dx = n1.x - n0.x;
      const dy = n1.y - n0.y;
      const len = Math.max(1, Math.hypot(dx, dy));
      const sign = ((a * 31 + b * 17 + i) % 2) === 0 ? 1 : -1;
      const bend = Math.min(38, Math.max(8, len * 0.1)) * sign;
      const cx = (n0.x + n1.x) / 2 - dy / len * bend;
      const cy = (n0.y + n1.y) / 2 + dx / len * bend;
      const cross = n0.g !== n1.g;
      const alpha = Math.min(0.9, 0.26 + cc / 32 + w * 0.24);
      ctx.strokeStyle = cross ? `rgba(255,99,130,${alpha})` : `rgba(142,205,255,${alpha * 0.82})`;
      ctx.lineWidth = cross ? Math.max(1.3, 1.8 * scale) : Math.max(0.65, 0.95 * scale);
      ctx.beginPath();
      ctx.moveTo(X(n0.x), Y(n0.y));
      ctx.quadraticCurveTo(X(cx), Y(cy), X(n1.x), Y(n1.y));
      ctx.stroke();
    }
  }

  function shouldDrawNode(index: number, act: number, match: Set<number> | null): boolean {
    if (match) return match.has(index) || scale >= 1.35;
    if (act >= 0) return index === act || (adj[act] || []).includes(index);
    if (scale < 0.9) return false;
    if (scale < 1.35) return nodeIsRepresentative[index] || (adj[index]?.length || 0) >= 40;
    return true;
  }

  function representativeForColor(color: string): number {
    const comm = COMMUNITIES.find((item) => item.c === color);
    let best = -1;
    let bestScore = -Infinity;
    for (let i = 0; i < NODES.length; i += 1) {
      const n = NODES[i];
      if (n.c !== color) continue;
      const degree = adj[i]?.length || 0;
      if (degree <= 0) continue;
      const distScore = comm
        ? 1 - Math.min(1, Math.hypot(n.x - comm.x, n.y - comm.y) / Math.max(comm.r, 1))
        : 0;
      const degreeScore = Math.min(degree, 24) / 24;
      const score = distScore * 2 + degreeScore + (nodeIsRepresentative[i] ? 1 : 0);
      if (score > bestScore) {
        bestScore = score;
        best = i;
      }
    }
    return best;
  }

  function drawActiveNode(index: number): void {
    const n = NODES[index];
    const r = n.r * Math.max(0.6, Math.sqrt(scale)) * 1.65;
    ctx.save();
    ctx.globalAlpha = 1;
    ctx.shadowBlur = Math.min(64, Math.max(24, r * 3.1));
    ctx.shadowColor = "rgba(255,220,128,0.95)";
    ctx.fillStyle = "rgba(255,194,92,0.96)";
    ctx.beginPath();
    ctx.arc(X(n.x), Y(n.y), r, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = Math.min(26, Math.max(10, r * 1.3));
    ctx.shadowColor = "rgba(255,255,255,0.9)";
    ctx.lineWidth = Math.max(2.5, 1.4 * scale);
    ctx.strokeStyle = "rgba(255,255,255,0.96)";
    ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.lineWidth = Math.max(1.2, 0.8 * scale);
    ctx.strokeStyle = "rgba(255,126,80,0.95)";
    ctx.stroke();
    ctx.restore();
  }

  function drawBase(): void {
    ctx.clearRect(0, 0, W, H);
    const viewLeft = -tx / scale;
    const viewRight = (W - tx) / scale;
    const viewTop = -ty / scale;
    const viewBottom = (H - ty) / scale;

    function inView(x: number, y: number, r: number) {
      return x + r >= viewLeft && x - r <= viewRight && y + r >= viewTop && y - r <= viewBottom;
    }

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

    drawIslands(act);
    if (act < 0 && !match && scale <= globalEdgeScaleLimit) {
      drawGlobalEdges();
    } else if (act >= 0) {
      drawActiveEdges(act);
    }

    for (let i = 0; i < NODES.length; i += 1) {
      const n = NODES[i];
      let r = n.r * Math.max(0.6, Math.sqrt(scale));
      if (!shouldDrawNode(i, act, match)) continue;
      if (!inView(n.x, n.y, r * 1.5)) continue;
      if (act === i) continue;

      let alpha = 1;
      if ((adj[i]?.length || 0) === 0 && ccThreshold > 1 && !match) alpha = 0.08;
      if (act >= 0 && !hl.has(i)) alpha = 0.05;
      if (match && !match.has(i)) alpha = 0.06;
      const isActive = act === i;
      const isNeighbor = act >= 0 && hl.has(i) && !isActive;
      const isMatch = match && match.has(i);

      if (act < 0 && !match && scale < 1.35) {
        alpha *= 0.82;
        r *= 0.82;
      }
      if (isMatch) r *= 1.5;
      if (isNeighbor) r *= 1.1;

      ctx.globalAlpha = alpha;
      ctx.fillStyle = n.c;

      if (isMatch) {
        ctx.shadowBlur = Math.min(40, Math.max(12, r * 2.5));
        ctx.shadowColor = n.c;
      } else if (isNeighbor) {
        ctx.shadowBlur = Math.min(20, Math.max(4, r * 1));
        ctx.shadowColor = n.c;
      } else {
        ctx.shadowBlur = 0;
      }

      ctx.beginPath();
      ctx.arc(X(n.x), Y(n.y), r, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;

      if (isMatch || isNeighbor) {
        ctx.globalAlpha = 1;
        if (isMatch) {
          ctx.lineWidth = 2;
          ctx.strokeStyle = "#fff";
        } else {
          ctx.lineWidth = 1;
          ctx.strokeStyle = (NODES[i].g !== NODES[act].g) ? "#ff5a78" : "rgba(255, 255, 255, 0.6)";
        }
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;

    if (act >= 0) {
      const n = NODES[act];
      drawActiveNode(act);
      ctx.fillStyle = getComputedStyle(document.body).getPropertyValue("--color-muted");
      ctx.font = "bold 14px sans-serif";
      ctx.shadowBlur = 4;
      ctx.shadowColor = "rgba(0,0,0,0.8)";
      const text = n.t.length > 25 ? `${n.t.slice(0, 25)}...` : n.t;
      ctx.fillText(text, X(n.x) + 12, Y(n.y) - 12);
      ctx.shadowBlur = 0;
    }
  }

  function draw(): void {
    if (hlColor) {
      ctx.clearRect(0, 0, W, H);

      const viewLeft = -tx / scale;
      const viewRight = (W - tx) / scale;
      const viewTop = -ty / scale;
      const viewBottom = (H - ty) / scale;

      function inView(x: number, y: number, r: number) {
        return x + r >= viewLeft && x - r <= viewRight && y + r >= viewTop && y - r <= viewBottom;
      }
      drawIslands(-1);
      for (let i = 0; i < NODES.length; i += 1) {
        const n = NODES[i];
        let r = n.r * Math.max(0.6, Math.sqrt(scale));
        const on = n.c === hlColor;
        if (!on && scale < 1.2) continue;
        if (on && scale < 1.05 && !nodeIsRepresentative[i] && (adj[i]?.length || 0) < 40) continue;
        if (on) r *= 1.3;
        if (!inView(n.x, n.y, r * 1.5)) continue;

        ctx.globalAlpha = on ? 1 : ((adj[i]?.length || 0) === 0 ? 0.03 : 0.05);
        ctx.fillStyle = n.c;
        if (on) {
          ctx.shadowBlur = Math.min(30, r * 2);
          ctx.shadowColor = n.c;
        } else {
          ctx.shadowBlur = 0;
        }
        ctx.beginPath();
        ctx.arc(X(n.x), Y(n.y), r, 0, Math.PI * 2);
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
    const internal: Array<GraphNode & { nodeIndex: number; w: number; cc: number; sim: number }> = [];
    const external: Array<GraphNode & { nodeIndex: number; w: number; cc: number; sim: number }> = [];
    for (const nb of neighbors) {
      const target = { ...NODES[nb.id], nodeIndex: nb.id, w: nb.w, cc: nb.cc, sim: nb.sim };
      if (target.g === n.g) internal.push(target);
      else external.push(target);
    }

    let html = '<div style="font-size:13px;color:#8b9eb5;margin-bottom:6px;">选中记忆切片</div>';
    html += `<div style="font-size:14px;padding:12px;background:linear-gradient(135deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.02) 100%); border: 1px solid rgba(255,255,255,0.05); border-radius:8px; margin-bottom:16px;">${ghEscape(n.t)}</div>`;
    if (external.length > 0) {
      html += `<div style="color:#ff5a78;font-weight:bold;margin-bottom:4px;border-bottom:1px solid rgba(255,90,120,0.3);padding-bottom:6px;">思想跳跃 / 跨界走神 (${external.length})</div>`;
      html += '<div style="font-size:11px;color:#737a88;margin-bottom:12px;line-height:1.4;">溯源：分属不同的话题岛屿，但在特定时间点被你跨界关联。</div>';
      for (const t of external) {
        html += `<div class="detail-item" data-node="${t.nodeIndex}" title="锁定这个记忆切片" style="display:flex;gap:8px;align-items:flex-start;background:linear-gradient(90deg, rgba(255,255,255,0.05) 0%, transparent 100%); border-left: 2px solid ${t.c}; padding: 8px; margin-bottom: 8px; cursor:pointer;">
          <div style="display:flex;flex-direction:column;flex:1;"><span style="opacity:0.95">${ghEscape(t.t)}</span>${badgeHTML(t)}</div>
        </div>`;
      }
    }
    if (internal.length > 0) {
      html += `<div style="color:#96c8ff;font-weight:bold;margin-bottom:4px;margin-top:20px;border-bottom:1px solid rgba(150,200,255,0.3);padding-bottom:6px;">核心圈层 (${internal.length})</div>`;
      html += '<div style="font-size:11px;color:#737a88;margin-bottom:12px;line-height:1.4;">溯源：基于模块度算法，这些话题形成了高频同框的内聚孤岛。</div>';
      for (const t of internal) {
        html += `<div class="detail-item" data-node="${t.nodeIndex}" title="锁定这个记忆切片" style="display:flex;gap:8px;align-items:flex-start;background:linear-gradient(90deg, rgba(255,255,255,0.05) 0%, transparent 100%); border-left: 2px solid ${t.c}; padding: 8px; margin-bottom: 8px; cursor:pointer;">
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
    const match = filter
      ? new Set(NODES.map((n, i) => n.t.toLowerCase().includes(filter) ? i : -1).filter((i) => i >= 0))
      : null;
    for (let i = 0; i < NODES.length; i += 1) {
      if ((adj[i]?.length || 0) === 0 && !filter) continue;
      if (!shouldDrawNode(i, pinned, match)) continue;
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

  detailPanel.addEventListener("click", (event) => {
    const target = (event.target as HTMLElement).closest<HTMLElement>(".detail-item[data-node]");
    if (!target) return;
    event.stopPropagation();
    const next = Number(target.dataset.node);
    if (Number.isInteger(next) && next >= 0 && next < NODES.length) selectNode(next);
  });

  cv.addEventListener("mousedown", (event) => {
    if (tweenFrame) cancelAnimationFrame(tweenFrame);
    drag = true;
    moved = false;
    hover = -1;
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
      updateHlPaths();
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
    updateHlPaths();
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
      updateHlPaths();
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
    legEl.innerHTML = '<div class="grab">岛屿</div><div class="content" style="padding-top:4px;"><div style="margin-bottom:12px;"><b style="color:#fff;font-size:13px;">记忆岛屿</b> <span style="color:#737a88;">(悬停预览 · 点击选代表记忆)</span></div>'
      + LEG.map((l) => `<div class="row" data-c="${ghEscape(l.c)}"><span class="dot" style="background:${ghEscape(l.c)}"></span><span><span style="color:#9aa3b5">[${l.size}]</span> ${ghEscape(l.label)}</span></div>`).join("")
      + "</div>";
    legEl.querySelectorAll<HTMLElement>(".row").forEach((row) => {
      row.addEventListener("mouseenter", () => {
        if (!lockedColor) {
          filter = "";
          hlColor = row.dataset.c || null;
          updateHlPaths();
          draw();
        }
      });
      row.addEventListener("mouseleave", () => {
        if (!lockedColor) {
          hlColor = null;
          updateHlPaths();
          draw();
        }
      });
      row.addEventListener("click", () => {
        const c = row.dataset.c || null;
        if (!c) return;
        lockedColor = null;
        hlColor = null;
        filter = "";
        searchEl.value = "";
        resEl.style.display = "none";
        updateHlPaths();
        const target = representativeForColor(c);
        if (target >= 0) {
          selectNode(target);
        } else {
          pinned = -1;
          updateDetailPanel();
          flyToCommunity(c);
        }
        legEl.querySelectorAll<HTMLElement>(".row").forEach(r => r.classList.remove("selected"));
        row.classList.add("selected");
        draw();
      });
    });
  }

  function setLoadingCard(visible: boolean, payload?: GraphPayload): void {
    loadingCard.classList.toggle("visible", visible);
    if (!visible) return;
    loadingTick += 1;
    const elapsed = Math.max(0, Math.round((Date.now() - loadingStartedAt) / 1000));
    const progress = Math.min(92, 18 + loadingTick * 7 + elapsed * 2);
    loadingBar.style.width = `${progress}%`;
    const state = payload?.meta?.rebuilding ? "后台布局计算中" : "等待布局任务启动";
    loadingNote.textContent = `${state} · ${elapsed}s`;
  }

  function applyPayload(payload: GraphPayload, refit: boolean): void {
    if (payload.meta?.missing) {
      stat.textContent = payload.meta.rebuilding ? "快照后台生成中..." : "等待快照生成...";
      setLoadingCard(NODES.length === 0, payload);
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
    setLoadingCard(false);
    rebalanceFractalSpacing();
    recomputeCommunities();
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
    if (refit) {
      stat.textContent = "加载全景快照...";
      setLoadingCard(NODES.length === 0);
    }
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
