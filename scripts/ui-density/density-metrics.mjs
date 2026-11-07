// 视口信息密度度量（Tullis 1983 四项 + 触控/Hick + Rosenholtz 杂乱代理）
// collectInk 在每个 frame 内运行，返回视口坐标下的墨迹/控件矩形；scoreDensity 在 Node 端汇总。
export function collectInk(primarySel) {
  const W = innerWidth, H = innerHeight;
  // 祖先的 opacity:0 / visibility / display 也算不可见（opacity 不继承，只看自身会把悬停才显现的行算进来）。
  const visible = (el) => { const s = getComputedStyle(el); return Number(s.opacity) > 0.05
    && (el.checkVisibility ? el.checkVisibility({ opacityProperty: true, visibilityProperty: true }) : s.visibility !== "hidden" && s.display !== "none"); };
  const onTop = (x, y, el) => { const hit = document.elementFromPoint(Math.min(W - 1, Math.max(0, x)), Math.min(H - 1, Math.max(0, y))); return !!hit && (hit === el || el.contains(hit) || hit.contains(el)); };
  const ink = [], acts = [];
  const push = (r, el, kind) => { if (r.width < 1 || r.bottom <= 0 || r.top >= H || r.right <= 0 || r.left >= W) return; if (!onTop((r.left + r.right) / 2, (r.top + r.bottom) / 2, el)) return; ink.push([r.left, r.top, r.right, r.bottom, primarySel && el.closest(primarySel) ? 1 : 0, kind]); };
  const tw = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  for (let n = tw.nextNode(); n; n = tw.nextNode()) {
    const el = n.parentElement; if (!n.textContent.trim() || !el || !visible(el) || el.closest("[aria-hidden=true],[inert]")) continue;
    const range = document.createRange(); range.selectNodeContents(n);
    for (const r of range.getClientRects()) push(r, el, "t");
  }
  for (const el of document.querySelectorAll("svg, img, canvas, video, input, textarea, [role=switch]")) {
    if (!visible(el) || (el.tagName !== "svg" && el.closest("svg")) || (el.tagName === "svg" && el.parentElement?.closest("svg"))) continue;
    const r = el.getBoundingClientRect(); if (r.width < 2) continue;
    if (el.tagName === "INPUT" || el.tagName === "TEXTAREA") { push({ left: r.left, right: r.right, top: r.top, bottom: r.top + 1, width: r.width }, el, "b"); push({ left: r.left, right: r.right, top: r.bottom - 1, bottom: r.bottom, width: r.width }, el, "b"); continue; }
    push(r, el, "i");
  }
  for (const el of document.querySelectorAll("button, a[href], [role=button], [role=link], [role=menuitem], [role=tab], [role=switch], [role=option], summary, select, input:not([type=hidden]), textarea")) {
    if (!visible(el) || el.closest("[inert],[aria-hidden=true]")) continue; const r = el.getBoundingClientRect();
    if (r.width < 4 || r.height < 4 || r.bottom <= 0 || r.top >= H || r.right <= 0 || r.left >= W) continue;
    if (!onTop(r.left + r.width / 2, r.top + r.height / 2, el)) continue;
    // 行内文字链接按 WCAG 2.5.8 的行内例外不计入触控尺寸，仍计入可操作项。
    const inline = el.tagName === "A" && getComputedStyle(el).display === "inline";
    acts.push([r.left, r.top, r.width, r.height, (el.getAttribute("aria-label") || el.textContent || "").trim().replace(/\s+/g, " ").slice(0, 32), inline]);
  }
  return { ink, acts };
}

export function scoreDensity(frames, W, H) {
  const CELL = 8, cols = Math.ceil(W / CELL), rows = Math.ceil(H / CELL), total = cols * rows;
  const ink = new Uint8Array(total), pri = new Uint8Array(total); const lefts = []; const acts = [];
  // 1. 各 frame 的矩形按 iframe 位置平移并裁剪到可见区，落到 8px 网格
  for (const f of frames) {
    const [ox, oy, cw, ch] = f.clip;
    for (const [l, t, r, b, p, kind] of f.data.ink) {
      const L = Math.max(ox, l + ox), T = Math.max(oy, t + oy), Rr = Math.min(ox + cw, r + ox), B = Math.min(oy + ch, b + oy); if (Rr <= L || B <= T) continue;
      if (kind === "t") lefts.push(Math.round(L / 4) * 4);
      for (let y = Math.max(0, Math.floor(T / CELL)); y <= Math.min(rows - 1, Math.floor((B - 1) / CELL)); y++) for (let x = Math.max(0, Math.floor(L / CELL)); x <= Math.min(cols - 1, Math.floor((Rr - 1) / CELL)); x++) { ink[y * cols + x] = 1; if (p) pri[y * cols + x] = 1; }
    }
    for (const [l, t, w, h, name, inline] of f.data.acts) if (l + ox < ox + cw && t + oy < oy + ch) acts.push({ w, h, name, inline });
  }
  const sat = new Uint32Array((cols + 1) * (rows + 1));
  for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) sat[(y + 1) * (cols + 1) + x + 1] = ink[y * cols + x] + sat[y * (cols + 1) + x + 1] + sat[(y + 1) * (cols + 1) + x] - sat[y * (cols + 1) + x];
  const box = (x0, y0, x1, y1) => sat[y1 * (cols + 1) + x1] - sat[y0 * (cols + 1) + x1] - sat[y1 * (cols + 1) + x0] + sat[y0 * (cols + 1) + x0];
  // 2. 整体/局部密度：局部取 ±10 格（≈5° 视角）
  let inkN = 0, priN = 0, localSum = 0; const R = 10;
  for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) { const i = y * cols + x; if (!ink[i]) continue; inkN++; priN += pri[i];
    const x0 = Math.max(0, x - R), y0 = Math.max(0, y - R), x1 = Math.min(cols, x + R + 1), y1 = Math.min(rows, y + R + 1); localSum += box(x0, y0, x1, y1) / ((x1 - x0) * (y1 - y0)); }
  // 3. 分组（接近律）：每个有墨格向四周膨胀 1 格后取连通块，相隔 ≤2 个空格（约 16px，8px 栅格量化误差 ±8px）的墨迹并成一组
  const G = 1, dil = new Uint8Array(total);
  for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) if (box(Math.max(0, x - G), Math.max(0, y - G), Math.min(cols, x + G + 1), Math.min(rows, y + G + 1)) > 0) dil[y * cols + x] = 1;
  const seen = new Uint8Array(total), sizes = [];
  for (let i = 0; i < total; i++) if (dil[i] && !seen[i]) { const q = [i]; seen[i] = 1; let n = 0;
    while (q.length) { const c = q.pop(); n += ink[c]; const x = c % cols, y = (c / cols) | 0;
      if (x > 0 && dil[c - 1] && !seen[c - 1]) { seen[c - 1] = 1; q.push(c - 1); } if (x < cols - 1 && dil[c + 1] && !seen[c + 1]) { seen[c + 1] = 1; q.push(c + 1); }
      if (y > 0 && dil[c - cols] && !seen[c - cols]) { seen[c - cols] = 1; q.push(c - cols); } if (y < rows - 1 && dil[c + cols] && !seen[c + cols]) { seen[c + cols] = 1; q.push(c + cols); } }
    if (n) sizes.push(n); }
  sizes.sort((a, b) => a - b);
  // 4. 布局复杂度：文字左缘取值熵
  const freq = new Map(); for (const l of lefts) freq.set(l, (freq.get(l) || 0) + 1); let ent = 0; for (const v of freq.values()) { const p = v / lefts.length; ent -= p * Math.log2(p); }
  const names = new Map(); for (const a of acts) if (a.name) names.set(a.name, (names.get(a.name) || 0) + 1);
  const sized = acts.filter((a) => !a.inline);
  return {
    overallDensity: +(inkN / total).toFixed(3), localDensity: +(inkN ? localSum / inkN : 0).toFixed(3),
    groups: sizes.length, groupMedianCells: sizes.length ? sizes[sizes.length >> 1] : 0,
    layoutEntropy: +ent.toFixed(2), primaryShare: +(inkN ? priN / inkN : 0).toFixed(3),
    actions: acts.length, touchUnder44: +(sized.length ? sized.filter((a) => Math.min(a.w, a.h) < 44).length / sized.length : 0).toFixed(2),
    touchUnder24: sized.filter((a) => Math.min(a.w, a.h) < 24).length,
    // 只按读屏名聚合：作用于不同对象的同名控件（每条消息的「复制」）也会出现，需人工核对是否同一目标。
    sameNameControls: [...names].filter(([, c]) => c > 1).map(([n, c]) => `${n}×${c}`),
  };
}
