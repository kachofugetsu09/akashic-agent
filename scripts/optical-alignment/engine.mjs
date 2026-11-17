/**
 * 光学对齐检测引擎（Optical Alignment Blur-Volume Analyzer）
 *
 * 原理：
 * 1. 在真实 Chromium 中捕获页面原貌与前景墨迹分离图（icon=G, text=R）。
 * 2. 使用高半径高斯模糊（blur volume）消除细线与尖角干扰，还原元素视觉质量分布。
 * 3. 计算像素级一阶质心、模糊核心质心（B^1.8 加权）与紧墨迹包围盒边距。
 * 4. 生成两类给 Agent 与开发者直接查看的图像：
 *    - 全页原地模糊透视图（in-place blur overlay）
 *    - 原样 vs 模糊体积特写拼版图（contact sheet，含容器蓝框、几何中心十字、光学质心标记）
 */

/** 在浏览器目标 frame 内运行：收集可见的 icon-only 与 icon-text 控件，并打上临时 data-oa-id。 */
export function collectOpticalTargetsInFrame(frameId) {
  const W = window.innerWidth;
  const H = window.innerHeight;
  const isVisible = (el) => {
    const style = getComputedStyle(el);
    if (Number(style.opacity) <= 0.05 || style.visibility === "hidden" || style.display === "none") return false;
    if (el.checkVisibility && !el.checkVisibility({ opacityProperty: true, visibilityProperty: true })) return false;
    const r = el.getBoundingClientRect();
    return r.width >= 6 && r.height >= 6 && r.right > 0 && r.bottom > 0 && r.left < W && r.top < H;
  };
  const onTop = (el) => {
    const r = el.getBoundingClientRect();
    const hitSelf = (px, py) => {
      if (px < 0 || py < 0 || px >= W || py >= H) return true;
      const hit = document.elementFromPoint(px, py);
      return !!hit && (hit === el || el.contains(hit) || hit.contains(el));
    };
    const cx = Math.min(W - 1, Math.max(0, r.left + r.width / 2));
    const cy = Math.min(H - 1, Math.max(0, r.top + r.height / 2));
    if (!hitSelf(cx, cy)) return false;
    // 宽按钮或边角被弹层局部盖住时（如主题浮层盖住底层导航行右侧），不把残缺轮廓误报为光学偏移
    if (r.width > 48 || r.height > 48) {
      const ix = Math.min(8, r.width / 4);
      const iy = Math.min(6, r.height / 4);
      if (
        !hitSelf(r.left + ix, r.top + iy) ||
        !hitSelf(r.right - ix, r.top + iy) ||
        !hitSelf(r.left + ix, r.bottom - iy) ||
        !hitSelf(r.right - ix, r.bottom - iy)
      ) {
        return false;
      }
    }
    return true;
  };
  const isMaskElement = (el) => {
    const s = getComputedStyle(el);
    return (s.maskImage && s.maskImage !== "none") || (s.webkitMaskImage && s.webkitMaskImage !== "none");
  };
  const hasVisibleVisual = (el) => {
    if (!isVisible(el)) return false;
    const s = getComputedStyle(el);
    if (s.backgroundColor && s.backgroundColor !== "rgba(0, 0, 0, 0)" && s.backgroundColor !== "transparent") return true;
    if (Number.parseFloat(s.borderTopWidth) > 0 && s.borderTopColor !== "transparent") return true;
    if (Number.parseFloat(s.borderBottomWidth) > 0 && s.borderBottomColor !== "transparent") return true;
    return false;
  };
  const visibleTextNodes = (root) => {
    const out = [];
    const tw = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    for (let n = tw.nextNode(); n; n = tw.nextNode()) {
      if (!n.textContent || !n.textContent.trim()) continue;
      const p = n.parentElement;
      if (!p || !isVisible(p) || p.closest("[aria-hidden='true'], .sr-only, .visually-hidden")) continue;
      // 跳过只读屏幕隐藏文本（clip / 1px 宽等）
      const ps = getComputedStyle(p);
      if (ps.position === "absolute" && Number.parseFloat(ps.width) <= 2 && Number.parseFloat(ps.height) <= 2) continue;
      out.push({ node: n, text: n.textContent.trim(), rect: p.getBoundingClientRect() });
    }
    return out;
  };
  const findIcons = (root) => {
    const icons = [];
    const candidates = root.querySelectorAll("svg, img, i, span");
    for (const el of candidates) {
      if (!isVisible(el)) continue;
      if (el.tagName.toLowerCase() === "svg") {
        if (el.parentElement?.closest("svg")) continue;
        // 忽略超大图表 svg 或装饰性隐身 0 面积 svg
        const r = el.getBoundingClientRect();
        if (r.width < 6 || r.height < 6 || r.width > 96 || r.height > 96) continue;
        // 忽略追逐圆环（composer-action-button__chase）这类外圈描边动画覆盖层
        if (el.classList.contains("composer-action-button__chase") || el.classList.contains("session-delete-notice__ring")) continue;
        icons.push({ el, kind: "svg", rect: r });
      } else if (el.tagName.toLowerCase() === "img") {
        const r = el.getBoundingClientRect();
        if (r.width < 8 || r.height < 8 || r.width > 64 || r.height > 64) continue;
        icons.push({ el, kind: "img", rect: r });
      } else if (isMaskElement(el)) {
        const r = el.getBoundingClientRect();
        if (r.width < 6 || r.height < 6 || r.width > 72 || r.height > 72) continue;
        icons.push({ el, kind: "mask", rect: r });
      }
    }
    // 对 icon-swap（叠放两个图标，其中一个是隐藏让位态 opacity < 0.1），isVisible 已过滤掉透明那枚
    return icons;
  };
  const describe = (el) => {
    const aria = el.getAttribute("aria-label") || el.getAttribute("title") || el.getAttribute("data-tip") || "";
    const cls = typeof el.className === "string"
      ? el.className.trim().split(/\s+/).filter(Boolean).slice(0, 3).join(".")
      : "";
    const tag = el.tagName.toLowerCase();
    const txt = (el.textContent || "").trim().replace(/\s+/g, " ").slice(0, 24);
    return {
      selectorHint: cls ? `${tag}.${cls}` : tag,
      label: aria || txt || cls || tag,
    };
  };

  // 先清理旧标记
  for (const prev of document.querySelectorAll("[data-oa-id]")) prev.removeAttribute("data-oa-id");

  const targets = [];
  const seenContainers = new Set();
  const controlSelectors = [
    ".destination-row__avatar",
    "button",
    "a[href]",
    "summary",
    "[role='button']",
    "[role='tab']",
    "[role='menuitem']",
    "[role='option']",
    ".product-band__brand",
    ".product-band__item",
    ".product-band__action",
    ".shell-plugin-icon",
    ".session-status-mark",
    ".composer-reply",
    ".search-trigger",
    ".sidebar-search",
  ].join(",");

  let seq = 0;
  for (const el of document.querySelectorAll(controlSelectors)) {
    if (!isVisible(el) || el.closest("[inert]")) continue;
    const r = el.getBoundingClientRect();
    if (r.width > 760 || r.height > 120) continue;
    if (!onTop(el)) continue;

    const icons = findIcons(el);
    if (icons.length === 0) continue;
    const texts = visibleTextNodes(el);
    if (seenContainers.has(el)) continue;
    seenContainers.add(el);

    const style = getComputedStyle(el);
    const textHeight = texts.length
      ? Math.max(...texts.map((t) => t.rect.bottom)) - Math.min(...texts.map((t) => t.rect.top))
      : 0;
    const fontSize = Number.parseFloat(style.fontSize) || 14;
    let category = null;
    if (texts.length === 0 && icons.length === 1) {
      category = "icon-only";
    } else if (texts.length >= 1 && icons.length >= 1) {
      if (r.height <= 64) {
        category = textHeight > fontSize * 1.9 ? "icon-text-multiline" : "icon-text";
      }
    }
    if (!category) continue;

    const id = `${frameId}-${++seq}`;
    el.setAttribute("data-oa-id", id);
    const meta = describe(el);
    const iconRect = icons[0].rect;
    targets.push({
      id,
      category,
      iconKind: icons[0].kind,
      iconCount: icons.length,
      hasContainerShape: hasVisibleVisual(el),
      justifyContent: style.justifyContent,
      textAlign: style.textAlign,
      rect: { x: r.left, y: r.top, w: r.width, h: r.height },
      iconRect: { x: iconRect.left, y: iconRect.top, w: iconRect.width, h: iconRect.height },
      borderRadius: style.borderRadius,
      padding: {
        top: Number.parseFloat(style.paddingTop) || 0,
        right: Number.parseFloat(style.paddingRight) || 0,
        bottom: Number.parseFloat(style.paddingBottom) || 0,
        left: Number.parseFloat(style.paddingLeft) || 0,
      },
      fontSize,
      textSample: texts.map((t) => t.text).join(" ").slice(0, 32),
      ...meta,
    });
  }

  // 再补充：未被任何 control 选中的独立可见小 SVG（如折叠 chevron、状态标志、消息内图标）
  for (const svg of document.querySelectorAll("svg")) {
    if (!isVisible(svg) || svg.parentElement?.closest("svg")) continue;
    if (svg.closest("[data-oa-id]")) continue;
    if (svg.classList.contains("composer-action-button__chase") || svg.classList.contains("session-delete-notice__ring")) continue;
    const r = svg.getBoundingClientRect();
    if (r.width < 10 || r.height < 10 || r.width > 64 || r.height > 64) continue;
    if (!onTop(svg)) continue;
    const id = `${frameId}-${++seq}`;
    svg.setAttribute("data-oa-id", id);
    const meta = describe(svg.parentElement || svg);
    targets.push({
      id,
      category: "icon-self",
      iconKind: "svg",
      iconCount: 1,
      hasContainerShape: false,
      rect: { x: r.left, y: r.top, w: r.width, h: r.height },
      iconRect: { x: r.left, y: r.top, w: r.width, h: r.height },
      borderRadius: "4px",
      padding: { top: 0, right: 0, bottom: 0, left: 0 },
      fontSize: Number.parseFloat(getComputedStyle(svg).fontSize) || 14,
      textSample: "",
      selectorHint: `${meta.selectorHint} > svg`,
      label: meta.label,
    });
  }

  return targets;
}

/**
 * 在 frame 中开启/关闭「前景通道分离模式」：
 * - 背景压黑 #000000
 * - 图标（SVG / mask / icon img）变纯绿 #00ff00（保持原有 stroke vs fill 语义）
 * - 文字变纯红 #ff0000
 * 返回恢复函数句柄数据；通过第二次调用 restoreInkIsolationInFrame 恢复。
 */
export function enableInkIsolationInFrame() {
  const mutated = [];
  const saveStyle = (el, prop, val) => {
    const prevVal = el.style.getPropertyValue(prop);
    const prevPrio = el.style.getPropertyPriority(prop);
    mutated.push([el, prop, prevVal, prevPrio]);
    el.style.setProperty(prop, val, "important");
  };

  // 全局注入强力覆盖表（压黑背景与边框，文字转红），避免伪元素或阴影污染像素
  const styleTag = document.createElement("style");
  styleTag.id = "__oa_ink_isolation_style__";
  styleTag.textContent = `
    html, body, body * {
      background-color: #000000 !important;
      background-image: none !important;
      border-color: transparent !important;
      outline-color: transparent !important;
      box-shadow: none !important;
      text-shadow: none !important;
      color: #ff0000 !important;
      caret-color: transparent !important;
    }
    html, body {
      background: #000000 !important;
    }
    [data-oa-id] *::before, [data-oa-id] *::after,
    [data-oa-id]::before, [data-oa-id]::after {
      opacity: 0 !important;
      background: transparent !important;
      border-color: transparent !important;
    }
  `;
  document.documentElement.appendChild(styleTag);

  const targets = document.querySelectorAll("[data-oa-id]");
  for (const root of targets) {
    // 若容器本身就是 icon-self（<svg>），或者内部有 svg / img / mask
    const svgs = root.tagName.toLowerCase() === "svg" ? [root] : [...root.querySelectorAll("svg")];
    for (const svg of svgs) {
      if (svg.classList.contains("composer-action-button__chase") || svg.classList.contains("session-delete-notice__ring")) {
        saveStyle(svg, "opacity", "0");
        continue;
      }
      saveStyle(svg, "color", "#00ff00");
      const shapes = svg.querySelectorAll("path, circle, rect, ellipse, polygon, polyline, line");
      const nodes = shapes.length ? shapes : [svg];
      for (const node of nodes) {
        const cs = getComputedStyle(node);
        const hasFill = cs.fill && cs.fill !== "none" && cs.fill !== "rgba(0, 0, 0, 0)" && cs.fill !== "transparent";
        const hasStroke = cs.stroke && cs.stroke !== "none" && cs.stroke !== "rgba(0, 0, 0, 0)" && cs.stroke !== "transparent";
        if (hasFill) saveStyle(node, "fill", "#00ff00");
        else saveStyle(node, "fill", "none");
        if (hasStroke) saveStyle(node, "stroke", "#00ff00");
        else saveStyle(node, "stroke", "none");
      }
    }

    // CSS mask 图标：靠 background-color 显示形状
    const allEls = [root, ...root.querySelectorAll("*")];
    for (const el of allEls) {
      const cs = getComputedStyle(el);
      const mask = (cs.maskImage && cs.maskImage !== "none") || (cs.webkitMaskImage && cs.webkitMaskImage !== "none");
      if (mask) {
        saveStyle(el, "background-color", "#00ff00");
        saveStyle(el, "opacity", "1");
      }
    }

    // <img> 图标：将透明底色图标染白，后续合并入 icon 通道
    for (const img of root.querySelectorAll("img")) {
      const r = img.getBoundingClientRect();
      if (r.width <= 48 && r.height <= 48) {
        saveStyle(img, "filter", "brightness(0) invert(1)");
      } else {
        saveStyle(img, "opacity", "0");
      }
    }
  }

  window.__oa_restore__ = () => {
    styleTag.remove();
    for (let i = mutated.length - 1; i >= 0; i--) {
      const [el, prop, val, prio] = mutated[i];
      if (val) el.style.setProperty(prop, val, prio);
      else el.style.removeProperty(prop);
    }
  };
}

export function disableInkIsolationInFrame() {
  if (typeof window.__oa_restore__ === "function") {
    window.__oa_restore__();
    delete window.__oa_restore__;
  }
  for (const prev of document.querySelectorAll("[data-oa-id]")) prev.removeAttribute("data-oa-id");
}

/**
 * 在专用 Chromium Canvas 页面内执行：
 * 传入 origPngBase64、inkPngBase64、dpr、targets（全页面坐标系）
 * 返回：
 *   - results: 每个控件的精确光学指标
 *   - sheetPngBase64: 原样 vs 模糊体积特写拼版对照图
 *   - overlayPngBase64: 全页原地模糊透视图
 */
export async function analyzeAndRenderInCanvasPage(canvasPage, { origPngBase64, inkPngBase64, dpr, viewport, targets, label }) {
  return canvasPage.evaluate(async ({ origPngBase64, inkPngBase64, dpr, viewport, targets, label }) => {
    const loadImg = (b64) => new Promise((res, rej) => {
      const img = new Image();
      img.onload = () => res(img);
      img.onerror = rej;
      img.src = "data:image/png;base64," + b64;
    });
    const [origImg, inkImg] = await Promise.all([loadImg(origPngBase64), loadImg(inkPngBase64)]);

    // 读出整张 inkImg 像素
    const inkCanvas = document.createElement("canvas");
    inkCanvas.width = inkImg.width;
    inkCanvas.height = inkImg.height;
    const inkCtx = inkCanvas.getContext("2d", { willReadFrequently: true });
    inkCtx.drawImage(inkImg, 0, 0);

    // 临时离屏画布，用来对单个控件生成高亮白色前景 -> 高斯模糊 -> 读取模糊场
    const workCanvas = document.createElement("canvas");
    const workCtx = workCanvas.getContext("2d", { willReadFrequently: true });
    const blurCanvas = document.createElement("canvas");
    const blurCtx = blurCanvas.getContext("2d", { willReadFrequently: true });

    const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

    const results = [];

    for (const t of targets) {
      const rx = Math.round(t.rect.x * dpr);
      const ry = Math.round(t.rect.y * dpr);
      const rw = Math.max(1, Math.round(t.rect.w * dpr));
      const rh = Math.max(1, Math.round(t.rect.h * dpr));
      if (rx + rw <= 0 || ry + rh <= 0 || rx >= inkImg.width || ry >= inkImg.height) continue;
      const sx = clamp(rx, 0, inkImg.width - 1);
      const sy = clamp(ry, 0, inkImg.height - 1);
      const sw = clamp(rx + rw, 0, inkImg.width) - sx;
      const sh = clamp(ry + rh, 0, inkImg.height) - sy;
      if (sw < 4 || sh < 4) continue;

      const data = inkCtx.getImageData(sx, sy, sw, sh).data;

      // 1. 提取各通道墨迹质心 & 紧包围盒
      let iconSum = 0, iconX = 0, iconY = 0;
      let textSum = 0, textX = 0, textY = 0;
      let allSum = 0, allX = 0, allY = 0;
      let iMinX = sw, iMinY = sh, iMaxX = -1, iMaxY = -1;
      let tMinX = sw, tMinY = sh, tMaxX = -1, tMaxY = -1;
      let aMinX = sw, aMinY = sh, aMaxX = -1, aMaxY = -1;

      workCanvas.width = sw;
      workCanvas.height = sh;
      const whiteImg = workCtx.createImageData(sw, sh);
      const wd = whiteImg.data;

      for (let py = 0; py < sh; py++) {
        const cyPix = py + 0.5;
        for (let px = 0; px < sw; px++) {
          const cxPix = px + 0.5;
          const idx = (py * sw + px) * 4;
          const r = data[idx];
          const g = data[idx + 1];
          const b = data[idx + 2];
          // img 被染成纯白 (r≈g≈b) 时归为 icon；否则 G=icon, R=text
          // icon-only / icon-self 不应混入重叠兄弟元素的文字红色通道
          const isWhiteImg = t.iconKind === "img" && t.category === "icon-only";
          const iW = isWhiteImg ? Math.max(r, g, b) : Math.max(0, g - Math.max(r, b) * 0.25);
          const tW = (isWhiteImg || t.category === "icon-only" || t.category === "icon-self")
            ? 0
            : Math.max(0, r - g * 0.5);
          const aW = Math.max(iW, tW);

          if (iW > 18) {
            iconSum += iW;
            iconX += cxPix * iW;
            iconY += cyPix * iW;
            if (px < iMinX) iMinX = px;
            if (px > iMaxX) iMaxX = px;
            if (py < iMinY) iMinY = py;
            if (py > iMaxY) iMaxY = py;
          }
          if (tW > 18) {
            textSum += tW;
            textX += cxPix * tW;
            textY += cyPix * tW;
            if (px < tMinX) tMinX = px;
            if (px > tMaxX) tMaxX = px;
            if (py < tMinY) tMinY = py;
            if (py > tMaxY) tMaxY = py;
          }
          if (aW > 18) {
            allSum += aW;
            allX += cxPix * aW;
            allY += cyPix * aW;
            if (px < aMinX) aMinX = px;
            if (px > aMaxX) aMaxX = px;
            if (py < aMinY) aMinY = py;
            if (py > aMaxY) aMaxY = py;
          }

          wd[idx] = 255;
          wd[idx + 1] = 255;
          wd[idx + 2] = 255;
          wd[idx + 3] = Math.min(255, Math.round(aW));
        }
      }

      if (allSum < 50 || iconSum < 30) continue;

      workCtx.putImageData(whiteImg, 0, 0);

      // 2. 生成自适应高斯模糊体积场（加 pad 防止边界截断）
      const sigmaPx = clamp(Math.min(sw, sh) * 0.14, 3 * dpr, 10 * dpr);
      const pad = Math.ceil(sigmaPx * 2.5);
      const bw = sw + pad * 2;
      const bh = sh + pad * 2;
      blurCanvas.width = bw;
      blurCanvas.height = bh;
      blurCtx.clearRect(0, 0, bw, bh);
      blurCtx.save();
      blurCtx.filter = `blur(${sigmaPx.toFixed(1)}px)`;
      // 叠画两次轻微增强细线条模糊后的体量可见度（与推文视觉一致）
      blurCtx.drawImage(workCanvas, pad, pad);
      blurCtx.globalAlpha = 0.65;
      blurCtx.drawImage(workCanvas, pad, pad);
      blurCtx.restore();

      const bd = blurCtx.getImageData(0, 0, bw, bh).data;
      let blurCoreSum = 0, blurCoreX = 0, blurCoreY = 0;
      let blurIconLeft = 0, blurIconRight = 0, blurIconTop = 0, blurIconBottom = 0;
      const geoCx = sw / 2;
      const geoCy = sh / 2;

      for (let py = 0; py < sh; py++) {
        const cyPix = py + 0.5;
        for (let px = 0; px < sw; px++) {
          const cxPix = px + 0.5;
          const bIdx = ((py + pad) * bw + (px + pad)) * 4;
          const alpha = bd[bIdx + 3] / 255;
          if (alpha <= 0.01) continue;
          const wCore = Math.pow(alpha, 1.8);
          blurCoreSum += wCore;
          blurCoreX += cxPix * wCore;
          blurCoreY += cyPix * wCore;
          if (cxPix < geoCx) blurIconLeft += alpha;
          else blurIconRight += alpha;
          if (cyPix < geoCy) blurIconTop += alpha;
          else blurIconBottom += alpha;
        }
      }

      const rawCx = (allX / allSum) / dpr;
      const rawCy = (allY / allSum) / dpr;
      const blurCx = (blurCoreSum > 0 ? blurCoreX / blurCoreSum : allX / allSum) / dpr;
      const blurCy = (blurCoreSum > 0 ? blurCoreY / blurCoreSum : allY / allSum) / dpr;
      const containerW = sw / dpr;
      const containerH = sh / dpr;

      const iconRawCx = (iconX / iconSum) / dpr;
      const iconRawCy = (iconY / iconSum) / dpr;
      const textRawCy = textSum > 20 ? (textY / textSum) / dpr : null;

      // 实测紧墨迹外边距（CSS px）
      const inkMargins = {
        left: +(aMinX / dpr).toFixed(2),
        right: +((sw - 1 - aMaxX) / dpr).toFixed(2),
        top: +(aMinY / dpr).toFixed(2),
        bottom: +((sh - 1 - aMaxY) / dpr).toFixed(2),
      };

      const iconBox = {
        left: +(iMinX / dpr).toFixed(2),
        right: +((iMaxX + 1) / dpr).toFixed(2),
        top: +(iMinY / dpr).toFixed(2),
        bottom: +((iMaxY + 1) / dpr).toFixed(2),
      };
      const textBox = textSum > 20 ? {
        left: +(tMinX / dpr).toFixed(2),
        right: +((tMaxX + 1) / dpr).toFixed(2),
        top: +(tMinY / dpr).toFixed(2),
        bottom: +((tMaxY + 1) / dpr).toFixed(2),
      } : null;

      const blurDx = +(blurCx - containerW / 2).toFixed(2);
      const blurDy = +(blurCy - containerH / 2).toFixed(2);
      const rawDx = +(rawCx - containerW / 2).toFixed(2);
      const rawDy = +(rawCy - containerH / 2).toFixed(2);
      const bboxDx = +(((aMinX + aMaxX + 1) / 2 / dpr) - containerW / 2).toFixed(2);
      const bboxDy = +(((aMinY + aMaxY + 1) / 2 / dpr) - containerH / 2).toFixed(2);

      // 区分评估指标：
      // - icon-only / icon-self: 关注容器中心 vs 模糊体积中心 (blurDx, blurDy)
      // - icon-text:
      //   * 所有图文控件：图标墨迹光学中心 Y vs 文字墨迹光学中心 Y 的差 (iconTextDeltaY)
      //   * 若为居中对称控件（justifyContent=center 或左右墨迹边距都 <= 32px 且非左对齐列表项），也检查左右光学边距平衡
      const iconTextDeltaY = textRawCy !== null ? +(iconRawCy - textRawCy).toFixed(2) : null;
      const marginAsymX = +(inkMargins.left - inkMargins.right).toFixed(2);
      const marginAsymY = +(inkMargins.top - inkMargins.bottom).toFixed(2);
      const isCenteredIconText = t.category === "icon-text"
        && (t.justifyContent === "center" || (inkMargins.left <= 32 && inkMargins.right <= 32 && containerW <= 220));

      let severityScore;
      if (t.category === "icon-only" || t.category === "icon-self") {
        severityScore = Math.hypot(blurDx, blurDy);
      } else if (t.category === "icon-text-multiline") {
        // 多行卡片左侧图标通常与整体/首行对齐，不把多行总高度差当作错位
        severityScore = Math.abs(marginAsymY) * 0.25;
      } else {
        severityScore = isCenteredIconText
          ? Math.max(Math.abs(iconTextDeltaY ?? 0), Math.abs(marginAsymX) * 0.6, Math.abs(blurDy))
          : Math.max(Math.abs(iconTextDeltaY ?? 0), Math.abs(marginAsymY) * 0.5);
      }

      results.push({
        id: t.id,
        category: t.category,
        isCenteredIconText,
        selectorHint: t.selectorHint,
        label: t.label,
        textSample: t.textSample,
        rect: { x: +t.rect.x.toFixed(1), y: +t.rect.y.toFixed(1), w: +containerW.toFixed(1), h: +containerH.toFixed(1) },
        hasContainerShape: t.hasContainerShape,
        padding: t.padding,
        blurDx,
        blurDy,
        blurDxPct: +((blurDx / containerW) * 100).toFixed(1),
        blurDyPct: +((blurDy / containerH) * 100).toFixed(1),
        rawDx,
        rawDy,
        bboxDx,
        bboxDy,
        inkMargins,
        marginAsymX,
        marginAsymY,
        iconRawCx: +iconRawCx.toFixed(2),
        iconRawCy: +iconRawCy.toFixed(2),
        textRawCy: textRawCy !== null ? +textRawCy.toFixed(2) : null,
        iconTextDeltaY,
        iconBox,
        textBox,
        volumeRatioLR: +(blurIconLeft / Math.max(1e-3, blurIconRight)).toFixed(2),
        volumeRatioTB: +(blurIconTop / Math.max(1e-3, blurIconBottom)).toFixed(2),
        severityScore: +severityScore.toFixed(2),
        _crop: { sx, sy, sw, sh, sigmaPx, blurDataUrl: blurCanvas.toDataURL("image/png"), pad },
      });
    }

    // 3. 渲染「全页原地模糊透视图」（In-Page Blur Overlay）
    const overlayCanvas = document.createElement("canvas");
    overlayCanvas.width = origImg.width;
    overlayCanvas.height = origImg.height;
    const octx = overlayCanvas.getContext("2d");
    octx.drawImage(origImg, 0, 0);
    // 压暗原图作为背景，让模糊胶囊跳出来
    octx.fillStyle = "rgba(18, 20, 24, 0.35)";
    octx.fillRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    for (const item of results) {
      const { sx, sy, sw, sh, pad } = item._crop;
      const bImg = await new Promise((res) => { const im = new Image(); im.onload = () => res(im); im.src = item._crop.blurDataUrl; });
      const radius = Math.min(sw, sh) * 0.28;
      // 黑色胶囊底
      octx.save();
      octx.beginPath();
      octx.roundRect(sx, sy, sw, sh, radius);
      octx.fillStyle = "#0b0d10";
      octx.fill();
      octx.clip();
      // 高亮模糊白色体积
      octx.drawImage(bImg, pad, pad, sw, sh, sx, sy, sw, sh);
      // 叠加青蓝色内框 + 几何中心十字
      octx.strokeStyle = "rgba(56, 189, 248, 0.85)";
      octx.lineWidth = Math.max(1, dpr);
      octx.beginPath();
      octx.roundRect(sx + dpr, sy + dpr, sw - dpr * 2, sh - dpr * 2, Math.max(2, radius - dpr));
      octx.stroke();
      octx.setLineDash([3 * dpr, 3 * dpr]);
      octx.strokeStyle = "rgba(56, 189, 248, 0.55)";
      octx.beginPath();
      octx.moveTo(sx + sw / 2, sy);
      octx.lineTo(sx + sw / 2, sy + sh);
      octx.moveTo(sx, sy + sh / 2);
      octx.lineTo(sx + sw, sy + sh / 2);
      octx.stroke();
      octx.setLineDash([]);
      // 光学重心标记（偏移大于 0.7px 用珊瑚红，否则绿色）
      const off = item.category === "icon-text" ? Math.abs(item.iconTextDeltaY ?? 0) : Math.hypot(item.blurDx, item.blurDy);
      const color = off >= 1.2 ? "#fb7185" : off >= 0.6 ? "#fbbf24" : "#34d399";
      const ox = sx + (item.category === "icon-text" ? item.iconRawCx : (item.rect.w / 2 + item.blurDx)) * dpr;
      const oy = sy + (item.category === "icon-text" ? item.iconRawCy : (item.rect.h / 2 + item.blurDy)) * dpr;
      octx.fillStyle = color;
      octx.beginPath();
      octx.arc(ox, oy, Math.max(2.5, 2.2 * dpr), 0, Math.PI * 2);
      octx.fill();
      octx.restore();
    }

    // 4. 渲染「原样 vs 模糊体积特写拼版图」（按 severityScore 降序）
    const sorted = [...results].sort((a, b) => b.severityScore - a.severityScore);
    const cols = 2;
    const cardW = 540;
    const cardH = 148;
    const gap = 14;
    const headerH = 44;
    const rows = Math.max(1, Math.ceil(sorted.length / cols));
    const sheetCanvas = document.createElement("canvas");
    sheetCanvas.width = cols * cardW + (cols + 1) * gap;
    sheetCanvas.height = headerH + rows * (cardH + gap) + gap;
    const sctx = sheetCanvas.getContext("2d");
    sctx.fillStyle = "#111317";
    sctx.fillRect(0, 0, sheetCanvas.width, sheetCanvas.height);

    sctx.fillStyle = "#f3f4f6";
    sctx.font = "600 15px system-ui, sans-serif";
    sctx.fillText(`Optical Alignment Blur-Volume Sheet (${label}) · ${sorted.length} items · sorted by offset`, gap, 27);

    for (let i = 0; i < sorted.length; i++) {
      const item = sorted[i];
      const col = i % cols;
      const row = Math.floor(i / cols);
      const cx = gap + col * (cardW + gap);
      const cy = headerH + gap + row * (cardH + gap);

      // Card bg
      sctx.fillStyle = "#191c22";
      sctx.beginPath();
      sctx.roundRect(cx, cy, cardW, cardH, 10);
      sctx.fill();
      sctx.strokeStyle = "#2b303b";
      sctx.lineWidth = 1;
      sctx.stroke();

      // Left box: Original magnified
      const boxW = 112;
      const boxH = 96;
      const origBoxX = cx + 12;
      const boxY = cy + 38;
      const blurBoxX = origBoxX + boxW + 10;

      // 标题栏
      const badgeColor = item.severityScore >= 1.5 ? "#fb7185" : item.severityScore >= 0.8 ? "#fbbf24" : "#34d399";
      sctx.fillStyle = badgeColor;
      sctx.beginPath();
      sctx.arc(cx + 18, cy + 20, 4.5, 0, Math.PI * 2);
      sctx.fill();

      sctx.fillStyle = "#e5e7eb";
      sctx.font = "600 12px system-ui, sans-serif";
      const titleText = `[${item.category}] ${item.label}`.slice(0, 42);
      sctx.fillText(titleText, cx + 28, cy + 24);

      const { sx, sy, sw, sh, pad, blurDataUrl } = item._crop;
      const scale = Math.min((boxW - 14) / sw, (boxH - 14) / sh);
      const dw = sw * scale;
      const dh = sh * scale;
      const ox = origBoxX + (boxW - dw) / 2;
      const oy = boxY + (boxH - dh) / 2;

      // Original panel
      sctx.fillStyle = "#252932";
      sctx.beginPath();
      sctx.roundRect(origBoxX, boxY, boxW, boxH, 8);
      sctx.fill();
      sctx.drawImage(origImg, sx, sy, sw, sh, ox, oy, dw, dh);
      sctx.strokeStyle = "rgba(56, 189, 248, 0.45)";
      sctx.strokeRect(ox, oy, dw, dh);

      // Blur Volume panel (Tweet style: dark capsule + blurred white core + cyan box/crosshair + optical centroid)
      const bx = blurBoxX + (boxW - dw) / 2;
      const by = boxY + (boxH - dh) / 2;
      sctx.fillStyle = "#090a0d";
      sctx.beginPath();
      sctx.roundRect(blurBoxX, boxY, boxW, boxH, 8);
      sctx.fill();

      const bImg = await new Promise((res) => { const im = new Image(); im.onload = () => res(im); im.src = blurDataUrl; });
      sctx.save();
      sctx.beginPath();
      sctx.roundRect(bx, by, dw, dh, Math.min(dw, dh) * 0.25);
      sctx.fillStyle = "#000000";
      sctx.fill();
      sctx.clip();
      sctx.drawImage(bImg, pad, pad, sw, sh, bx, by, dw, dh);
      sctx.restore();

      // Cyan border + center crosshair
      sctx.strokeStyle = "#38bdf8";
      sctx.lineWidth = 1.25;
      sctx.beginPath();
      sctx.roundRect(bx, by, dw, dh, Math.min(dw, dh) * 0.25);
      sctx.stroke();

      sctx.save();
      sctx.setLineDash([3, 3]);
      sctx.strokeStyle = "rgba(56, 189, 248, 0.65)";
      sctx.beginPath();
      sctx.moveTo(bx + dw / 2, by);
      sctx.lineTo(bx + dw / 2, by + dh);
      sctx.moveTo(bx, by + dh / 2);
      sctx.lineTo(bx + dw, by + dh / 2);
      sctx.stroke();
      sctx.restore();

      if (item.category === "icon-text") {
        // 分别画 icon Y 与 text Y
        const iy = by + (item.iconRawCy / item.rect.h) * dh;
        sctx.strokeStyle = "#34d399";
        sctx.lineWidth = 1.4;
        sctx.beginPath();
        sctx.moveTo(bx, iy);
        sctx.lineTo(bx + dw, iy);
        sctx.stroke();
        if (item.textRawCy !== null) {
          const ty = by + (item.textRawCy / item.rect.h) * dh;
          sctx.strokeStyle = "#fb7185";
          sctx.beginPath();
          sctx.moveTo(bx, ty);
          sctx.lineTo(bx + dw, ty);
          sctx.stroke();
        }
      } else {
        // 光学体积中心点 + 偏移向量
        const mCx = bx + dw / 2 + (item.blurDx / item.rect.w) * dw;
        const mCy = by + dh / 2 + (item.blurDy / item.rect.h) * dh;
        sctx.strokeStyle = badgeColor;
        sctx.lineWidth = 1.5;
        sctx.beginPath();
        sctx.moveTo(bx + dw / 2, by + dh / 2);
        sctx.lineTo(mCx, mCy);
        sctx.stroke();
        sctx.fillStyle = badgeColor;
        sctx.beginPath();
        sctx.arc(mCx, mCy, 3.8, 0, Math.PI * 2);
        sctx.fill();
      }

      // Metrics column
      const mx = blurBoxX + boxW + 14;
      sctx.fillStyle = "#9ca3af";
      sctx.font = "11px monospace";
      const lines = [
        `${item.selectorHint.slice(0, 34)}`,
        `box: ${item.rect.w}×${item.rect.h}px`,
        `blur Δ: (${item.blurDx >= 0 ? "+" : ""}${item.blurDx}, ${item.blurDy >= 0 ? "+" : ""}${item.blurDy})px`,
        `raw  Δ: (${item.rawDx >= 0 ? "+" : ""}${item.rawDx}, ${item.rawDy >= 0 ? "+" : ""}${item.rawDy})px`,
        `ink pad L/R: ${item.inkMargins.left} / ${item.inkMargins.right}`,
        item.category === "icon-text"
          ? `icon-text ΔY: ${item.iconTextDeltaY ?? "-"}px`
          : `vol L/R: ${item.volumeRatioLR}  T/B: ${item.volumeRatioTB}`,
      ];
      lines.forEach((ln, li) => {
        sctx.fillStyle = li === 2 || (item.category === "icon-text" && li === 5) ? "#f3f4f6" : "#9ca3af";
        sctx.fillText(ln, mx, boxY + 14 + li * 15);
      });
    }

    // 剥离内部 base64，只返回干净数据
    const cleanResults = sorted.map(({ _crop, ...rest }) => rest);
    return {
      label,
      viewport,
      count: cleanResults.length,
      results: cleanResults,
      overlayPngBase64: overlayCanvas.toDataURL("image/png").split(",")[1],
      sheetPngBase64: sheetCanvas.toDataURL("image/png").split(",")[1],
    };
  }, { origPngBase64, inkPngBase64, dpr, viewport, targets, label });
}

/**
 * 在页面（或同源 iframe）内采集“多层嵌套容器关系”几何信息（L2 控件⊂容器、L3 容器⊂主区、L4 主区/弹框⊂视口）。
 */
export function collectNestedLayoutInFrame() {
  const rectOf = (el) => {
    if (!el) return null;
    const b = el.getBoundingClientRect();
    if (b.width <= 0 || b.height <= 0) return null;
    const cs = getComputedStyle(el);
    if (cs.display === "none" || cs.visibility === "hidden" || parseFloat(cs.opacity) === 0) return null;
    return {
      x: +b.left.toFixed(2),
      y: +b.top.toFixed(2),
      w: +b.width.toFixed(2),
      h: +b.height.toFixed(2),
      r: +(b.left + b.width).toFixed(2),
      b: +(b.top + b.height).toFixed(2),
      cx: +(b.left + b.width / 2).toFixed(2),
      cy: +(b.top + b.height / 2).toFixed(2),
      radius: parseFloat(cs.borderTopLeftRadius) || 0,
      pad: [
        parseFloat(cs.paddingTop) || 0,
        parseFloat(cs.paddingRight) || 0,
        parseFloat(cs.paddingBottom) || 0,
        parseFloat(cs.paddingLeft) || 0,
      ],
      border: [
        parseFloat(cs.borderTopWidth) || 0,
        parseFloat(cs.borderRightWidth) || 0,
        parseFloat(cs.borderBottomWidth) || 0,
        parseFloat(cs.borderLeftWidth) || 0,
      ],
    };
  };
  const q = (s) => document.querySelector(s);
  const qa = (s) => [...document.querySelectorAll(s)];

  return {
    hasChatMain: Boolean(q(".chat-main")),
    isEmpty: q(".chat-main")?.classList.contains("is-empty") ?? false,
    isMobileLayout: Boolean(q("[data-chat-width~='820'], [data-chat-short]")),
    isNarrowComposer: Boolean(q("[data-chat-width~='36rem']")),
    chatMain: rectOf(q(".chat-main")),
    sidebar: rectOf(q(".chat-sidebar")),
    heading: rectOf(q(".conversation-heading")),
    toolsBar: rectOf(q(".conversation-tools-bar")),
    firstMsgP: rectOf(q(".agent-content p, .timeline-output .plain-message-response, .agent-content")),
    userBubble: rectOf(qa(".user-bubble").at(-1)),
    scrollReturn: rectOf(q(".desktop-scroll-return[data-state='open']")),
    homeHero: rectOf(q(".home-hero, .home-state__ready")),
    composerWrap: rectOf(q(".composer-wrap")),
    composer: rectOf(q("form.composer")),
    isComposerCompact: q("form.composer")?.classList.contains("is-compact") ?? false,
    textarea: rectOf(q("form.composer textarea")),
    modelPicker: rectOf(q("form.composer .model-picker__trigger")),
    composerTool: rectOf(q("form.composer .composer-tool")),
    actionBtn: rectOf(q("form.composer .composer-action-button")),
    actionBtnMode: q("form.composer .composer-action-button")?.getAttribute("data-mode") || null,
    composerReply: rectOf(q("form.composer .composer-reply")),
    statsLine: rectOf(q(".composer-stats-line")),
    codeHdr: rectOf(q(".code-block-header")),
    codeTitle: rectOf(q(".code-block-header .code-header-title")),
    codeCopy: rectOf(q(".code-block-header .code-action-btn")),
    settingsDialog: rectOf(q("dialog.shell-settings-dialog[open]")),
    settingsNavH2: rectOf(q("dialog.shell-settings-dialog[open] .shell-settings-nav h2")),
    settingsHeadingH3: rectOf(q("dialog.shell-settings-dialog[open] .shell-settings-heading h3")),
    settingsClose: rectOf(q("dialog.shell-settings-dialog[open] .shell-settings-close")),
  };
}

/**
 * 计算 L2/L3/L4 多层嵌套对齐误差，并在独立 canvas 页中绘制嵌套框线诊断图与发送框模糊体积带。
 */
export async function analyzeNestedInCanvasPage(canvasPage, {
  origPngBase64,
  inkPngBase64,
  dpr,
  viewport,
  layout,
  label,
}) {
  return canvasPage.evaluate(async ({ origB64, inkB64, dpr, viewport, layout, label }) => {
    const loadImg = (b64) => new Promise((res, rej) => {
      const img = new Image();
      img.onload = () => res(img);
      img.onerror = rej;
      img.src = `data:image/png;base64,${b64}`;
    });
    const [origImg, inkImg] = await Promise.all([loadImg(origB64), loadImg(inkB64)]);

    const inkCanvas = document.createElement("canvas");
    inkCanvas.width = inkImg.width;
    inkCanvas.height = inkImg.height;
    const ictx = inkCanvas.getContext("2d", { willReadFrequently: true });
    ictx.drawImage(inkImg, 0, 0);

    const measureInk = (box) => {
      if (!box) return null;
      const sx = Math.max(0, Math.round(box.x * dpr));
      const sy = Math.max(0, Math.round(box.y * dpr));
      const sw = Math.min(inkImg.width - sx, Math.round(box.w * dpr));
      const sh = Math.min(inkImg.height - sy, Math.round(box.h * dpr));
      if (sw < 2 || sh < 2) return null;
      const px = ictx.getImageData(sx, sy, sw, sh).data;
      let sum = 0;
      let cx = 0;
      let cy = 0;
      for (let y = 0; y < sh; y++) {
        for (let x = 0; x < sw; x++) {
          const i = (y * sw + x) * 4;
          const w = Math.max(px[i], px[i + 1], px[i + 2]);
          if (w > 24) {
            sum += w;
            cx += (x + 0.5) * w;
            cy += (y + 0.5) * w;
          }
        }
      }
      if (sum < 10) return null;
      return {
        inkCx: +(box.x + (cx / sum) / dpr).toFixed(2),
        inkCy: +(box.y + (cy / sum) / dpr).toFixed(2),
      };
    };

    const checks = [];
    const addCheck = (level, name, deltaPx, limitPx, detail) => {
      const absDelta = +Math.abs(deltaPx).toFixed(2);
      checks.push({
        view: label,
        level,
        name,
        deltaPx: +deltaPx.toFixed(2),
        absDelta,
        limitPx,
        passed: absDelta <= limitPx,
        detail,
      });
    };

    const c = layout.composer;
    const taInk = measureInk(layout.textarea);
    const mpInk = measureInk(layout.modelPicker);

    // L2: 发送/停止按钮在 form.composer 内的对称与圆角同心度
    if (c && layout.actionBtn) {
      const a = layout.actionBtn;
      const padT = a.y - c.y - c.border[0];
      const padB = c.b - c.border[2] - a.b;
      const padR = c.r - c.border[1] - a.r;
      const btnPaintedR = layout.isNarrowComposer ? 18 : a.h / 2;
      const effectiveInset = (layout.isNarrowComposer ? padR + a.border[1] : padR) + c.border[1];
      if (layout.isComposerCompact) {
        addCheck("L2", "composer-action-vertical-inset", padT - padB, 1.0, { padT: +padT.toFixed(2), padB: +padB.toFixed(2), mode: layout.actionBtnMode });
      }
      addCheck("L2", "composer-action-concentric-radius", c.radius - btnPaintedR - effectiveInset, 1.5, {
        outerRadius: c.radius,
        innerRadius: btnPaintedR,
        inset: +effectiveInset.toFixed(2),
      });
    }

    // L2: 紧凑发送框内部文字与模型胶囊垂直对齐（弹窗打开时遮罩会干扰底层颜色，跳过）
    if (c && layout.isComposerCompact && !layout.settingsDialog && taInk && mpInk) {
      addCheck("L2", "composer-text-vs-model-y", taInk.inkCy - mpInk.inkCy, 1.5, {
        textareaInkCy: taInk.inkCy,
        modelPickerInkCy: mpInk.inkCy,
        composerCy: c.cy,
      });
    }

    // L2: 设置弹窗标题栏 h2/h3 与关闭按钮 × 对齐（仅桌面双栏态，移动端关闭钮位于独立顶栏）
    if (layout.settingsDialog && layout.settingsDialog.w >= 680 && layout.settingsHeadingH3 && layout.settingsClose) {
      addCheck("L2", "settings-close-vs-heading-y", layout.settingsClose.cy - layout.settingsHeadingH3.cy, 1.5, {
        closeCy: layout.settingsClose.cy,
        headingCy: layout.settingsHeadingH3.cy,
        navTitleCy: layout.settingsNavH2?.cy ?? null,
      });
      if (layout.settingsNavH2) {
        addCheck("L2", "settings-nav-vs-heading-y", layout.settingsNavH2.cy - layout.settingsHeadingH3.cy, 2.0, {
          navTitleCy: layout.settingsNavH2.cy,
          headingCy: layout.settingsHeadingH3.cy,
        });
      }
    }

    // L2: 代码块头栏 title 与复制按钮对齐
    if (layout.codeHdr && layout.codeTitle && layout.codeCopy) {
      addCheck("L2", "code-header-title-vs-copy-y", layout.codeTitle.cy - layout.codeCopy.cy, 1.0, {
        headerH: layout.codeHdr.h,
        titleCy: layout.codeTitle.cy,
        copyCy: layout.codeCopy.cy,
      });
    }

    // L3: 宽屏阅读竖线（消息正文左缘 == composer 输入文字左缘）
    if (!layout.isMobileLayout && layout.firstMsgP && layout.textarea && c && !layout.isEmpty) {
      addCheck("L3", "reading-left-line", layout.firstMsgP.x - (c.x + 16), 1.5, {
        msgTextLeft: layout.firstMsgP.x,
        composerInnerTextX: +(c.x + 16).toFixed(2),
      });
    }

    // L3: 底部统计行底距平衡（不贴底死角）
    if (!layout.isNarrowComposer && layout.chatMain && layout.statsLine && c) {
      const topGap = layout.statsLine.y - c.b;
      const bottomGap = layout.chatMain.b - layout.statsLine.b;
      if (bottomGap < 3.5) {
        addCheck("L3", "composer-stats-bottom-clearance", 4.0 - bottomGap, 0.5, {
          topGap: +topGap.toFixed(2),
          bottomGap: +bottomGap.toFixed(2),
        });
      }
    }

    // L3: 空态视觉簇（问候 + 输入框）在 .chat-main 内的垂直光学中心（弹窗遮罩打开时跳过）
    if (layout.isEmpty && !layout.settingsDialog && layout.chatMain && layout.homeHero && c) {
      const m = layout.chatMain;
      const clusterTop = layout.homeHero.y;
      const clusterBot = (layout.composerWrap || c).b;
      // 下方输入卡较上方文字更重，按 46% / 54% 权重估算感知重心
      const clusterOpticalCy = clusterTop + (clusterBot - clusterTop) * 0.54;
      addCheck("L3", "empty-cluster-optical-center-y", clusterOpticalCy - m.cy, 25.0, {
        clusterTop,
        clusterBot,
        clusterOpticalCy: +clusterOpticalCy.toFixed(2),
        chatMainCy: m.cy,
      });
    }

    // L4: 顶栏与工作台栏底边对齐、设置弹窗视口居中
    if (layout.heading && layout.toolsBar) {
      addCheck("L4", "heading-vs-workbench-bottom", layout.heading.b - layout.toolsBar.b, 0.5, {
        headingBottom: layout.heading.b,
        toolsBarBottom: layout.toolsBar.b,
      });
    }
    if (layout.settingsDialog) {
      addCheck("L4", "settings-dialog-viewport-center", Math.hypot(layout.settingsDialog.cx - viewport.width / 2, layout.settingsDialog.cy - viewport.height / 2), 2.0, {
        dialogCx: layout.settingsDialog.cx,
        dialogCy: layout.settingsDialog.cy,
      });
    }

    // 绘制带嵌套框线标注的整页图
    const out = document.createElement("canvas");
    out.width = origImg.width;
    out.height = origImg.height;
    const octx = out.getContext("2d");
    octx.drawImage(origImg, 0, 0);

    const drawBox = (box, color, tag, dashed = false) => {
      if (!box) return;
      octx.save();
      octx.strokeStyle = color;
      octx.lineWidth = 1.5 * dpr;
      if (dashed) octx.setLineDash([4 * dpr, 4 * dpr]);
      octx.strokeRect(box.x * dpr, box.y * dpr, box.w * dpr, box.h * dpr);
      if (tag) {
        octx.fillStyle = color;
        octx.font = `600 ${11 * dpr}px monospace`;
        octx.fillText(tag, box.x * dpr + 4 * dpr, Math.max(12 * dpr, box.y * dpr - 4 * dpr));
      }
      octx.restore();
    };
    drawBox(layout.chatMain, "#a855f7", "L4 chat-main", true);
    drawBox(layout.settingsDialog, "#a855f7", "L4 settings-dialog", true);
    drawBox(layout.composerWrap, "#f97316", "L3 composer-wrap", true);
    drawBox(layout.composer, "#38bdf8", "L2 form.composer");
    drawBox(layout.actionBtn, "#22c55e", "L1 action");
    drawBox(layout.composerTool, "#22c55e", "");
    drawBox(layout.modelPicker, "#22c55e", "");
    drawBox(layout.statsLine, "#f97316", "");
    drawBox(layout.scrollReturn, "#f97316", "L3 return");

    if (layout.firstMsgP) {
      octx.save();
      octx.strokeStyle = "rgba(251, 113, 133, 0.85)";
      octx.lineWidth = 1.2 * dpr;
      octx.setLineDash([5 * dpr, 3 * dpr]);
      octx.beginPath();
      octx.moveTo(layout.firstMsgP.x * dpr, 0);
      octx.lineTo(layout.firstMsgP.x * dpr, out.height);
      octx.stroke();
      octx.restore();
    }

    // 生成输入卡（L2）原图 vs 模糊体积对照长条
    let composerPngBase64 = null;
    if (c) {
      const sx = Math.max(0, Math.round((c.x - 16) * dpr));
      const sy = Math.max(0, Math.round((c.y - 16) * dpr));
      const sw = Math.min(origImg.width - sx, Math.round((c.w + 32) * dpr));
      const sh = Math.min(origImg.height - sy, Math.round((c.h + 36) * dpr));
      const comp = document.createElement("canvas");
      comp.width = sw;
      comp.height = sh * 2 + 10 * dpr;
      const cctx = comp.getContext("2d");
      cctx.fillStyle = "#111317";
      cctx.fillRect(0, 0, comp.width, comp.height);
      cctx.drawImage(origImg, sx, sy, sw, sh, 0, 0, sw, sh);
      cctx.save();
      cctx.fillStyle = "#080a0d";
      cctx.fillRect(0, sh + 10 * dpr, sw, sh);
      cctx.filter = `blur(${4 * dpr}px)`;
      cctx.drawImage(inkImg, sx, sy, sw, sh, 0, sh + 10 * dpr, sw, sh);
      cctx.restore();

      const ox = c.x * dpr - sx;
      const oy = c.y * dpr - sy + sh + 10 * dpr;
      cctx.strokeStyle = "#38bdf8";
      cctx.lineWidth = 1.5 * dpr;
      cctx.strokeRect(ox, oy, c.w * dpr, c.h * dpr);
      cctx.setLineDash([3 * dpr, 3 * dpr]);
      cctx.strokeStyle = "rgba(56, 189, 248, 0.55)";
      cctx.beginPath();
      cctx.moveTo(ox, oy + (c.h * dpr) / 2);
      cctx.lineTo(ox + c.w * dpr, oy + (c.h * dpr) / 2);
      cctx.stroke();
      cctx.setLineDash([]);
      for (const sub of [layout.textarea, layout.modelPicker, layout.composerTool, layout.actionBtn]) {
        if (!sub) continue;
        cctx.strokeStyle = "#22c55e";
        cctx.lineWidth = dpr;
        cctx.strokeRect(sub.x * dpr - sx, sub.y * dpr - sy + sh + 10 * dpr, sub.w * dpr, sub.h * dpr);
      }
      composerPngBase64 = comp.toDataURL("image/png").split(",")[1];
    }

    return {
      checks,
      nestedPngBase64: out.toDataURL("image/png").split(",")[1],
      composerPngBase64,
    };
  }, { origB64: origPngBase64, inkB64: inkPngBase64, dpr, viewport, layout, label });
}
