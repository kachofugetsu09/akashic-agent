function checkView(view) {
  if (!view || typeof view.requestAttention !== "function") {
    throw new Error("Computer 缺少 conversation.tools.v1 view");
  }
  return view;
}

export function browserPoint(bounds, clientX, clientY, sourceWidth = 1280, sourceHeight = 800) {
  if (!bounds.width || !bounds.height || !sourceWidth || !sourceHeight) return null;
  const scale = Math.min(bounds.width / sourceWidth, bounds.height / sourceHeight);
  const shownWidth = sourceWidth * scale;
  const shownHeight = sourceHeight * scale;
  const shownLeft = bounds.left + (bounds.width - shownWidth) / 2;
  const shownTop = bounds.top + (bounds.height - shownHeight) / 2;
  const shownX = clientX - shownLeft;
  const shownY = clientY - shownTop;
  if (shownX < 0 || shownY < 0 || shownX >= shownWidth || shownY >= shownHeight) return null;
  return {
    x: Math.min(sourceWidth - 1, Math.floor(shownX / scale)),
    y: Math.min(sourceHeight - 1, Math.floor(shownY / scale)),
  };
}

export function activate(ctx) {
  return ctx.ui.inject("conversation.tools.v1", (mount) => mount.register({
    id: "browser",
    label: "Browser",
    order: 10,
    render(host, _entryView, props) {
      const view = checkView(props);
      const root = document.createElement("div");
      root.className = "computer-view";
      const status = document.createElement("span");
      status.className = "computer-status";
      status.setAttribute("role", "status");
      status.textContent = "正在连接";
      const stage = document.createElement("div");
      stage.className = "computer-stage";
      stage.tabIndex = 0;
      stage.setAttribute("role", "group");
      stage.setAttribute("aria-label", "浏览器画面。选择画面后，可使用 Enter 和方向键操作。");
      const image = document.createElement("img");
      image.alt = "Agent 当前操作的浏览器画面";
      image.draggable = false;
      const empty = document.createElement("div");
      empty.className = "computer-empty";
      empty.textContent = "Agent 使用浏览器时，画面会显示在这里";
      stage.append(image, empty, status);
      const typeForm = document.createElement("form");
      typeForm.className = "computer-type";
      const typeLabel = document.createElement("label");
      typeLabel.textContent = "发送文字";
      const typeInput = document.createElement("input");
      typeInput.type = "password";
      typeInput.autocomplete = "off";
      typeInput.spellcheck = false;
      typeInput.placeholder = "输入后按 Enter";
      typeLabel.appendChild(typeInput);
      typeForm.appendChild(typeLabel);
      root.append(stage, typeForm);
      host.replaceChildren(root);

      let imageUrl = "";
      let lastNotice = null;
      let screenshotBusy = false;
      let scrollTimer = 0;
      let scrollPixels = 0;

      async function sendInput(payload) {
        try {
          const response = await ctx.http.request("/api/dashboard/computer/input", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify(payload),
          });
          if (!response.ok) throw new Error(`input ${response.status}`);
          status.textContent = "已发送";
          void loadScreenshot();
        } catch {
          status.textContent = "发送失败，请重试";
        }
      }

      async function loadActivity() {
        const response = await ctx.http.request("/api/dashboard/computer/activity");
        if (!response.ok) throw new Error(`activity ${response.status}`);
        const activity = await response.json();
        if (!activity || !Number.isInteger(activity.noticeId) || typeof activity.active !== "boolean") {
          throw new Error("Computer activity 回执无效");
        }
        status.textContent = activity.active ? "Agent 正在操作" : "已就绪";
        status.classList.toggle("is-active", activity.active);
        const newNotice = lastNotice !== null && activity.noticeId !== lastNotice;
        if (newNotice) {
          view.requestAttention(`computer:${activity.noticeId}`);
        }
        lastNotice = activity.noticeId;
      }

      async function loadScreenshot() {
        if (screenshotBusy || !view.active) return;
        screenshotBusy = true;
        try {
          const response = await ctx.http.request(
            `/api/dashboard/computer/screenshot?tick=${Date.now()}`,
            { cache: "no-store" },
          );
          if (!response.ok) throw new Error(`screenshot ${response.status}`);
          const next = URL.createObjectURL(await response.blob());
          if (imageUrl) URL.revokeObjectURL(imageUrl);
          imageUrl = next;
          image.src = next;
          empty.hidden = true;
        } finally {
          screenshotBusy = false;
        }
      }

      async function refresh() {
        await loadActivity();
        if (view.active) await loadScreenshot();
      }

      image.addEventListener("click", (event) => {
        stage.focus();
        const bounds = image.getBoundingClientRect();
        const point = browserPoint(
          bounds,
          event.clientX,
          event.clientY,
          image.naturalWidth || 1280,
          image.naturalHeight || 800,
        );
        if (point) void sendInput({ action: "click", ...point });
      });
      stage.addEventListener("wheel", (event) => {
        if (!view.active || !event.deltaY) return;
        event.preventDefault();
        scrollPixels += event.deltaY;
        if (scrollTimer) return;
        scrollTimer = window.setTimeout(() => {
          const amount = Math.sign(scrollPixels) * Math.min(
            10,
            Math.max(1, Math.round(Math.abs(scrollPixels) / 100)),
          );
          scrollPixels = 0;
          scrollTimer = 0;
          void sendInput({ action: "scroll", amount });
        }, 80);
      }, { passive: false });
      stage.addEventListener("keydown", (event) => {
        const key = event.key;
        if (!["Enter", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(key)) {
          return;
        }
        event.preventDefault();
        void sendInput({ action: "key", key });
      });
      typeForm.addEventListener("submit", (event) => {
        event.preventDefault();
        if (!typeInput.value) return;
        const text = typeInput.value;
        typeInput.value = "";
        void sendInput({ action: "type", text });
      });

      const poll = window.setInterval(() => {
        void refresh().catch(() => { status.textContent = "连接中断，正在重试"; });
      }, 800);
      void refresh().catch(() => { status.textContent = "连接中断，正在重试"; });
      return () => {
        window.clearInterval(poll);
        if (scrollTimer) window.clearTimeout(scrollTimer);
        if (imageUrl) URL.revokeObjectURL(imageUrl);
        host.replaceChildren();
      };
    },
  }));
}
