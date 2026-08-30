function checkView(view) {
  if (!view || typeof view.requestAttention !== "function") {
    throw new Error("Computer 缺少 conversation.tools.v1 view");
  }
  return view;
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
      const controls = document.createElement("div");
      controls.className = "computer-controls";
      const keys = document.createElement("div");
      keys.className = "computer-keys";
      keys.setAttribute("aria-label", "浏览器按键");
      for (const [label, accessibleLabel, key] of [
        ["⇧ Tab", "上一个控件", "Shift+Tab"],
        ["Tab", "下一个控件", "Tab"],
        ["Enter", "确认", "Enter"],
        ["Esc", "返回", "Escape"],
      ]) {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = label;
        button.setAttribute("aria-label", accessibleLabel);
        button.addEventListener("click", () => void sendInput({ action: "key", key }));
        keys.appendChild(button);
      }
      const typeForm = document.createElement("form");
      typeForm.className = "computer-type";
      const typeLabel = document.createElement("label");
      typeLabel.textContent = "输入";
      const typeInput = document.createElement("input");
      typeInput.type = "password";
      typeInput.autocomplete = "off";
      typeInput.spellcheck = false;
      typeInput.placeholder = "发送文字到浏览器";
      const reveal = document.createElement("button");
      reveal.type = "button";
      reveal.textContent = "显示";
      reveal.setAttribute("aria-pressed", "false");
      reveal.addEventListener("click", () => {
        const visible = typeInput.type === "text";
        typeInput.type = visible ? "password" : "text";
        reveal.textContent = visible ? "显示" : "隐藏";
        reveal.setAttribute("aria-pressed", String(!visible));
        typeInput.focus();
      });
      const send = document.createElement("button");
      send.type = "submit";
      send.textContent = "发送";
      typeLabel.appendChild(typeInput);
      typeForm.append(typeLabel, reveal, send);
      controls.append(keys, typeForm);
      root.append(stage, controls);
      host.replaceChildren(root);

      let imageUrl = "";
      let lastNotice = null;
      let screenshotBusy = false;
      let wasPanelActive = false;

      async function sendInput(payload) {
        try {
          const response = await ctx.http.request("/api/dashboard/computer/input", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify(payload),
          });
          if (!response.ok) throw new Error(`input ${response.status}`);
          status.textContent = "已发送";
          window.setTimeout(() => void loadScreenshot(), 150);
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
        return { active: activity.active, newNotice };
      }

      async function loadScreenshot() {
        if (screenshotBusy || !view.active) return;
        screenshotBusy = true;
        try {
          const response = await ctx.http.request("/api/dashboard/computer/screenshot");
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
        const activity = await loadActivity();
        if (view.active && (activity.active || activity.newNotice || !wasPanelActive)) {
          await loadScreenshot();
        }
        wasPanelActive = view.active;
      }

      image.addEventListener("click", (event) => {
        stage.focus();
        const bounds = image.getBoundingClientRect();
        if (!bounds.width || !bounds.height) return;
        const x = Math.min(1279, Math.max(0, Math.floor(
          (event.clientX - bounds.left) * 1280 / bounds.width,
        )));
        const y = Math.min(799, Math.max(0, Math.floor(
          (event.clientY - bounds.top) * 800 / bounds.height,
        )));
        void sendInput({ action: "click", x, y });
      });
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
        if (imageUrl) URL.revokeObjectURL(imageUrl);
        host.replaceChildren();
      };
    },
  }));
}
