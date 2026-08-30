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
      const header = document.createElement("header");
      header.className = "computer-header";
      const title = document.createElement("div");
      title.innerHTML = "<strong>Computer</strong><span>持久浏览器</span>";
      const status = document.createElement("span");
      status.className = "computer-status";
      status.setAttribute("role", "status");
      status.textContent = "正在连接";
      header.append(title, status);
      const stage = document.createElement("div");
      stage.className = "computer-stage";
      const image = document.createElement("img");
      image.alt = "Agent 当前操作的浏览器画面";
      image.draggable = false;
      const empty = document.createElement("div");
      empty.className = "computer-empty";
      empty.textContent = "Agent 使用浏览器时，画面会显示在这里";
      stage.append(image, empty);
      const controls = document.createElement("div");
      controls.className = "computer-controls";
      const keys = document.createElement("div");
      keys.className = "computer-keys";
      keys.setAttribute("aria-label", "浏览器按键");
      for (const [label, key] of [["上一个", "Shift+Tab"], ["下一个", "Tab"], ["确认", "Enter"], ["返回", "Escape"]]) {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = label;
        button.addEventListener("click", () => void sendInput({ action: "key", key }));
        keys.appendChild(button);
      }
      const typeForm = document.createElement("form");
      typeForm.className = "computer-type";
      const typeLabel = document.createElement("label");
      typeLabel.textContent = "发送文字";
      const typeInput = document.createElement("input");
      typeInput.type = "password";
      typeInput.autocomplete = "off";
      typeInput.spellcheck = false;
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
      const hint = document.createElement("p");
      hint.className = "computer-hint";
      hint.textContent = "需要登录时，可点画面并发送文字。停用插件会关闭浏览器，但保留登录状态。";
      root.append(header, stage, controls, hint);
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
          status.textContent = "操作已发送";
          window.setTimeout(() => void loadScreenshot(), 150);
        } catch {
          status.textContent = "操作失败，请重试";
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
        const bounds = image.getBoundingClientRect();
        if (!bounds.width || !bounds.height) return;
        const x = Math.round((event.clientX - bounds.left) * 1280 / bounds.width);
        const y = Math.round((event.clientY - bounds.top) * 800 / bounds.height);
        void sendInput({ action: "click", x, y });
      });
      typeForm.addEventListener("submit", (event) => {
        event.preventDefault();
        if (!typeInput.value) return;
        const text = typeInput.value;
        typeInput.value = "";
        void sendInput({ action: "type", text });
      });

      const poll = window.setInterval(() => {
        void refresh().catch(() => { status.textContent = "连接中断"; });
      }, 800);
      void refresh().catch(() => { status.textContent = "连接中断"; });
      return () => {
        window.clearInterval(poll);
        if (imageUrl) URL.revokeObjectURL(imageUrl);
        host.replaceChildren();
      };
    },
  }));
}
