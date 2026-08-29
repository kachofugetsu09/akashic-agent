export function activate(ctx) {
  return ctx.ui.inject("shell.pages.v1", (mount) => mount.register({
    id: "workbench",
    label: "工作台",
    route: "workbench",
    order: 20,
    children: [{id: "workbench.panels.v1", cardinality: "list"}],
    render(host, view) {
      const panels = view.child("workbench.panels.v1");
      const page = document.createElement("div");
      page.className = "workbench-plugin-page";
      page.innerHTML = `<aside><header><h1>工作台</h1><p>由已安装插件提供</p></header><nav aria-label="工作台面板"></nav><p data-errors role="alert"></p></aside><main tabindex="-1"></main>`;
      host.replaceChildren(page);
      const navigation = page.querySelector("nav");
      const errors = page.querySelector("[data-errors]");
      const content = page.querySelector("main");
      let activeId = "";
      let disposePanel = () => {};

      const show = (entry) => {
        disposePanel();
        disposePanel = () => {};
        activeId = entry.id;
        for (const button of navigation.querySelectorAll("button")) {
          const selected = button.dataset.panelId === activeId;
          button.classList.toggle("is-active", selected);
          button.setAttribute("aria-current", selected ? "page" : "false");
        }
        content.replaceChildren();
        try {
          disposePanel = panels.render(entry.id, content);
          content.focus();
        } catch (reason) {
          const error = document.createElement("p");
          error.className = "workbench-panel-error";
          error.setAttribute("role", "alert");
          error.textContent = reason instanceof Error ? reason.message : String(reason);
          content.replaceChildren(error);
        }
      };

      const entries = panels.entries.filter((entry) => {
        if (typeof entry.label === "string" && entry.label.trim()) return true;
        errors.textContent += `${entry.id} 缺少面板名称。 `;
        return false;
      });
      if (!entries.length) {
        const empty = document.createElement("p");
        empty.className = "workbench-panel-empty";
        empty.textContent = "尚未安装提供工作台面板的插件。";
        content.replaceChildren(empty);
      } else {
        for (const entry of entries) {
          const button = document.createElement("button");
          button.type = "button";
          button.dataset.panelId = entry.id;
          button.textContent = entry.label;
          button.addEventListener("click", () => show(entry));
          navigation.append(button);
        }
        show(entries[0]);
      }
      return () => {
        disposePanel();
        host.replaceChildren();
      };
    },
  }));
}
