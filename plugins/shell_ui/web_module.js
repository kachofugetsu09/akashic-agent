const PAGE_MOUNT = "shell.pages.v1";

export function activate(ctx) {
  return ctx.ui.inject("web.root.v1", (mount) => mount.register({
    id: "shell",
    children: [{id: PAGE_MOUNT, cardinality: "list"}],
    render(host, view) {
      const pages = view.child(PAGE_MOUNT);
      const shell = document.createElement("div");
      shell.className = "shell-ui-root";
      const header = document.createElement("header");
      header.className = "shell-ui-band";
      header.setAttribute("aria-label", "Akashic 主导航");
      const brand = document.createElement("div");
      brand.className = "shell-ui-brand";
      brand.innerHTML = '<span class="shell-ui-mark" aria-hidden="true">◉</span><strong>Akashic</strong>';
      const nav = document.createElement("nav");
      nav.className = "shell-ui-nav";
      nav.setAttribute("aria-label", "主要功能");
      const pageHost = document.createElement("main");
      pageHost.className = "shell-ui-page-stack";
      header.append(brand, nav);
      shell.append(header, pageHost);
      host.replaceChildren(shell);

      let disposePage = () => {};
      const entries = pages.entries.map((entry) => {
        if (typeof entry.label !== "string" || typeof entry.route !== "string") {
          throw new Error(`页面 ${entry.id} 缺少 label 或 route`);
        }
        return entry;
      });
      if (new Set(entries.map((entry) => entry.route)).size !== entries.length) {
        throw new Error("页面 route 不能重复");
      }
      const defaultPage = entries.find((entry) => entry.route === "") ?? entries[0];
      const renderActive = () => {
        const route = window.location.hash.slice(1);
        const active = entries.find((entry) => entry.route === route) ?? defaultPage;
        disposePage();
        pageHost.replaceChildren();
        if (active) disposePage = pages.render(active.id, pageHost);
        for (const button of nav.querySelectorAll("button")) {
          const selected = button.dataset.pageId === active?.id;
          button.classList.toggle("is-active", selected);
          if (selected) button.setAttribute("aria-current", "page");
          else button.removeAttribute("aria-current");
        }
      };
      for (const entry of entries) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "shell-ui-page-button";
        button.dataset.pageId = entry.id;
        button.textContent = entry.label;
        button.addEventListener("click", () => {
          const base = `${window.location.pathname}${window.location.search}`;
          window.history.pushState(null, "", entry.route ? `${base}#${entry.route}` : base);
          window.dispatchEvent(new PopStateEvent("popstate"));
        });
        nav.appendChild(button);
      }
      window.addEventListener("popstate", renderActive);
      renderActive();
      return () => {
        window.removeEventListener("popstate", renderActive);
        disposePage();
        host.replaceChildren();
      };
    },
  }));
}
