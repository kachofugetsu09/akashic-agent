api_version = 3
name = "workbench-ui"
version = "1.0.0"
dashboard_module = "dashboard.py"
web_module = "web_module.js"
workspace_files = ("sessions.db",)
web_requires = ("shell.pages.v1",)
web_provides = ("workbench.panels.v2",)
web_contract_digests = {
    "workbench.panels.v2": "fb6417c9bf532c1fdb344767d06065d5d3293da85deb64eff1e8088889a33bcb",
}


def apply(ctx, config):
    pass
