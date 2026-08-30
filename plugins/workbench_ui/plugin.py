api_version = 3
name = "workbench-ui"
version = "1.0.0"
dashboard_module = "dashboard.py"
web_module = "web_module.js"
workspace_files = ("sessions.db",)
web_requires = ("shell.pages.v1",)
web_provides = ("workbench.panels.v2",)
web_contract_digests = {
    "workbench.panels.v2": "17a005a381b362ae25a0499dbf95bf7a2c3ff0bb4e9b415e7357db458de6b5db",
}


def apply(ctx, config):
    pass
