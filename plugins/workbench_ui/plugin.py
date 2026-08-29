api_version = 3
name = "workbench-ui"
version = "1.0.0"
web_module = "web_module.js"
web_requires = ("shell.pages.v1",)
web_provides = ("workbench.panels.v1",)
web_contract_digests = {
    "workbench.panels.v1": "724b282c22c4b3f3a36967ab664c4dfd8bce4257665f99459000306938caf527",
}


def apply(ctx, config):
    pass
