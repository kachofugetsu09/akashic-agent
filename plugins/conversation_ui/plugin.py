api_version = 3
name = "conversation-ui"
version = "1.0.0"
web_module = "web_module.js"
web_requires = ("shell.pages.v1",)
web_provides = ("conversation.tools.v1",)
web_contract_digests = {
    "conversation.tools.v1": "1ce9b1dfe70907c50c00c17bc428c8ddab91d4c4839db9da13b0283aad1035fb",
}


def apply(ctx, config):
    pass
