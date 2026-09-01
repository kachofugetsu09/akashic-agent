api_version = 3
name = "conversation-ui"
version = "1.0.0"
web_module = "web_module.js"
web_requires = ("shell.pages.v1",)
web_provides = ("conversation.tools.v1",)
web_contract_digests = {
    "conversation.tools.v1": "ed47d69b84e946e27a2e297634e96bcc6afc72a3d3089caac1a14632703efb54",
}


def apply(ctx, config):
    pass
