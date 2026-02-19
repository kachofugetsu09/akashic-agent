import platform


class ContextBuilder:

    def build_system_prompt(self) -> str:
        parts = []
        # 核心identity
        parts.append(self._get_identity())

    # def _get_identity(self) ->str :
    #     now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
    #     tz = _time.strftime("%Z") or "UTC"
    #     system = platform.system()
    #     runtime = platform.machine()
    #     python_version = platform.python_version()
    #
    #     return f""" akasic-bot
    #     you are akasic-bot , a helpful AI assistant. You have access to tools that allow you to:
    #     - Read, write and edit files
    #     - Execute shell commands
    #     - Search the web and fetch web pages
    #     -Send messages to users on chat channels
    #
    #     ## Current Time
    #     {now}({tz})
    #
    #     ## Runtime
    #     {runtime}
    #
    #     Your workspace is at: {workspace_path}
    #
    #
    #     """

