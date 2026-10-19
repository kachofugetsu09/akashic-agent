import { Check, X } from "lucide-react";
import { formatDuration, render, ToolData, ToolText, type PluginUiDefinition } from "@akashic/chat-ui-v1";
import "./plugin-ui.css";

interface ShellResult {
  command: string;
  output: string;
  exitCode?: number;
  wallTimeMs?: number;
  status?: string;
  rest: Record<string, unknown>;
}

// 识别本插件的命令执行结果：至少带命令与输出字符串；其余字段收进原始数据。
function readShellResult(value: unknown): ShellResult | null {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return null;
  const { command, output, exit_code: exitCode, wall_time_ms: wallTimeMs, process_status: status, ...rest } = value as Record<string, unknown>;
  if (typeof command !== "string" || typeof output !== "string") return null;
  return {
    command,
    output,
    exitCode: typeof exitCode === "number" ? exitCode : undefined,
    wallTimeMs: typeof wallTimeMs === "number" ? wallTimeMs : undefined,
    status: typeof status === "string" ? status : undefined,
    rest: { ...rest, ...(typeof status === "string" ? { process_status: status } : {}) },
  };
}

// 命令块：$ 命令 → 输出（预览看末尾）→ 底栏结果与耗时。
function ShellResultView({ result }: { result: ShellResult }) {
  const failed = result.exitCode !== undefined && result.exitCode !== 0;
  return <>
    <div className="shell-result">
      <div className="shell-result-command"><span aria-hidden="true">$</span><span>{result.command}</span></div>
      {result.output.trim()
        ? <ToolText text={result.output} tail className="tool-result" />
        : <p className="shell-result-empty">无输出</p>}
      <div className="shell-result-foot">
        {result.exitCode === undefined ? <span>{result.status ?? "运行中"}</span>
          : <span className={`shell-result-exit${failed ? " is-failed" : ""}`}>
            {failed ? <X size={13} aria-hidden="true" /> : <Check size={13} aria-hidden="true" />}
            {failed ? `退出码 ${result.exitCode}` : "成功"}
          </span>}
        {result.wallTimeMs !== undefined ? <span>{formatDuration(result.wallTimeMs)}</span> : null}
      </div>
    </div>
    {Object.keys(result.rest).length ? <details className="shell-result-raw">
      <summary>原始数据</summary>
      <ToolData value={result.rest} />
    </details> : null}
  </>;
}

const definition: PluginUiDefinition = {
  slots: {
    // 宿主只把本插件注册的工具结果交过来；不是命令结果形状时退回通用展示。
    "tool.result": {
      accepts: (block) => readShellResult(block.value) !== null,
      mount(host, context) {
        const result = readShellResult(context.block.value);
        return result ? render(host, <ShellResultView result={result} />) : undefined;
      },
    },
  },
};
export default definition;
