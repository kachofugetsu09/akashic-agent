#!/usr/bin/env bun
/**
 * akashic 的 harness-perf-benchmark 接入：由 CI 拷到上游 checkout 的 playground/akashic/ 下运行，
 * 复用上游的 mock、编排、分段计时与 GB·秒积分。
 *
 *   AK_ROOT=<被测源码> AK_SIDE=<prepare.py 的 --side> AK_TOOLS=<本目录的上一级> AK_PYTHON=<python> \
 *     bun playground/akashic/perf-demo.ts --script <剧本> --out-dir <目录> --work-dir <目录>
 *
 * 被测命令是 AK_TOOLS/measured.py（A/B 两侧共用同一份测量代码）：gateway 冷启动 → `exec --new --final-only` 跑完剧本 → SIGTERM。
 * 每次运行前从模板恢复 workspace；插件 home（字节码、依赖环境等宿主缓存）跨次保留。
 *
 * 上游采样器只提供 macOS 的 proc_pid_rusage 后端，Linux 上会退到 ps（CPU 分辨率约 60ms）。
 * 这里注入 /proc 读法，并把进程树刷新改为每拍一次：gateway 由 measured.py 拉起，
 * 默认 2s 刷新会漏掉整个启动段的内存。CPU 不取采样值，而取 measured.py 的 wait4 精确值。
 */
import { appendFileSync, cpSync, readFileSync, readdirSync, rmSync } from "node:fs";
import { join } from "node:path";
import { REPO_ROOT, loadPerfConfig, type PerfConfig } from "../../scripts/perf/config";
import { EXIT_SETUP, runPerf } from "../../scripts/perf/run";
import {
    ProcessSampler,
    type CumulativeReading,
    type SamplerBackend,
    type SamplerKind,
} from "../../scripts/perf/sampler";

const TICK_NS = 1e9 / 100;
const PAGE_BYTES = 4096;

function statFields(pid: number | string): string[] {
    const stat = readFileSync(`/proc/${pid}/stat`, "utf8");
    return stat.slice(stat.lastIndexOf(")") + 2).split(" ");
}

// /proc/<pid>/stat 的 utime/stime/cutime/cstime 与 statm 的常驻页数。
const procfsBackend: SamplerBackend = {
    kind: "procfs" as SamplerKind,
    describe: () => "/proc/<pid>/stat + statm（tick 10ms，含已回收子进程）",
    read(pid: number): CumulativeReading | null {
        try {
            const rest = statFields(pid);
            if (rest[0] === "Z") return null;
            const statm = readFileSync(`/proc/${pid}/statm`, "utf8").split(" ");
            return {
                cpuNs: (Number(rest[11]) + Number(rest[12])) * TICK_NS,
                childCpuNs: (Number(rest[13]) + Number(rest[14])) * TICK_NS,
                rssBytes: Number(statm[1]) * PAGE_BYTES,
            };
        } catch {
            return null;
        }
    },
};

function listProcfsProcesses(): Array<{ pid: number; ppid: number }> {
    const rows: Array<{ pid: number; ppid: number }> = [];
    for (const name of readdirSync("/proc")) {
        if (!/^\d+$/.test(name)) continue;
        try {
            rows.push({ pid: Number(name), ppid: Number(statFields(name)[1]) });
        } catch {
            // 进程在扫描期间退出。
        }
    }
    return rows;
}

type TreeOptions = { options: { listProcesses: () => Array<{ pid: number; ppid: number }> } };
const originalSample = ProcessSampler.prototype.sample;
ProcessSampler.prototype.sample = function (timing) {
    (this as unknown as TreeOptions).options.listProcesses = listProcfsProcesses;
    this.refreshTree();
    return originalSample.call(this, timing);
};

function required(name: string): string {
    const value = process.env[name];
    if (value === undefined || value === "") throw new Error(`缺少环境变量 ${name}`);
    return value;
}

// mock 外包一层请求记录：校验每轮工具结果与上下文体积都依赖它。
function mockCommand(cfg: PerfConfig, wire: string): string[] {
    return [process.execPath, join(import.meta.dir, "mock.ts"), wire,
        "--script", cfg.scriptPath, "--port", String(cfg.port), "--exhausted", cfg.exhausted];
}

try {
    const root = required("AK_ROOT");
    const side = required("AK_SIDE");
    const tools = required("AK_TOOLS");
    const python = required("AK_PYTHON");
    const config = loadPerfConfig(process.argv.slice(2), REPO_ROOT);
    config.harnessId = "akashic";
    config.exhausted = "error";
    rmSync(join(side, "state/workspace"), { recursive: true, force: true });
    cpSync(join(side, "template"), join(side, "state/workspace"), { recursive: true });
    const wire = join(config.outDir, "wire.jsonl");
    appendFileSync(wire, "");
    process.exitCode = await runPerf(config, {
        backend: procfsBackend,
        harnessCommand: (cfg) => [python, join(tools, "measured.py"),
            "--root", root, "--side", side, "--out", cfg.outDir, "--prompt", cfg.prompt],
        mockCommand: (cfg) => mockCommand(cfg, wire),
    });
} catch (error) {
    console.error(`[perf] 启动失败: ${(error as Error).message}`);
    process.exitCode = EXIT_SETUP;
}
