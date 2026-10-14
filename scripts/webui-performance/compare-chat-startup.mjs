/** 交错切换真实聊天制品，按相邻冷/热样本比较，降低宿主负载漂移的影响。 */
import assert from "node:assert/strict";
import { spawnSync, execFileSync } from "node:child_process";
import { cpSync, existsSync, mkdirSync, readFileSync, readdirSync, realpathSync, utimesSync, writeFileSync } from "node:fs";
import { loadavg } from "node:os";
import { basename, dirname, resolve, sep } from "node:path";
import { parseArgs } from "node:util";

const { values } = parseArgs({ options: {
  url: { type: "string" }, browser: { type: "string" }, output: { type: "string" },
  baseline: { type: "string" }, candidate: { type: "string" }, "runtime-chat": { type: "string" },
  runs: { type: "string", default: "7" }, profiles: { type: "string", default: "lan,latency,limited" },
} });
for (const key of ["url", "browser", "output", "baseline", "candidate", "runtime-chat"]) {
  if (!values[key]) throw new Error(`需要 --${key}`);
}
// 校验与后续写入必须针对同一个规范真实路径：词法 resolve() 不解析符号链接。
const target = realpathSync(resolve(values["runtime-chat"]));
const builds = {
  baseline: realpathSync(resolve(values.baseline)),
  candidate: realpathSync(resolve(values.candidate)),
};
// output 尚不存在，先锚定到已存在父目录的真实路径，创建前即可做位置校验。
const output = resolve(realpathSync(dirname(resolve(values.output))), basename(resolve(values.output)));
assert.ok(Object.values(builds).every(path => existsSync(resolve(path, "index.html"))));

// 一次性 fixture 身份：目标真实路径必须落在一个由本次运行创建、带隔离标记的
// workspace 内，并且与任何被测构建目录不相同、不互相嵌套。目录名后缀或
// localhost 都不构成隔离证明；resolve() 是词法解析，不解析符号链接。
const FIXTURE_MARKER = ".akashic-oneoff-fixture.json";

function isInside(child, parent) {
  return child === parent || child.startsWith(parent + sep);
}

function findFixtureMarker(start) {
  let directory = start;
  for (let depth = 0; depth < 8; depth += 1) {
    const candidate = resolve(directory, FIXTURE_MARKER);
    if (existsSync(candidate)) return candidate;
    const parent = dirname(directory);
    if (parent === directory) break;
    directory = parent;
  }
  return null;
}

assert.ok(target.endsWith("/plugins/akashic_clients/static/chat"),
  `runtime-chat 不是聊天静态目录: ${target}`);
assert.ok(Object.values(builds).every(path => path !== target), "runtime-chat 不能就是被测构建目录");
assert.ok(Object.values(builds).every(path => !isInside(target, path) && !isInside(path, target)),
  "runtime-chat 与被测构建目录不得互相嵌套");

const marker = findFixtureMarker(target);
assert.ok(marker, `缺少一次性 fixture 隔离标记 ${FIXTURE_MARKER}: ${target}`);
let fixture;
try {
  fixture = JSON.parse(readFileSync(marker, "utf8"));
} catch (error) {
  throw new Error(`隔离标记不可读: ${marker}: ${error.message}`);
}
assert.equal(fixture.kind, "akashic-webui-oneoff", "隔离标记类型不符");
for (const key of ["state", "workspace", "chat"]) {
  assert.ok(typeof fixture[key] === "string" && fixture[key], `隔离标记缺少 ${key}`);
}
// 资产根是 state/plugins 下的 chat 目录，与消息 workspace 不是同一个根；必须由 fixture
// 直接发布实际允许写入的 chat 目录并精确比对，不能用「某个祖先包含目标」代替。
assert.equal(realpathSync(fixture.chat), target, "隔离标记声明的 chat 目录与目标不一致");
assert.ok(isInside(target, realpathSync(fixture.state)), "目标不在标记声明的 state 内");
assert.ok(isInside(realpathSync(fixture.workspace), realpathSync(fixture.state)),
  "标记的 workspace 不在 state 内");

// 运行绑定：marker 是自我声明，不能自己证明自己。这里独立读活跃进程命令行，并要求
// --url 指向该 fixture 声明的端口，避免把别的服务或旧 fixture 纳入写入型实验。
assert.ok(Number.isSafeInteger(fixture.pid) && fixture.pid > 0, "隔离标记缺少 pid");
assert.ok(Number.isSafeInteger(fixture.port) && fixture.port > 0, "隔离标记缺少 port");
let processes;
let fixtureCwd;
try {
  processes = readFileSync(`/proc/${fixture.pid}/cmdline`, "utf8").split("\0").filter(Boolean);
  fixtureCwd = realpathSync(`/proc/${fixture.pid}/cwd`);
} catch {
  throw new Error(`一次性 fixture 进程不存在或不可读: pid ${fixture.pid}`);
}
const stateReal = realpathSync(fixture.state);
// 正常启动形式包括 `--state /abs`、`--state=/abs` 和相对路径。相对值必须按被检查
// 进程自己的 cwd 解析，不能按本脚本的 cwd，否则会把正常命令误拒。
const ownsState = processes.some(token => {
  const value = token.startsWith("--state=") ? token.slice("--state=".length) : token;
  if (!value || value.startsWith("-")) return false;
  try {
    const candidate = resolve(fixtureCwd, value);
    return existsSync(candidate) && realpathSync(candidate) === stateReal;
  } catch {
    return false;
  }
});
assert.ok(ownsState, `活跃进程 ${fixture.pid} 的命令行未引用该 fixture state`);
const origin = `http://127.0.0.1:${fixture.port}`;
assert.ok(values.url === origin || values.url.startsWith(`${origin}/`),
  `--url 必须指向本次 fixture 的地址 ${origin}`);

// 端口归属：必须按被测地址的族与本地地址核对。只按端口汇总 pid 无法区分
// 127.0.0.1 / ::1 / 0.0.0.0，也就无法证明"这个 URL 由该 fixture 提供"。
function listenerOwners(host, port) {
  let table;
  try {
    table = execFileSync("ss", ["-ltnpH", `sport = :${port}`], { encoding: "utf8" });
  } catch {
    throw new Error("无法读取监听表（需要 ss）");
  }
  const ipv6 = host.includes(":");
  const wanted = ipv6 ? `[${host}]` : host;
  const wildcard = ipv6 ? "[::]" : "0.0.0.0";
  const pids = new Set();
  let unknown = false;
  for (const line of table.split("\n")) {
    const columns = line.trim().split(/\s+/);
    if (columns.length < 5) continue;
    const local = columns[3];
    const separator = local.lastIndexOf(":");
    if (separator <= 0) continue;
    if (Number(local.slice(separator + 1)) !== port) continue;
    const address = local.slice(0, separator);
    if (address !== wanted && address !== wildcard) continue;
    const matches = [...line.matchAll(/pid=(\d+)/g)].map(match => Number(match[1]));
    if (!matches.length) unknown = true;   // 该端点有人监听，但拿不到 pid
    for (const pid of matches) pids.add(pid);
  }
  return { pids, unknown };
}
const host = new URL(origin).hostname;
function assertListenerOwned(stage) {
  const { pids, unknown } = listenerOwners(host, fixture.port);
  const owners = [...pids];
  assert.ok(!unknown, `${stage}：${host}:${fixture.port} 存在无法识别归属的监听者`);
  assert.ok(owners.length === 1 && owners[0] === fixture.pid,
    `${stage}：${host}:${fixture.port} 的监听者必须唯一且是该 fixture（实测 ${owners.join(",") || "无"}，声明 ${fixture.pid}）`);
}
assertListenerOwned("写入前");

// 就绪门：启动未完成就写入资产会把"还没起来"记成性能结果。就绪必须由真实健康接口确认
// ——/api/chat/health 只有在消息目录能打开时才返回 {"status":"ready"}；只看"有响应"会把
// 404/403 甚至别的服务当成就绪。禁用手动跳转，避免被 3xx 引到别处。
const health = `${origin}/api/chat/health`;
let ready = false;
const deadline = Date.now() + 30000;
while (Date.now() < deadline) {
  const remaining = deadline - Date.now();
  if (remaining <= 0) break;
  try {
    const response = await fetch(health, {
      redirect: "manual",
      signal: AbortSignal.timeout(Math.min(2000, remaining)),
    });
    if (response.status === 200) {
      const payload = await response.json();
      if (Date.now() < deadline && payload?.status === "ready") { ready = true; break; }
    } else {
      await response.body?.cancel();
    }
  } catch {
    // 尚未监听、单次超时或响应不是 JSON，继续等。
  }
  const pause = deadline - Date.now();
  if (pause <= 0) break;
  await new Promise(resolve_ => setTimeout(resolve_, Math.min(500, pause)));
}
assert.ok(ready, `fixture 未在 30 秒内就绪（${health} 未返回 status=ready）：${origin}`);
// 等待期间监听关系可能变化，就绪后再核对一次归属。
assertListenerOwned("就绪后");
const profiles = values.profiles.split(","), runs = Number(values.runs);
assert.ok(Number.isSafeInteger(runs) && runs > 0);
assert.ok(profiles.every(name => ["lan", "latency", "limited"].includes(name)));
// 创建前先校验预期位置：被拒绝时不得在目标或被测目录留下任何副作用。
assert.ok(!existsSync(output), `output 必须是不存在的新目录: ${output}`);
assert.ok(!isInside(output, target) && !isInside(target, output),
  "output 目录不得与 runtime-chat 互相嵌套");
assert.ok(Object.values(builds).every(path => !isInside(output, path) && !isInside(path, output)),
  "output 目录不得与被测构建目录互相嵌套");
mkdirSync(output, { recursive: false });
const recovery = resolve(output, "starting-chat");
cpSync(target, recovery, { recursive: true });
const report = { plannedSamples: runs * profiles.length * 4, complete: false, builds, target,
  affinity: process.platform === "linux" ? readFileSync("/proc/self/status", "utf8").match(/^Cpus_allowed_list:.*$/mu)?.[0] : null,
  pairs: [], summary: {} };
const save = () => writeFileSync(resolve(output, "report.json"), JSON.stringify(report, null, 2));
const reports = new Map(), artifacts = {}, rows = [];
let frozen;

function copyBuild(source) {
  // 每次替换资产前重新核对归属：等待、采样与上一轮之间监听关系都可能变化。
  assertListenerOwned(`替换前 ${source}`);
  execFileSync("rsync", ["-a", "--delete", `${source}/`, `${target}/`]);
  function normalize(directory) {
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      const path = resolve(directory, entry.name);
      if (entry.isDirectory()) normalize(path);
      else utimesSync(path, 1767225600, 1767225600);
    }
  }
  normalize(target);
}

try {
  // 1. 每轮旋转网络顺序，A/B 顺序交替；不在采样期间构建或换代 Runtime。
  for (let run = 0; run < runs; run++) {
    for (let offset = 0; offset < profiles.length; offset++) {
      const profile = profiles[(run + offset) % profiles.length];
      const order = (run + offset) % 2 ? ["candidate", "baseline"] : ["baseline", "candidate"];
      const pair = { run, profile, order, reports: {} };
      report.pairs.push(pair);
      for (const kind of order) {
        copyBuild(builds[kind]);
        const directory = resolve(output, `${run}-${profile}-${kind}`);
        const beforeLoad = loadavg();
        const child = spawnSync(process.execPath, ["scripts/webui-performance/measure-startup.mjs",
          "--url", values.url, "--browser", values.browser, "--output", directory,
          "--runs", "1", "--profiles", profile, "--label", kind], { encoding: "utf8" });
        writeFileSync(`${directory}.log`, `${child.stdout ?? ""}${child.stderr ?? ""}`);
        if (child.error) throw child.error;
        const result = JSON.parse(readFileSync(resolve(directory, "report.json"), "utf8"));
        assert.ok(result.summary, `采样未完成：${directory}`);
        const inputs = [result.harnessSha256, result.lockSha256, result.bootstrapSha256,
          result.entrySha256, result.browserVersion];
        if (!frozen) frozen = inputs;
        else assert.deepEqual(inputs, frozen, "基准输入在采样期间变化");
        if (!artifacts[kind]) artifacts[kind] = result.chatEntrySha256;
        else assert.equal(result.chatEntrySha256, artifacts[kind], "聊天制品在采样期间变化");
        pair.reports[kind] = { directory, beforeLoad, afterLoad: loadavg(), exitCode: child.status };
        reports.set(`${run}/${profile}/${kind}`, result);
        rows.push(...result.runs);
        console.log(JSON.stringify({ run, profile, kind, results: result.runs.map(row => ({
          cache: row.cache, readyMs: row.readyMs, sockets: row.sockets, success: row.success,
        })) }));
        save();
      }
    }
  }
  // 2. 所有计划样本进入分母；同时报告整体中位数和每对变化的中位数。
  const median = nums => [...nums].sort((a, b) => a - b)[Math.floor(nums.length / 2)];
  for (const profile of profiles) for (const cache of ["cold", "warm"]) {
    const pairs = Array.from({ length: runs }, (_, run) => Object.fromEntries(["baseline", "candidate"].map(kind =>
      [kind, reports.get(`${run}/${profile}/${kind}`).runs.find(row => row.cache === cache)])));
    const passed = pairs.every(pair => pair.baseline.success && pair.candidate.success);
    report.summary[`${profile}/${cache}`] = {
      baselineMs: passed ? median(pairs.map(pair => pair.baseline.readyMs)) : null,
      candidateMs: passed ? median(pairs.map(pair => pair.candidate.readyMs)) : null,
      pairedChangeMs: passed ? median(pairs.map(pair => pair.candidate.readyMs - pair.baseline.readyMs)) : null,
      pairedChangePct: passed ? median(pairs.map(pair => (pair.candidate.readyMs / pair.baseline.readyMs - 1) * 100)) : null,
      baselineSockets: median(pairs.map(pair => pair.baseline.sockets)),
      candidateSockets: median(pairs.map(pair => pair.candidate.sockets)),
      successes: pairs.filter(pair => pair.baseline.success && pair.candidate.success).length,
      pairs: runs,
    };
  }
  report.complete = true;
  report.artifacts = artifacts;
  report.frozenInputs = frozen;
  save();
  console.log(JSON.stringify(report.summary, null, 2));
  if (rows.some(row => !row.success)) process.exitCode = 1;
} finally {
  // 不覆盖正式目录；仅恢复本次隔离服务进入实验前的构建。
  execFileSync("rsync", ["-a", "--delete", `${recovery}/`, `${target}/`]);
}
