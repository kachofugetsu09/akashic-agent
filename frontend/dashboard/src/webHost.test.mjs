import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { createRequire } from "node:module";
import test from "node:test";
import { build } from "esbuild";
import { JSDOM } from "jsdom";

const compiled = await build({
  entryPoints: [new URL("./webHost.ts", import.meta.url).pathname],
  bundle: true, write: false, platform: "node", format: "cjs", packages: "external",
});
const module = { exports: {} };
new Function("require", "module", "exports", compiled.outputFiles[0].text)(
  createRequire(import.meta.url), module, module.exports,
);
const { startWebHost } = module.exports;
const digest = value => createHash("sha256").update(value).digest("hex");

/** 构造有真实资源摘要的插件；仍由生产 Host 导入、校验和清理。 */
function plugin(pluginId, source, requires, provides = []) {
  const contract = { contractDigests: {}, provides, requires };
  return { pluginId, generationId: `${pluginId}:1`, module: source,
    moduleSha256: digest(source), moduleBytes: Buffer.byteLength(source),
    stylesheet: "", stylesheetSha256: null, stylesheetBytes: 0,
    ...contract, contractSha256: digest(JSON.stringify(contract)) };
}

function parent(cardinality = "list") {
  return plugin("parent", `export function activate(ctx) {
    return ctx.ui.inject("web.root.v1", mount => mount.register({id:"shell",
      children:[{id:"fixture.panels.v1",cardinality:"${cardinality}"}],
      render(host, view) {
        host.textContent = "Working shell";
        const child = view.child("fixture.panels.v1");
        for (const entry of child.entries) {
          const target = document.createElement("section"); host.append(target);
          child.render(entry.id, target);
        }
        return () => { window.closed.push("shell"); };
      }
    }));
  }`, ["web.root.v1"], ["fixture.panels.v1"]);
}

function panels(ids, requires = ["fixture.panels.v1"]) {
  return plugin("panels", `export function activate(ctx) {
    const effects = ${JSON.stringify(ids)}.map(id => ctx.ui.inject("fixture.panels.v1", mount =>
      mount.register({id,render(host) {host.textContent=id; return () => window.closed.push(id);}})));
    return () => { effects.reverse().forEach(dispose => dispose()); window.closed.push("module"); };
  }`, requires);
}

/** JSDOM 不执行 Blob URL；只将资源传输换成 Node 可导入的 data URL。 */
async function withHost(t, modules, run) {
  const dom = new JSDOM("<div id='root'></div>", {url:"http://localhost/"});
  dom.window.closed = [];
  const NativeBlob = globalThis.Blob;
  class SourceBlob extends NativeBlob {
    constructor(parts, options) { super(parts, options); this.source = parts.join(""); }
  }
  const globals = {window:dom.window, document:dom.window.document, Blob:SourceBlob};
  const previous = new Map(Object.keys(globals).map(key => [key,Object.getOwnPropertyDescriptor(globalThis,key)]));
  for (const [key,value] of Object.entries(globals)) Object.defineProperty(globalThis,key,{value,configurable:true});
  t.mock.method(URL,"createObjectURL",blob => `data:text/javascript,${encodeURIComponent(blob.source)}`);
  t.mock.method(URL,"revokeObjectURL",() => {});
  t.mock.method(globalThis,"fetch",async () => new Response(JSON.stringify({
    schemaVersion:1,snapshotId:"snapshot",catalogId:"a".repeat(64),modules,
  })));
  t.mock.method(console,"error",() => {});
  let session;
  try {
    session = await startWebHost(document.getElementById("root"));
    await run({session,host:document.getElementById("root"),dom});
  } finally {
    session?.close();
    t.mock.restoreAll();
    dom.window.close();
    for (const [key,descriptor] of previous) {
      if (descriptor) Object.defineProperty(globalThis,key,descriptor);
      else delete globalThis[key];
    }
  }
}

test("one declared list dependency retains both panels and disposes each once", async t => {
  await withHost(t,[panels(["recall","ledger"]),parent()],({session,host,dom}) => {
    assert.deepEqual([...host.querySelectorAll("section")].map(x=>x.textContent),["ledger","recall"]);
    assert.equal(document.querySelector('[role="alert"]'),null);
    session.close(); session.close();
    assert.equal(host.textContent,"");
    assert.deepEqual([...dom.window.closed].sort(),["ledger","module","recall","shell"]);
    assert.equal(document.querySelector('[data-akashic-entry]'),null);
  });
});

for (const [label,cardinality,ids,requires] of [
  ["duplicate entry","list",["same","same"],["fixture.panels.v1"]],
  ["single mount","single",["first","second"],["fixture.panels.v1"]],
  ["undeclared dependency","list",["panel"],[]],
  ["unused dependency","list",["panel"],["fixture.panels.v1","unused.panels.v1"]],
]) {
  test(`${label} stays rejected and visible without removing the healthy shell`, async t => {
    await withHost(t,[parent(cardinality),panels(ids,requires)],({session,host}) => {
      assert.equal(host.textContent,"Working shell");
      const alert = document.querySelector('[role="alert"]');
      assert.ok(alert,"module failure must be visible next to a working shell");
      assert.match(alert.textContent,/panels/);
      session.close();
      assert.equal(document.querySelector('[role="alert"]'),null);
    });
  });
}
