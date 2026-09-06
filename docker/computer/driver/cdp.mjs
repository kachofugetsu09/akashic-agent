import { EventEmitter, once } from "node:events";
import { randomInt } from "node:crypto";

/** 每个 WebSocket 拥有自己的请求表；断连同时拒绝全部等待者。 */
class CdpConnection extends EventEmitter {
  pending = new Map();
  nextId = 1;
  async open(url) {
    this.socket = new WebSocket(url);
    this.socket.addEventListener("message", ({ data }) => {
      const message = JSON.parse(data);
      if (message.id != null) {
        const pending = this.pending.get(message.id);
        if (!pending) return;
        this.pending.delete(message.id);
        clearTimeout(pending.timer);
        if (message.error) pending.reject(new Error(message.error.message));
        else pending.resolve(message.result);
      } else this.emit("event", message);
    });
    this.socket.addEventListener("close", () => {
      for (const pending of this.pending.values()) {
        clearTimeout(pending.timer);
        pending.reject(
          new Error("Chromium connection closed; earlier effects may remain"),
        );
      }
      this.pending.clear();
      this.emit("close");
    });
    await Promise.race([
      once(this.socket, "open"),
      new Promise((_, reject) => {
        const timer = setTimeout(() => {
          this.socket.close();
          reject(new Error("Chromium connection timed out"));
        }, 5000);
        this.socket.addEventListener("open", () => clearTimeout(timer), {
          once: true,
        });
        this.socket.addEventListener(
          "error",
          () => {
            clearTimeout(timer);
            reject(new Error("Chromium connection failed"));
          },
          { once: true },
        );
      }),
    ]);
    return this;
  }
  send(method, params = {}, sessionId) {
    if (this.socket.readyState !== WebSocket.OPEN)
      return Promise.reject(new Error("Chromium connection is closed"));
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`CDP timeout: ${method}; effects may remain`));
      }, 10_000);
      this.pending.set(id, { resolve, reject, timer });
      this.socket.send(
        JSON.stringify({
          id,
          method,
          params,
          ...(sessionId ? { sessionId } : {}),
        }),
      );
    });
  }
  close() {
    this.socket.close();
  }
}

/** 将原 Browser service 的后端协议绑定到容器内唯一 Chromium。 */
export class BrowserBackend extends EventEmitter {
  tabs = new Map();
  connections = new Map();
  connecting = new Map();
  nextTab = randomInt(1, 2 ** 40);
  inputs = new Map();
  expressions = new Map();
  sessionNames = new Map();
  async start() {
    const response = await fetch("http://127.0.0.1:9222/json/version", {
      signal: AbortSignal.timeout(5000),
    });
    if (!response.ok)
      throw new Error(`Chromium discovery returned ${response.status}`);
    const { webSocketDebuggerUrl } = await response.json();
    this.browser = await new CdpConnection().open(webSocketDebuggerUrl);
    this.browser.on("close", () => this.emit("disconnected"));
  }
  async listTabs() {
    const { targetInfos } = await this.browser.send("Target.getTargets");
    const live = new Set();
    const results = [];
    for (const target of targetInfos.filter((item) => item.type === "page")) {
      live.add(target.targetId);
      let tab = this.tabs.get(target.targetId);
      if (!tab) {
        tab = {
          id: this.nextTab++,
          targetId: target.targetId,
          created: null,
          claimed: null,
          status: null,
        };
        this.tabs.set(target.targetId, tab);
      }
      Object.assign(tab, { url: target.url, title: target.title });
      const { windowId } = await this.browser.send(
        "Browser.getWindowForTarget",
        { targetId: target.targetId },
      );
      results.push({
        id: tab.id,
        title: tab.title,
        url: tab.url,
        windowId,
      });
    }
    for (const [id, tab] of this.tabs)
      if (!live.has(id)) {
        this.connections.get(tab.id)?.close();
        this.connections.delete(tab.id);
        this.tabs.delete(id);
      }
    return results;
  }
  tab(id) {
    const tab = [...this.tabs.values()].find((tab) => tab.id === id);
    if (!tab)
      throw new Error(`Unknown or stale browser tab: ${id}; list tabs again`);
    return tab;
  }
  async attach(id) {
    if (this.connections.has(id)) return this.connections.get(id);
    if (!this.connecting.has(id))
      this.connecting.set(
        id,
        this.openTab(id).finally(() => this.connecting.delete(id)),
      );
    return this.connecting.get(id);
  }
  async openTab(id) {
    if (this.connections.has(id)) return this.connections.get(id);
    const tab = this.tab(id);
    const connection = await new CdpConnection().open(
      `ws://127.0.0.1:9222/devtools/page/${tab.targetId}`,
    );
    this.connections.set(id, connection);
    connection.on("event", (event) =>
      this.emit("event", {
        source: {
          tabId: id,
          ...(event.sessionId ? { sessionId: event.sessionId } : {}),
        },
        method: event.method,
        params: event.params ?? {},
      }),
    );
    connection.on("close", () => {
      if (this.connections.get(id) === connection) this.connections.delete(id);
    });
    return connection;
  }
  /** 记录已送出的输入，取消后用相同 CDP target 发送 release。 */
  async execute(params, context) {
    const { target, method, commandParams = {} } = params;
    const connection = await this.attach(target.tabId);
    const key = `${target.tabId}:${target.sessionId ?? ""}`;
    let input = this.inputs.get(key);
    if (!input) {
      input = {
        target,
        buttons: new Map(),
        keys: new Map(),
        touch: false,
        emulatedTouch: null,
      };
      this.inputs.set(key, input);
    }
    if (method === "Input.dispatchMouseEvent") {
      if (commandParams.type === "mousePressed")
        input.buttons.set(commandParams.button, { ...commandParams });
    }
    if (method === "Input.dispatchKeyEvent") {
      const code =
        commandParams.code ??
        commandParams.key ??
        commandParams.windowsVirtualKeyCode;
      if (["keyDown", "rawKeyDown"].includes(commandParams.type))
        input.keys.set(code, { ...commandParams });
    }
    if (
      method === "Input.dispatchTouchEvent" &&
      ["touchStart", "touchMove"].includes(commandParams.type)
    )
      input.touch = true;
    if (method === "Input.emulateTouchFromMouseEvent") {
      if (commandParams.type === "mousePressed")
        input.emulatedTouch = { ...commandParams };
      if (commandParams.type === "mouseMoved" && input.emulatedTouch)
        Object.assign(input.emulatedTouch, {
          x: commandParams.x,
          y: commandParams.y,
        });
    }
    if (method === "Page.close") {
      const tab = this.tab(target.tabId);
      this.checkOwner(tab, context);
    }
    const result = await connection.send(
      method,
      commandParams,
      target.sessionId,
    );
    if (
      method === "Input.dispatchMouseEvent" &&
      commandParams.type === "mouseReleased"
    )
      input.buttons.delete(commandParams.button);
    if (method === "Input.dispatchKeyEvent" && commandParams.type === "keyUp")
      input.keys.delete(
        commandParams.code ??
          commandParams.key ??
          commandParams.windowsVirtualKeyCode,
      );
    if (
      method === "Input.dispatchTouchEvent" &&
      ["touchEnd", "touchCancel"].includes(commandParams.type)
    )
      input.touch = false;
    if (
      method === "Input.emulateTouchFromMouseEvent" &&
      commandParams.type === "mouseReleased"
    )
      input.emulatedTouch = null;
    return result;
  }
  checkOwner(tab, context) {
    const owner = tab.created ?? tab.claimed;
    if (!context || owner?.session_id !== context.session_id)
      throw new Error(
        "Claim this tab in the current Session before closing or marking it",
      );
  }
  async call(method, params = {}, context) {
    switch (method) {
      case "ping":
        return "pong";
      case "getInfo":
        return {
          type: "cdp",
          name: "Akashic Chromium",
          family: "chrome",
          capabilities: { browser: [], tab: [] },
          apiSupportOverrides: {
            "Browser.user": true,
            "Tab.markDeliverable": true,
            "Tab.markHandoff": true,
          },
        };
      case "getTabs":
      case "getUserTabs":
        return this.listTabs();
      case "attach":
        await this.attach(params.tabId);
        return;
      case "detach":
        this.connections.get(params.tabId)?.close();
        this.connections.delete(params.tabId);
        return;
      case "attachTarget": {
        const connection = await this.attach(params.tabId);
        return connection.send("Target.attachToTarget", {
          targetId: params.targetId,
          flatten: true,
        });
      }
      case "detachTarget":
        return (await this.attach(params.tabId)).send(
          "Target.detachFromTarget",
          { targetId: params.targetId },
        );
      case "executeCdp":
        return this.execute(params, context);
      case "executeCdpWithCachedExpression": {
        const { expressionCacheKey: key, commandParams } = params;
        if (commandParams.expression != null)
          this.expressions.set(key, commandParams.expression);
        if (!this.expressions.has(key)) return { kind: "cache_miss" };
        return {
          kind: "executed",
          result: await this.execute(
            {
              ...params,
              commandParams: {
                ...commandParams,
                expression: this.expressions.get(key),
              },
            },
            context,
          ),
        };
      }
      case "createTab": {
        const { targetId } = await this.browser.send("Target.createTarget", {
          url: "about:blank",
        });
        await this.listTabs();
        const tab = this.tabs.get(targetId);
        tab.created = {
          session_id: context.session_id,
          turn_id: context.turn_id,
        };
        return { id: tab.id, title: tab.title, url: tab.url };
      }
      case "claimUserTab": {
        await this.listTabs();
        const tab = this.tab(params.tabId);
        const owner = tab.created ?? tab.claimed;
        if (owner && owner.session_id !== context.session_id)
          throw new Error("Tab belongs to another live Session");
        tab.claimed = {
          session_id: context.session_id,
          turn_id: context.turn_id,
        };
        return { id: tab.id, title: tab.title, url: tab.url };
      }
      case "markTab":
        this.checkOwner(this.tab(params.tabId), context);
        this.tab(params.tabId).status = params.status;
        return;
      case "nameSession":
        this.sessionNames.set(context.session_id, params.name);
        return;
      case "moveMouse":
        return this.execute({
          target: { tabId: params.tabId },
          method: "Input.dispatchMouseEvent",
          commandParams: { type: "mouseMoved", x: params.x, y: params.y },
        });
      case "turnEnded":
        return this.endTurn(context);
      default:
        throw new Error(
          `Browser backend method is unavailable in this container: ${method}`,
        );
    }
  }
  async endTurn({ session_id, turn_id }) {
    for (const tab of this.tabs.values()) {
      const owned = tab.created ?? tab.claimed;
      if (owned?.session_id !== session_id || owned?.turn_id !== turn_id)
        continue;
      if (tab.created && !["deliverable", "handoff"].includes(tab.status)) {
        await this.browser.send("Target.closeTarget", {
          targetId: tab.targetId,
        });
      }
      tab.created = null;
      tab.claimed = null;
      tab.status = null;
    }
  }
  async releaseInputs() {
    const errors = [];
    for (const input of this.inputs.values()) {
      const connection = this.connections.get(input.target.tabId);
      if (!connection) {
        if (
          input.buttons.size ||
          input.keys.size ||
          input.touch ||
          input.emulatedTouch
        )
          errors.push(
            new Error("Lost CDP input owner; release cannot be confirmed"),
          );
        continue;
      }
      if (input.touch) {
        try {
          await connection.send(
            "Input.dispatchTouchEvent",
            { type: "touchCancel", touchPoints: [] },
            input.target.sessionId,
          );
          input.touch = false;
        } catch (error) {
          errors.push(error);
        }
      }
      if (input.emulatedTouch) {
        try {
          await connection.send(
            "Input.emulateTouchFromMouseEvent",
            { ...input.emulatedTouch, type: "mouseReleased" },
            input.target.sessionId,
          );
          input.emulatedTouch = null;
        } catch (error) {
          errors.push(error);
        }
      }
      for (const [button, params] of input.buttons) {
        try {
          await connection.send(
            "Input.dispatchMouseEvent",
            { ...params, type: "mouseReleased", buttons: 0 },
            input.target.sessionId,
          );
          input.buttons.delete(button);
        } catch (error) {
          errors.push(error);
        }
      }
      for (const [key, params] of input.keys) {
        try {
          await connection.send(
            "Input.dispatchKeyEvent",
            { ...params, type: "keyUp" },
            input.target.sessionId,
          );
          input.keys.delete(key);
        } catch (error) {
          errors.push(error);
        }
      }
    }
    for (const [key, input] of this.inputs)
      if (
        !input.buttons.size &&
        !input.keys.size &&
        !input.touch &&
        !input.emulatedTouch
      )
        this.inputs.delete(key);
    if (errors.length)
      throw new AggregateError(errors, "Browser input release failed");
  }
  close() {
    for (const connection of this.connections.values()) connection.close();
    this.connections.clear();
    this.browser.close();
  }
}
