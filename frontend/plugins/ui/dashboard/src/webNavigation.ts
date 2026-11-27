import type {
  WebNavigationOutcome,
  WebNavigationReason,
  WebNavigationRequest,
  WebUiDisposer,
} from "@akashic/web-ui-v1";

type Guard = (request: WebNavigationRequest) => void;

// 宿主唯一拥有的导航约定：页面切换与目录换代先问守卫，安装/配置提交与设置关闭各是一个具名信号。
// 换代时新旧会话共用这一份登记，旧会话模块的守卫随模块注销一起撤掉。
export class WebNavigation {
  private readonly guards = new Set<Guard>();
  private readonly leaveListeners = new Set<(reason: WebNavigationReason) => void>();
  private readonly revealListeners = new Set<() => void>();
  private readonly catalogListeners = new Set<() => void>();

  guard(guard: Guard): WebUiDisposer {
    this.guards.add(guard);
    return () => { this.guards.delete(guard); };
  }

  // 依登记顺序问每个守卫；任一守卫拦下就不执行 go，答应稍后自己续上的记为 deferred。
  request(reason: WebNavigationReason, go: () => void): WebNavigationOutcome {
    let held = false;
    let resume = false;
    const request: WebNavigationRequest = {
      reason,
      go,
      get held() { return held; },
      hold: (options) => { held = true; if (options?.resume) resume = true; },
    };
    for (const guard of [...this.guards]) {
      try { guard(request); } catch (error) { console.error("[web-host] navigation guard failed", error); }
    }
    if (!held) {
      for (const listener of [...this.leaveListeners]) {
        try { listener(reason); } catch (error) { console.error("[web-host] leave listener failed", error); }
      }
      go();
      return "proceeded";
    }
    return resume ? "deferred" : "held";
  }

  onLeave(listener: (reason: WebNavigationReason) => void): WebUiDisposer {
    this.leaveListeners.add(listener);
    return () => { this.leaveListeners.delete(listener); };
  }

  expectCatalogChange(): void {
    for (const listener of [...this.catalogListeners]) listener();
  }

  onCatalogChangeExpected(listener: () => void): WebUiDisposer {
    this.catalogListeners.add(listener);
    return () => { this.catalogListeners.delete(listener); };
  }

  reveal(): void {
    for (const listener of [...this.revealListeners]) {
      try { listener(); } catch (error) { console.error("[web-host] reveal listener failed", error); }
    }
  }

  onReveal(listener: () => void): WebUiDisposer {
    this.revealListeners.add(listener);
    return () => { this.revealListeners.delete(listener); };
  }
}
