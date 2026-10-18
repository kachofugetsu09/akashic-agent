import type { ScopeFieldControls } from "@akashic/chat-ui-v1";

type Commit = (value: string) => Promise<void>;

/** 一个新建表单的附加设置：scope.fields 每个挂载各登记自己的提交，卸载即撤回。 */
export interface ScopeFields {
  controls(): { controls: ScopeFieldControls; release: () => void };
  /** 新取值确定后、创建之前按登记顺序依次提交；任一拒绝即整体失败。 */
  commit(value: string): Promise<void>;
}

// 每个表单实例一份；busy 由表单读出自己的提交状态。
export function createScopeFields(busy: () => boolean): ScopeFields {
  const commits = new Set<Commit>();
  return {
    controls() {
      const own = new Set<Commit>();
      let live = true;
      return {
        controls: {
          busy,
          onCommit(commit) {
            if (typeof commit !== "function") throw new Error("提交处理必须是函数");
            if (!live) return () => {};
            commits.add(commit);
            own.add(commit);
            return () => { commits.delete(commit); own.delete(commit); };
          },
        },
        release() {
          live = false;
          for (const commit of own) commits.delete(commit);
          own.clear();
        },
      };
    },
    async commit(value) {
      for (const commit of [...commits]) await commit(value);
    },
  };
}
