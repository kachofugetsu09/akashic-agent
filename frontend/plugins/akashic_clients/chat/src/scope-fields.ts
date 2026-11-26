import type { ScopeFieldControls, ScopeFieldValues } from "@akashic/chat-ui-v1";

type Read = () => Record<string, string | number | boolean>;

/** 一个新建表单的附加设置：scope.fields 每个挂载按命名空间登记取值，卸载即撤回。 */
export interface ScopeFields {
  controls(): { controls: ScopeFieldControls; release: () => void };
  /** 提交时读出全部登记的取值，交给维度 owner 随创建一起写入。 */
  values(): ScopeFieldValues;
}

// 每个表单实例一份；busy 由表单读出自己的提交状态。
export function createScopeFields(busy: () => boolean): ScopeFields {
  const reads = new Map<string, Read>();
  return {
    controls() {
      const own = new Map<string, Read>();
      let live = true;
      return {
        controls: {
          busy,
          contribute(namespace, read) {
            if (typeof read !== "function") throw new Error("取值读取必须是函数");
            if (!/^[a-z][a-z0-9_]*$/.test(namespace)) throw new Error(`附加设置命名空间无效：${namespace}`);
            if (!live) return () => {};
            if (reads.has(namespace)) throw new Error(`附加设置命名空间已被占用：${namespace}`);
            reads.set(namespace, read);
            own.set(namespace, read);
            return () => {
              if (reads.get(namespace) === read) reads.delete(namespace);
              own.delete(namespace);
            };
          },
        },
        release() {
          live = false;
          for (const [namespace, read] of own) if (reads.get(namespace) === read) reads.delete(namespace);
          own.clear();
        },
      };
    },
    values() {
      const result: ScopeFieldValues = {};
      for (const [namespace, read] of reads) result[namespace] = { ...read() };
      return result;
    },
  };
}
