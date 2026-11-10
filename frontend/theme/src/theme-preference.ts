/** 把已退役的主题偏好映射到当前配色，保留亮暗与跟随系统模式。 */
export function migrateThemePreference(id: string): string {
  switch (id) {
    case "warm-paper":
    case "tuned-light":
      return "light";
    case "tuned-dark":
      return "dark";
    case "system-tuned":
      return "system-paper";
    default:
      return id;
  }
}
