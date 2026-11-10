/**
 * 统一的复制入口。Clipboard API 只在 secure context 可用，局域网 http 部署
 * （如 http://192.168.x.x）下同iframe 里 navigator.clipboard 直接是 undefined，
 * 同步访问会抛 TypeError；这里回退到隐藏 textarea + execCommand，保持用户手势内可用。
 */
export async function copyText(text: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return;
    } catch {
      // 权限被拒或焦点不在文档时继续走回退，而不是直接失败。
    }
  }
  const area = document.createElement("textarea");
  area.value = text;
  area.setAttribute("readonly", "");
  area.style.cssText = "position:fixed;top:0;inset-inline-start:-9999px;opacity:0";
  document.body.appendChild(area);
  area.select();
  try {
    if (!document.execCommand("copy")) throw new Error("浏览器拒绝了复制请求");
  } finally {
    area.remove();
  }
}
