export async function requestSettingsJson<T>(url: string, init: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, {
      ...init,
      headers: { "Content-Type": "application/json", "X-Akasic-CSRF": "1", ...init.headers },
    });
  } catch (reason) {
    if (reason instanceof TypeError) throw new Error("无法连接 Akashic。请确认服务仍在运行，然后重试。", { cause: reason });
    throw reason;
  }
  const text = await response.text();
  let payload: { detail?: string; message?: string };
  try {
    payload = text ? JSON.parse(text) as { detail?: string; message?: string } : {};
  } catch {
    throw new Error(`设置服务返回了无效响应 (${response.status})`);
  }
  if (!response.ok) throw new Error(payload.detail || payload.message || `请求失败 (${response.status})`);
  return payload as T;
}

export function settingsErrorMessage(reason: unknown) {
  return reason instanceof Error ? reason.message : String(reason);
}
