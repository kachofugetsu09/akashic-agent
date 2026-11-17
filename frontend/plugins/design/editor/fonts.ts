import { fontManager, type Editor } from "@open-pencil/core";
import interBold from "@open-pencil/core/assets/Inter-Bold.ttf";
import interExtraBold from "@open-pencil/core/assets/Inter-ExtraBold.ttf";
import interMedium from "@open-pencil/core/assets/Inter-Medium.ttf";
import interRegular from "@open-pencil/core/assets/Inter-Regular.ttf";
import interSemiBold from "@open-pencil/core/assets/Inter-SemiBold.ttf";
import notoNaskhArabic from "@open-pencil/core/assets/NotoNaskhArabic-Regular.ttf";

const FONTS: Record<string, string> = {
  "Inter-Bold.ttf": interBold, "Inter-ExtraBold.ttf": interExtraBold, "Inter-Medium.ttf": interMedium,
  "Inter-Regular.ttf": interRegular, "Inter-SemiBold.ttf": interSemiBold, "NotoNaskhArabic-Regular.ttf": notoNaskhArabic,
};

// 画布与服务端无头画布用同一套字体：内置的只有 Inter，中日韩文字与 Inter 以外的字体从 Google Fonts 取（只开这一家），经本插件的后端代取并缓存。
export function configureFonts(request: (path: string) => Promise<Response>): void {
  fontManager.setBundledFontLocator((file) => FONTS[file.replace(/^\//u, "")] ?? file);
  fontManager.setOnlineFontProviders({ google: true, fontsource: false, bunny: false, fontshare: false });
  fontManager.setWebFontFetch((url) => request(`/api/dashboard/design/fonts?${new URLSearchParams({ url })}`));
  serializeRemoteFonts();
}

// 远程字体每次按“已取过的字 + 这次要的字”取一份子集当主文件。同一字体的请求并发时，后完成的那份不一定最全，
// 先到的字就只留在旧子集里画不出来；同一字体排队取，每份新子集都包含之前所有的字。
function serializeRemoteFonts(): void {
  const load = fontManager.loadRemoteFont.bind(fontManager);
  const queues = new Map<string, Promise<ArrayBuffer | null>>();
  fontManager.loadRemoteFont = (family, style = "Regular", characters = "", signal) => {
    const key = `${family}|${style}`;
    const previous = queues.get(key) ?? Promise.resolve(null);
    const next = previous.catch(() => null).then(() => load(family, style, characters, signal));
    queues.set(key, next);
    return next;
  };
}

// 每个渲染器上次重建字体提供器时各字体的主文件：主文件换了才需要再重建。
const providerFaces = new WeakMap<object, readonly (ArrayBuffer | null)[]>();

// 远程字体（含中日韩回退）按用到的字取子集，字变多时会换来一份更全的子集，旧子集仍留在同名字体下；
// CanvasKit 的字体提供器同名字体只认最早注册的那份，于是新出现的字在画框标签、正文里画成豆腐块。
// 字体管理器把提供器挂上时会先登记各字体的最新主文件，所以主文件变了就换一个新提供器，最全的子集排在最前。
export async function preferNewestFonts(editor: Editor): Promise<void> {
  const renderer = editor.renderer;
  if (!renderer) return;
  // 1. 回退字体与文档里用到的字体，取各自当前的主文件
  const families = [...fontManager.getCJKFallbackFamilies(), ...fontManager.getArabicFallbackFamilies()]
    .map((family): [string, string] => [family, "Regular"]);
  const pages = editor.graph.getPages().flatMap((page) => page.childIds);
  const faces = [...families, ...fontManager.collectFontKeys(editor.graph, pages)]
    .map(([family, style]) => fontManager.loadedData(family, style));
  // 2. 这个渲染器第一次见时也重建一次：之后字体补齐的回调才会经过这里
  const previous = providerFaces.get(renderer);
  if (previous && previous.length === faces.length && previous.every((face, index) => face === faces[index])) return;
  providerFaces.set(renderer, faces);
  // 3. 旧提供器先从字体管理器摘下，再由渲染器换上新的；之后按需补字形的回调同样走这里
  fontManager.detachProvider(renderer.getFontProvider());
  await renderer.loadFonts(() => {
    void preferNewestFonts(editor);
    editor.requestRender();
  });
  editor.requestRender();
}
