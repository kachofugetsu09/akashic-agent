import { fontManager } from "@open-pencil/core";
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
}
