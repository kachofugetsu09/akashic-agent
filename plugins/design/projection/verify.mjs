// 验证投影链。先按 README 构建 project.mjs，再用可选路径参数指定它。
//   node plugins/design/projection/verify.mjs [bundlePath]
const bundlePath = process.argv[2] ?? './dist/project.mjs';
const { project, SceneGraph } = await import(bundlePath);
const g = new SceneGraph();
const page = g.addPage();
const SOLID = (r, gg, b) => [{ type: 'SOLID', color: { r, g: gg, b, a: 1 }, visible: true, opacity: 1 }];
const frame = g.createNode('FRAME', page.id, { name: 'Card', x: 0, y: 0, width: 320, height: 200, fills: SOLID(1, 1, 1), cornerRadius: 12 });
g.createNode('TEXT', frame.id, { name: 'Title', x: 24, y: 24, width: 240, height: 32, text: 'Paper 风格原型', fontSize: 24, fills: SOLID(0.1, 0.1, 0.1) });
const html = await project(g);
const checks = {
  '非空': html.length > 0,
  '是完整文档': html.startsWith('<!doctype html>'),
  '含文本': html.includes('Paper 风格原型'),
  '含圆角': html.includes('border-radius: 12px'),
};
for (const [name, ok] of Object.entries(checks)) console.log(`${ok ? 'OK  ' : 'FAIL'} ${name}`);
console.log('HTML 长度:', html.length);
if (Object.values(checks).some(v => !v)) process.exitCode = 1;
