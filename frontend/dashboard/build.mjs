import { build } from "esbuild";

await build({
  entryPoints: ["frontend/dashboard/src/main.tsx"],
  outfile: "static/dashboard/app.js",
  bundle: true,
  format: "iife",
  platform: "browser",
  target: "es2021",
  define: {
    "process.env.NODE_ENV": '"production"',
  },
  minify: true,
  sourcemap: false,
  jsx: "automatic",
  loader: {
    ".css": "css",
  },
});
