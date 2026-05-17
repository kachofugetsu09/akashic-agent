import { build, context } from "esbuild";

const watch = process.argv.includes("--watch");

const config = {
  entryPoints: ["frontend/dashboard/src/main.tsx"],
  outfile: "static/dashboard/app.js",
  bundle: true,
  format: "iife",
  platform: "browser",
  target: "es2021",
  define: {
    "process.env.NODE_ENV": watch ? '"development"' : '"production"',
  },
  minify: !watch,
  sourcemap: watch ? "inline" : false,
  jsx: "automatic",
  loader: {
    ".css": "css",
  },
};

if (watch) {
  const ctx = await context(config);
  await ctx.watch();
  console.log("watching frontend/dashboard/src/ ...");
} else {
  await build(config);
}
