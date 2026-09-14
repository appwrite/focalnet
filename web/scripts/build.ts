#!/usr/bin/env bun
import { mkdir, rm } from "node:fs/promises";
import { join } from "node:path";

const root = join(import.meta.dir, "..");
const dist = join(root, "dist");
const onnx = join(root, "public", "focalnet-human.onnx");
const wasm = join(
  root,
  "node_modules",
  "onnxruntime-web",
  "dist",
  "ort-wasm-simd-threaded.wasm",
);

await rm(dist, { recursive: true, force: true });
await mkdir(dist, { recursive: true });

const result = await Bun.build({
  entrypoints: [join(root, "index.html")],
  outdir: dist,
  minify: true,
  sourcemap: "none",
  publicPath: "./",
  naming: {
    entry: "[name].[ext]",
    chunk: "assets/[name]-[hash].[ext]",
    asset: "assets/[name]-[hash].[ext]",
  },
});

if (!result.success) {
  for (const log of result.logs) {
    console.error(log);
  }
  process.exit(1);
}

if (!(await Bun.file(wasm).exists())) {
  throw new Error("Missing onnxruntime-web WASM; run `bun install`");
}
await Bun.write(join(dist, "ort-wasm-simd-threaded.wasm"), Bun.file(wasm));

if (await Bun.file(onnx).exists()) {
  await Bun.write(join(dist, "focalnet-human.onnx"), Bun.file(onnx));
} else {
  console.warn("Missing public/focalnet-human.onnx; run `bun run fetch-model`");
}

console.log(`Wrote ${dist}`);
