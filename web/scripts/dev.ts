#!/usr/bin/env bun
import homepage from "../index.html";

const root = `${import.meta.dir}/..`;
const port = Number(Bun.env.PORT ?? 5173);
const onnx = `${root}/public/focalnet-human.onnx`;
const wasm = `${root}/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm`;

function file(path: string, type: string): Response {
  return new Response(Bun.file(path), { headers: { "content-type": type } });
}

Bun.serve({
  port,
  hostname: "127.0.0.1",
  development: {
    hmr: true,
    console: true,
  },
  routes: {
    "/": homepage,
    "/focalnet-human.onnx": () => file(onnx, "application/octet-stream"),
    "/ort-wasm-simd-threaded.wasm": () => file(wasm, "application/wasm"),
  },
});

console.log(`FocalNet crop lab → http://127.0.0.1:${port}/`);
