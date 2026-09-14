#!/usr/bin/env bun
import { join } from "node:path";

const dist = join(import.meta.dir, "..", "dist");
const port = Number(Bun.env.PORT ?? 4173);
const index = join(dist, "index.html");

if (!(await Bun.file(index).exists())) {
  throw new Error("Missing dist/index.html; run `bun run build` first");
}

function mime(pathname: string): string | undefined {
  if (pathname.endsWith(".html")) {
    return "text/html; charset=utf-8";
  }
  if (pathname.endsWith(".js")) {
    return "text/javascript; charset=utf-8";
  }
  if (pathname.endsWith(".css")) {
    return "text/css; charset=utf-8";
  }
  if (pathname.endsWith(".wasm")) {
    return "application/wasm";
  }
  if (pathname.endsWith(".mjs")) {
    return "text/javascript; charset=utf-8";
  }
  if (pathname.endsWith(".onnx")) {
    return "application/octet-stream";
  }
  return undefined;
}

Bun.serve({
  port,
  hostname: "127.0.0.1",
  async fetch(request) {
    const url = new URL(request.url);
    let pathname = decodeURIComponent(url.pathname);
    if (pathname === "/" || pathname.endsWith("/")) {
      pathname = `${pathname}index.html`.replace("//", "/");
    }
    const file = Bun.file(join(dist, pathname));
    if (!(await file.exists())) {
      return new Response("Not found", { status: 404 });
    }
    const type = mime(pathname);
    return new Response(file, type ? { headers: { "content-type": type } } : undefined);
  },
});

console.log(`FocalNet crop lab → http://127.0.0.1:${port}/`);
