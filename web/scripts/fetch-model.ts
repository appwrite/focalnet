#!/usr/bin/env bun
import { mkdir, unlink } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

const root = join(import.meta.dir, "..", "..");
const dest = join(root, "web", "public", "focalnet-human.onnx");
const expected = Bun.env.FOCALNET_ONNX_SHA256;
const modelUrl = Bun.env.FOCALNET_ONNX_URL;
const allowDummy = Bun.env.FOCALNET_ALLOW_DUMMY === "1";

await mkdir(join(dest, ".."), { recursive: true });

async function sha256(path: string): Promise<string> {
  const hasher = new Bun.CryptoHasher("sha256");
  hasher.update(await Bun.file(path).arrayBuffer());
  return hasher.digest("hex");
}

async function installModel(source: string): Promise<void> {
  await Bun.write(dest, Bun.file(source));
  const actual = await sha256(dest);
  if (expected && actual !== expected) {
    await unlink(dest);
    throw new Error(`Model hash ${actual} does not match ${expected}`);
  }
  console.log(`Wrote ${dest} (${actual})`);
}

if (await Bun.file(dest).exists()) {
  const actual = await sha256(dest);
  if (!expected || actual === expected || allowDummy) {
    console.log(`Using existing model at ${dest} (${actual})`);
    process.exit(0);
  }
  console.warn(`Existing model hash ${actual} does not match ${expected}; replacing`);
  await unlink(dest);
}

if (Bun.env.FOCALNET_ONNX) {
  await installModel(Bun.env.FOCALNET_ONNX);
  process.exit(0);
}

if (modelUrl) {
  const tmp = join(tmpdir(), "focalnet-human.onnx.download");
  let installed = false;
  try {
    const response = await fetch(modelUrl);
    if (!response.ok) {
      throw new Error(`GET ${modelUrl} → ${response.status}`);
    }
    await Bun.write(tmp, response);
    await installModel(tmp);
    installed = true;
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    if (!allowDummy) {
      throw new Error(detail);
    }
    console.warn(detail);
  }
  if (installed) {
    await unlink(tmp).catch((error) => {
      console.warn(error instanceof Error ? error.message : error);
    });
    process.exit(0);
  }
}

if (!allowDummy) {
  throw new Error(
    "No local checkpoint. Export a model, then set FOCALNET_ONNX to that file " +
      "(optional FOCALNET_ONNX_SHA256 to verify). FOCALNET_ALLOW_DUMMY=1 writes a UI placeholder.",
  );
}

const dummy = Bun.spawn(
  ["uv", "run", "--with", "onnx", "python", join(root, "web/scripts/export-dummy-onnx.py")],
  { cwd: root, stdout: "inherit", stderr: "inherit" },
);
if ((await dummy.exited) !== 0) {
  throw new Error("Dummy ONNX export failed");
}
console.log("Wrote dummy ONNX for local UI testing; do not publish this file");
