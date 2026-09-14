#!/usr/bin/env bun
import { mkdir, unlink } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

const root = join(import.meta.dir, "..", "..");
const dest = join(root, "web", "public", "focalnet-human.onnx");
const expected =
  Bun.env.FOCALNET_ONNX_SHA256 ??
  "59164c601c98cea3f62b25166710831dac63e1a872fc64767c65316ad5385439";
const modelUrl =
  Bun.env.FOCALNET_ONNX_URL ?? "https://knowledgeable-sun-ef9f.yeet.page/focalnet-human.onnx";
const volumePath = "/runs/repvit-m0-9-500k-cpc-gaic-lr3e4-candidate/focalnet-human.onnx";
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
  if (actual !== expected) {
    await unlink(dest);
    throw new Error(`Model hash ${actual} does not match ${expected}`);
  }
  console.log(`Wrote ${dest}`);
}

if (await Bun.file(dest).exists()) {
  const actual = await sha256(dest);
  if (actual === expected) {
    console.log(`Using existing model at ${dest}`);
    process.exit(0);
  }
  if (allowDummy) {
    console.log(`Using existing dummy model at ${dest}`);
    process.exit(0);
  }
  console.warn(`Existing model hash ${actual} does not match ${expected}; re-downloading`);
  await unlink(dest);
}

if (Bun.env.FOCALNET_ONNX) {
  await installModel(Bun.env.FOCALNET_ONNX);
  process.exit(0);
}

try {
  const response = await fetch(modelUrl);
  if (!response.ok) {
    throw new Error(`GET ${modelUrl} → ${response.status}`);
  }
  const tmp = join(tmpdir(), "focalnet-human.onnx.download");
  await Bun.write(tmp, response);
  await installModel(tmp);
  await unlink(tmp);
  process.exit(0);
} catch (error) {
  console.warn(error instanceof Error ? error.message : error);
}

if (!allowDummy) {
  const modal = Bun.which("uvx");
  if (modal) {
    const tmpdirPath = join(tmpdir(), "focalnet-modal-onnx");
    await mkdir(tmpdirPath, { recursive: true });
    const proc = Bun.spawn(
      ["uvx", "modal", "volume", "get", "focalnet-data", volumePath, tmpdirPath],
      { stdout: "inherit", stderr: "inherit" },
    );
    if ((await proc.exited) === 0) {
      const glob = new Bun.Glob("**/focalnet-human.onnx");
      for await (const path of glob.scan({ cwd: tmpdirPath, absolute: true })) {
        await installModel(path);
        process.exit(0);
      }
      throw new Error(`Modal volume get did not produce ${volumePath}`);
    }
  }
  throw new Error(
    `Could not fetch the production ONNX from ${modelUrl}. Set FOCALNET_ONNX or FOCALNET_ALLOW_DUMMY=1 for UI layout only.`,
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
