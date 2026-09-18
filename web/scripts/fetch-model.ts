#!/usr/bin/env bun
import { mkdir, unlink } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

export const HUB_REPO = "appwrite/focalnet";
export const DEFAULT_ONNX_URL =
  "https://huggingface.co/appwrite/focalnet/resolve/main/focalnet-human.onnx";
export const DEFAULT_ONNX_SHA256 =
  "59164c601c98cea3f62b25166710831dac63e1a872fc64767c65316ad5385439";

const root = join(import.meta.dir, "..", "..");
const dest = join(root, "web", "public", "focalnet-human.onnx");

export async function fetchModel(env: typeof Bun.env = Bun.env): Promise<string> {
  const expected = env.FOCALNET_ONNX_SHA256 ?? DEFAULT_ONNX_SHA256;
  const modelUrl = env.FOCALNET_ONNX_URL ?? DEFAULT_ONNX_URL;
  const allowDummy = env.FOCALNET_ALLOW_DUMMY === "1";

  await mkdir(join(dest, ".."), { recursive: true });

  async function sha256(path: string): Promise<string> {
    const hasher = new Bun.CryptoHasher("sha256");
    hasher.update(await Bun.file(path).arrayBuffer());
    return hasher.digest("hex");
  }

  async function installModel(source: string): Promise<string> {
    await Bun.write(dest, Bun.file(source));
    const actual = await sha256(dest);
    if (expected && actual !== expected) {
      await unlink(dest);
      throw new Error(`Model hash ${actual} does not match ${expected}`);
    }
    console.log(`Wrote ${dest} (${actual})`);
    return actual;
  }

  if (await Bun.file(dest).exists()) {
    const actual = await sha256(dest);
    if (!expected || actual === expected || allowDummy) {
      console.log(`Using existing model at ${dest} (${actual})`);
      return actual;
    }
    console.warn(`Existing model hash ${actual} does not match ${expected}; replacing`);
    await unlink(dest);
  }

  if (env.FOCALNET_ONNX) {
    return installModel(env.FOCALNET_ONNX);
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
      const actual = await installModel(tmp);
      installed = true;
      await unlink(tmp).catch((error) => {
        console.warn(error instanceof Error ? error.message : error);
      });
      return actual;
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      if (installed) {
        throw new Error(detail);
      }
      if (!allowDummy) {
        throw new Error(detail);
      }
      console.warn(detail);
    }
  }

  if (!allowDummy) {
    throw new Error(
      "No local checkpoint. bun run fetch-model downloads the published Hub " +
        "weights, or set FOCALNET_ONNX to a local file. FOCALNET_ALLOW_DUMMY=1 " +
        "writes a UI placeholder.",
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
  return sha256(dest);
}

if (import.meta.main) {
  await fetchModel();
}
