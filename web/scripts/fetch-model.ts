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

export type FetchModelEnv = {
  FOCALNET_ONNX_SHA256?: string;
  FOCALNET_ONNX_URL?: string;
  FOCALNET_ONNX?: string;
  FOCALNET_ALLOW_DUMMY?: string;
};

export function resolvedHash(value: string | undefined): string {
  const trimmed = value?.trim() ?? "";
  return trimmed || DEFAULT_ONNX_SHA256;
}

export async function fetchModel(
  env: FetchModelEnv = Bun.env,
  destPath = dest,
): Promise<string> {
  const expected = resolvedHash(env.FOCALNET_ONNX_SHA256);
  const modelUrl = env.FOCALNET_ONNX_URL?.trim() || DEFAULT_ONNX_URL;
  const allowDummy = env.FOCALNET_ALLOW_DUMMY === "1";

  await mkdir(join(destPath, ".."), { recursive: true });

  async function sha256(path: string): Promise<string> {
    const hasher = new Bun.CryptoHasher("sha256");
    hasher.update(await Bun.file(path).arrayBuffer());
    return hasher.digest("hex");
  }

  async function installModel(source: string): Promise<string> {
    await Bun.write(destPath, Bun.file(source));
    const actual = await sha256(destPath);
    if (actual !== expected) {
      await unlink(destPath);
      throw new Error(`Model hash ${actual} does not match ${expected}`);
    }
    console.log(`Wrote ${destPath} (${actual})`);
    return actual;
  }

  if (await Bun.file(destPath).exists()) {
    const actual = await sha256(destPath);
    if (actual === expected) {
      console.log(`Using existing model at ${destPath} (${actual})`);
      return actual;
    }
    if (allowDummy) {
      console.log(`Using existing model at ${destPath} (${actual})`);
      return actual;
    }
    console.warn(`Existing model hash ${actual} does not match ${expected}; replacing`);
    await unlink(destPath);
  }

  if (env.FOCALNET_ONNX) {
    return installModel(env.FOCALNET_ONNX);
  }

  if (modelUrl) {
    const tmp = join(tmpdir(), `focalnet-human-${crypto.randomUUID()}.onnx.download`);
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
    } finally {
      await unlink(tmp).catch(() => undefined);
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
  return sha256(destPath);
}

if (import.meta.main) {
  await fetchModel();
}
