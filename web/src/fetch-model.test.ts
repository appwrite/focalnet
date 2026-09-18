import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, expect, test } from "bun:test";
import { DEFAULT_ONNX_SHA256, fetchModel, resolvedHash } from "../scripts/fetch-model";

const dirs: string[] = [];
const originalFetch = globalThis.fetch;

afterEach(async () => {
  globalThis.fetch = originalFetch;
  await Promise.all(dirs.splice(0).map((dir) => rm(dir, { recursive: true, force: true })));
});

async function tempDest(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), "focalnet-fetch-"));
  dirs.push(dir);
  return join(dir, "focalnet-human.onnx");
}

async function digest(payload: string | Uint8Array): Promise<string> {
  const hasher = new Bun.CryptoHasher("sha256");
  hasher.update(payload);
  return hasher.digest("hex");
}

test("empty SHA256 still verifies against the published default", () => {
  expect(resolvedHash(undefined)).toBe(DEFAULT_ONNX_SHA256);
  expect(resolvedHash("")).toBe(DEFAULT_ONNX_SHA256);
  expect(resolvedHash("   ")).toBe(DEFAULT_ONNX_SHA256);
  const override = "a".repeat(64);
  expect(resolvedHash(` ${override} `)).toBe(override);
});

test("installs a local checkpoint whose hash matches", async () => {
  const dest = await tempDest();
  const source = join(dest, "..", "source.onnx");
  const payload = "matching-checkpoint";
  await writeFile(source, payload);
  const expected = await digest(payload);
  const actual = await fetchModel(
    { FOCALNET_ONNX: source, FOCALNET_ONNX_SHA256: expected },
    dest,
  );
  expect(actual).toBe(expected);
  expect(await Bun.file(dest).text()).toBe(payload);
});

test("rejects a local checkpoint when SHA256 is empty and the default does not match", async () => {
  const dest = await tempDest();
  const source = join(dest, "..", "source.onnx");
  await writeFile(source, "not-the-published-model");
  await expect(fetchModel({ FOCALNET_ONNX: source, FOCALNET_ONNX_SHA256: "" }, dest)).rejects.toThrow(
    /does not match/,
  );
  expect(await Bun.file(dest).exists()).toBe(false);
});

test("downloads a matching checkpoint and rejects a mismatched one", async () => {
  const dest = await tempDest();
  const payload = "downloaded-checkpoint";
  const expected = await digest(payload);
  globalThis.fetch = async () => new Response(payload, { status: 200 });
  const actual = await fetchModel(
    {
      FOCALNET_ONNX_URL: "https://example.test/focalnet-human.onnx",
      FOCALNET_ONNX_SHA256: expected,
    },
    dest,
  );
  expect(actual).toBe(expected);
  expect(await Bun.file(dest).text()).toBe(payload);

  const other = await tempDest();
  globalThis.fetch = async () => new Response("tampered-checkpoint", { status: 200 });
  await expect(
    fetchModel(
      {
        FOCALNET_ONNX_URL: "https://example.test/focalnet-human.onnx",
        FOCALNET_ONNX_SHA256: expected,
      },
      other,
    ),
  ).rejects.toThrow(/does not match/);
  expect(await Bun.file(other).exists()).toBe(false);
});

test("reuses an existing matching file and keeps a dummy when allowed", async () => {
  const dest = await tempDest();
  const payload = "already-fetched";
  await writeFile(dest, payload);
  const expected = await digest(payload);
  globalThis.fetch = async () => {
    throw new Error("should not download when the existing hash matches");
  };
  expect(
    await fetchModel(
      {
        FOCALNET_ONNX_URL: "https://example.test/focalnet-human.onnx",
        FOCALNET_ONNX_SHA256: expected,
      },
      dest,
    ),
  ).toBe(expected);

  const dummy = await tempDest();
  await writeFile(dummy, "ui-placeholder");
  expect(
    await fetchModel(
      {
        FOCALNET_ALLOW_DUMMY: "1",
        FOCALNET_ONNX_SHA256: expected,
      },
      dummy,
    ),
  ).not.toBe(expected);
  expect(await Bun.file(dummy).text()).toBe("ui-placeholder");
});
