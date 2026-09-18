import { expect, test } from "bun:test";
import {
  DEFAULT_ONNX_SHA256,
  DEFAULT_ONNX_URL,
  HUB_REPO,
} from "../scripts/fetch-model";

test("fetch-model defaults to the published Hub ranking checkpoint", () => {
  expect(HUB_REPO).toBe("appwrite/focalnet");
  expect(DEFAULT_ONNX_URL).toBe(
    "https://huggingface.co/appwrite/focalnet/resolve/main/focalnet-human.onnx",
  );
  expect(DEFAULT_ONNX_SHA256).toBe(
    "59164c601c98cea3f62b25166710831dac63e1a872fc64767c65316ad5385439",
  );
});
