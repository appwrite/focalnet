import * as ort from "onnxruntime-web/wasm";
import {
  generateCandidates,
  letterboxContent,
  MAX_CANDIDATES,
  padBoxes,
} from "./candidates";
import { Crop, focalPoint, scoreCrop, solveCrop } from "./crop";
import {
  INPUT_SIZE,
  MAP_SIZE,
  type RgbaImage,
  prepareImage,
  restoreMap,
} from "./imaging";
import { Mat } from "./matrix";

export const RETENTION_TOLERANCE = 0.05;
export const MODEL_URL = new URL("focalnet-human.onnx", document.baseURI).href;
export const ORT_WASM_URL = new URL("ort-wasm-simd-threaded.wasm", document.baseURI).href;

export type Prediction = {
  gravity: { x: number; y: number };
  importance_peak: number;
  image: { width: number; height: number };
  crop: Crop;
  importance_crop: Crop;
  human_score: number;
  candidate_count: number;
  retention_regret: number;
  heatmap: Mat;
};

function pixelCrop(box: Float32Array, width: number, height: number): Crop {
  const left = Math.max(0, Math.min(width - 1, Math.trunc(box[0] * width + 0.5)));
  const top = Math.max(0, Math.min(height - 1, Math.trunc(box[1] * height + 0.5)));
  const right = Math.max(left + 1, Math.min(width, Math.trunc(box[2] * width + 0.5)));
  const bottom = Math.max(top + 1, Math.min(height, Math.trunc(box[3] * height + 0.5)));
  return { left, top, width: right - left, height: bottom - top, retained_importance: 0 };
}

export function rankCrops(
  image: RgbaImage,
  heatmap: Mat,
  scores: Float32Array,
  aspectRatio: number,
  retentionTolerance = RETENTION_TOLERANCE,
): Prediction {
  if (retentionTolerance < 0 || retentionTolerance > 1) {
    throw new Error("Retention tolerance must be in [0, 1]");
  }
  const candidates = generateCandidates(image.width, image.height, aspectRatio);
  if (candidates.length > MAX_CANDIDATES) {
    throw new Error(`Generated ${candidates.length} crops; maximum is ${MAX_CANDIDATES}`);
  }
  const pixelCrops = candidates.map((box) => pixelCrop(box, image.width, image.height));
  const retention = pixelCrops.map((crop) => scoreCrop(heatmap, crop, image.width, image.height));
  const maximumRetention = Math.max(...retention);
  let selected = 0;
  let bestScore = -Infinity;
  for (let index = 0; index < candidates.length; index += 1) {
    if (retention[index] < maximumRetention - retentionTolerance) {
      continue;
    }
    if (scores[index] > bestScore) {
      bestScore = scores[index];
      selected = index;
    }
  }
  const crop = {
    ...pixelCrops[selected],
    retained_importance: retention[selected],
  };
  return {
    gravity: focalPoint(heatmap),
    importance_peak: heatmap.max(),
    image: { width: image.width, height: image.height },
    crop,
    importance_crop: solveCrop(heatmap, image.width, image.height, aspectRatio),
    human_score: scores[selected],
    candidate_count: candidates.length,
    retention_regret: maximumRetention - retention[selected],
    heatmap,
  };
}

export class HumanCropPredictor {
  private constructor(private readonly session: ort.InferenceSession) {}

  static async load(modelUrl = MODEL_URL): Promise<HumanCropPredictor> {
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;
    ort.env.wasm.proxy = false;
    ort.env.wasm.wasmPaths = { wasm: ORT_WASM_URL };
    const session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    const inputs = new Set(session.inputNames);
    const outputs = new Set(session.outputNames);
    if (
      session.inputNames.length !== 3 ||
      !inputs.has("image") ||
      !inputs.has("boxes") ||
      !inputs.has("content")
    ) {
      throw new Error("Human crop model has an incompatible input contract");
    }
    if (
      session.outputNames.length !== 2 ||
      !outputs.has("importance") ||
      !outputs.has("crop_scores")
    ) {
      throw new Error("Human crop model has an incompatible output contract");
    }
    return new HumanCropPredictor(session);
  }

  async predict(image: RgbaImage, aspectRatio: number): Promise<Prediction> {
    const { tensor, box } = prepareImage(image);
    const candidates = generateCandidates(image.width, image.height, aspectRatio);
    const results = await this.session.run({
      image: new ort.Tensor("float32", tensor, [1, 3, INPUT_SIZE, INPUT_SIZE]),
      boxes: new ort.Tensor("float32", padBoxes(candidates), [1, MAX_CANDIDATES, 4]),
      content: new ort.Tensor("float32", letterboxContent(box), [1, 4]),
    });
    const importance = results.importance.data as Float32Array;
    const scores = results.crop_scores.data as Float32Array;
    let min = Infinity;
    let max = -Infinity;
    for (const value of importance) {
      if (!Number.isFinite(value)) {
        throw new Error("Model produced invalid importance or crop scores");
      }
      min = Math.min(min, value);
      max = Math.max(max, value);
    }
    for (let index = 0; index < candidates.length; index += 1) {
      if (!Number.isFinite(scores[index])) {
        throw new Error("Model produced invalid importance or crop scores");
      }
    }
    if (min < -1e-5 || max > 1.00001) {
      throw new Error("Model produced invalid importance or crop scores");
    }
    const clipped = Mat.zeros(MAP_SIZE, MAP_SIZE);
    for (let index = 0; index < clipped.data.length; index += 1) {
      clipped.data[index] = Math.min(1, Math.max(0, importance[index]));
    }
    return rankCrops(image, restoreMap(clipped, box), scores, aspectRatio);
  }
}

