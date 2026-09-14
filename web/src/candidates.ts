export const MAX_CANDIDATES = 128;
export const DEFAULT_SCALES = [1.0, 0.9, 0.8, 0.7, 0.65] as const;
export const DEFAULT_POSITIONS = 5;

export type Letterbox = {
  size: number;
  left: number;
  top: number;
  width: number;
  height: number;
};

function linspace(start: number, stop: number, count: number): number[] {
  if (count === 1) {
    return [start];
  }
  const step = (stop - start) / (count - 1);
  const values = Array.from({ length: count }, (_, index) => start + step * index);
  values[count - 1] = stop;
  return values;
}

function round7(value: number): number {
  return Math.round(value * 1e7) / 1e7;
}

function uniqueSortedBoxes(boxes: number[][]): Float32Array[] {
  const seen = new Map<string, Float32Array>();
  for (const box of boxes) {
    const rounded = new Float32Array(4);
    for (let index = 0; index < 4; index += 1) {
      rounded[index] = Math.fround(round7(Math.fround(box[index])));
    }
    seen.set(rounded.join(","), rounded);
  }
  return [...seen.values()].sort((left, right) => {
    for (let index = 0; index < 4; index += 1) {
      if (left[index] < right[index]) {
        return -1;
      }
      if (left[index] > right[index]) {
        return 1;
      }
    }
    return 0;
  });
}

export function generateCandidates(
  imageWidth: number,
  imageHeight: number,
  aspectRatio: number,
  positions = DEFAULT_POSITIONS,
  scales: readonly number[] = DEFAULT_SCALES,
): Float32Array[] {
  if (Math.min(imageWidth, imageHeight, positions) <= 0) {
    throw new Error("Image dimensions and candidate positions must be positive");
  }
  if (!Number.isFinite(aspectRatio) || aspectRatio <= 0) {
    throw new Error("Aspect ratio must be finite and positive");
  }
  if (
    !scales.length ||
    scales.some((scale) => !Number.isFinite(scale) || scale <= 0 || scale > 1)
  ) {
    throw new Error("Candidate scales must be finite values in (0, 1]");
  }
  const sourceRatio = imageWidth / imageHeight;
  const maximumWidth = sourceRatio > aspectRatio ? aspectRatio / sourceRatio : 1;
  const maximumHeight = sourceRatio > aspectRatio ? 1 : sourceRatio / aspectRatio;
  const boxes: number[][] = [];
  for (const scale of scales) {
    const width = maximumWidth * scale;
    const height = maximumHeight * scale;
    const xs = linspace(0, 1 - width, width < 1 - 1e-7 ? positions : 1);
    const ys = linspace(0, 1 - height, height < 1 - 1e-7 ? positions : 1);
    for (const top of ys) {
      for (const left of xs) {
        boxes.push([left, top, left + width, top + height]);
      }
    }
  }
  const unique = uniqueSortedBoxes(boxes);
  if (unique.length > MAX_CANDIDATES) {
    throw new Error(`Generated ${unique.length} crops; maximum is ${MAX_CANDIDATES}`);
  }
  return unique;
}

export function letterboxContent(box: Letterbox): Float32Array {
  return new Float32Array([
    box.left / box.size,
    box.top / box.size,
    (box.left + box.width) / box.size,
    (box.top + box.height) / box.size,
  ]);
}

export function padBoxes(boxes: Float32Array[]): Float32Array {
  const padded = new Float32Array(MAX_CANDIDATES * 4);
  for (let index = 0; index < boxes.length; index += 1) {
    padded.set(boxes[index], index * 4);
  }
  return padded;
}
