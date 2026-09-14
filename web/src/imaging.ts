import { type Letterbox } from "./candidates";
import { Mat, linspace } from "./matrix";

export const INPUT_SIZE = 256;
export const MAP_SIZE = 64;
export const MEAN = [0.485, 0.456, 0.406] as const;
export const STD = [0.229, 0.224, 0.225] as const;
export const MAX_PIXELS = 20_000_000;

export type RgbaImage = {
  width: number;
  height: number;
  data: Uint8ClampedArray;
};

export function fitLetterbox(
  width: number,
  height: number,
  size: number = INPUT_SIZE,
): Letterbox {
  if (Math.min(width, height, size) <= 0) {
    throw new Error("Image and input dimensions must be positive");
  }
  const scale = Math.min(size / width, size / height);
  const fittedWidth = Math.max(1, Math.min(size, Math.trunc(width * scale + 0.5)));
  const fittedHeight = Math.max(1, Math.min(size, Math.trunc(height * scale + 0.5)));
  return {
    size,
    left: Math.floor((size - fittedWidth) / 2),
    top: Math.floor((size - fittedHeight) / 2),
    width: fittedWidth,
    height: fittedHeight,
  };
}

export function coverage(box: Letterbox, mapSize: number = MAP_SIZE): Mat {
  const edges = linspace(0, box.size, mapSize + 1);
  const cell = box.size / mapSize;
  const weightsX = new Float64Array(mapSize);
  const weightsY = new Float64Array(mapSize);
  const right = box.left + box.width;
  const bottom = box.top + box.height;
  for (let index = 0; index < mapSize; index += 1) {
    weightsX[index] = Math.max(
      0,
      Math.min(edges[index + 1], right) - Math.max(edges[index], box.left),
    );
    weightsY[index] = Math.max(
      0,
      Math.min(edges[index + 1], bottom) - Math.max(edges[index], box.top),
    );
  }
  const mask = Mat.zeros(mapSize, mapSize);
  const area = cell * cell;
  for (let row = 0; row < mapSize; row += 1) {
    for (let col = 0; col < mapSize; col += 1) {
      mask.set(row, col, (weightsY[row] * weightsX[col]) / area);
    }
  }
  return mask;
}

function sample(values: Mat, xs: Float64Array, ys: Float64Array): Mat {
  const result = Mat.zeros(ys.length, xs.length);
  const lastCol = values.cols - 1;
  const lastRow = values.rows - 1;
  for (let row = 0; row < ys.length; row += 1) {
    const y = Math.min(Math.max(ys[row], 0), lastRow);
    const y0 = Math.floor(y);
    const y1 = Math.min(y0 + 1, lastRow);
    const wy = y - y0;
    for (let col = 0; col < xs.length; col += 1) {
      const x = Math.min(Math.max(xs[col], 0), lastCol);
      const x0 = Math.floor(x);
      const x1 = Math.min(x0 + 1, lastCol);
      const wx = x - x0;
      const top = values.at(y0, x0) * (1 - wx) + values.at(y0, x1) * wx;
      const bottom = values.at(y1, x0) * (1 - wx) + values.at(y1, x1) * wx;
      result.set(row, col, top * (1 - wy) + bottom * wy);
    }
  }
  return result;
}

export function projectMap(values: Mat, box: Letterbox): { projected: Mat; valid: Mat } {
  const centers = new Float64Array(MAP_SIZE);
  for (let index = 0; index < MAP_SIZE; index += 1) {
    centers[index] = (index + 0.5) * (box.size / MAP_SIZE);
  }
  const xs = new Float64Array(MAP_SIZE);
  const ys = new Float64Array(MAP_SIZE);
  for (let index = 0; index < MAP_SIZE; index += 1) {
    xs[index] = ((centers[index] - box.left) / box.width) * values.cols - 0.5;
    ys[index] = ((centers[index] - box.top) / box.height) * values.rows - 0.5;
  }
  const valid = coverage(box);
  const projected = sample(values, xs, ys);
  for (let index = 0; index < projected.data.length; index += 1) {
    if (valid.data[index] === 0) {
      projected.data[index] = 0;
    }
  }
  return { projected, valid };
}

export function restoreMap(values: Mat, box: Letterbox, size: number = MAP_SIZE): Mat {
  if (values.rows !== values.cols) {
    throw new Error("Expected a square 2D map");
  }
  const grid = values.rows;
  const centers = new Float64Array(size);
  for (let index = 0; index < size; index += 1) {
    centers[index] = (index + 0.5) / size;
  }
  const xs = new Float64Array(size);
  const ys = new Float64Array(size);
  for (let index = 0; index < size; index += 1) {
    xs[index] = ((box.left + centers[index] * box.width) / box.size) * grid - 0.5;
    ys[index] = ((box.top + centers[index] * box.height) / box.size) * grid - 0.5;
  }
  const validMask = coverage(box, grid);
  const valid = Mat.zeros(grid, grid);
  const masked = Mat.zeros(grid, grid);
  for (let index = 0; index < validMask.data.length; index += 1) {
    const inside = validMask.data[index] > 0 ? 1 : 0;
    valid.data[index] = inside;
    masked.data[index] = inside > 0 ? values.data[index] : 0;
  }
  const numerator = sample(masked, xs, ys);
  const denominator = sample(valid, xs, ys);
  const restored = Mat.zeros(size, size);
  for (let index = 0; index < restored.data.length; index += 1) {
    restored.data[index] = numerator.data[index] / Math.max(denominator.data[index], 1e-8);
  }
  return restored;
}

function resizeRgba(image: RgbaImage, width: number, height: number): RgbaImage {
  if (width === image.width && height === image.height) {
    return { width, height, data: new Uint8ClampedArray(image.data) };
  }
  const resized = new Uint8ClampedArray(width * height * 4);
  for (let row = 0; row < height; row += 1) {
    const y = ((row + 0.5) * image.height) / height - 0.5;
    const yClamped = Math.min(Math.max(y, 0), image.height - 1);
    const y0 = Math.floor(yClamped);
    const y1 = Math.min(y0 + 1, image.height - 1);
    const wy = yClamped - y0;
    for (let col = 0; col < width; col += 1) {
      const x = ((col + 0.5) * image.width) / width - 0.5;
      const xClamped = Math.min(Math.max(x, 0), image.width - 1);
      const x0 = Math.floor(xClamped);
      const x1 = Math.min(x0 + 1, image.width - 1);
      const wx = xClamped - x0;
      const dest = (row * width + col) * 4;
      for (let channel = 0; channel < 4; channel += 1) {
        const top =
          image.data[(y0 * image.width + x0) * 4 + channel] * (1 - wx) +
          image.data[(y0 * image.width + x1) * 4 + channel] * wx;
        const bottom =
          image.data[(y1 * image.width + x0) * 4 + channel] * (1 - wx) +
          image.data[(y1 * image.width + x1) * 4 + channel] * wx;
        resized[dest + channel] = Math.round(top * (1 - wy) + bottom * wy);
      }
    }
  }
  return { width, height, data: resized };
}

export function prepareImage(
  image: RgbaImage,
  size: number = INPUT_SIZE,
): { tensor: Float32Array; box: Letterbox } {
  const box = fitLetterbox(image.width, image.height, size);
  const resized = resizeRgba(image, box.width, box.height);
  const tensor = new Float32Array(3 * size * size);
  const plane = size * size;
  for (let row = 0; row < box.height; row += 1) {
    for (let col = 0; col < box.width; col += 1) {
      const source = (row * box.width + col) * 4;
      const red = resized.data[source];
      const green = resized.data[source + 1];
      const blue = resized.data[source + 2];
      const alpha = resized.data[source + 3];
      const dest = (box.top + row) * size + (box.left + col);
      tensor[dest] = (Math.floor((red * 257 * alpha) / 255) / 65535 - MEAN[0]) / STD[0];
      tensor[plane + dest] =
        (Math.floor((green * 257 * alpha) / 255) / 65535 - MEAN[1]) / STD[1];
      tensor[2 * plane + dest] =
        (Math.floor((blue * 257 * alpha) / 255) / 65535 - MEAN[2]) / STD[2];
    }
  }
  return { tensor, box };
}

export async function loadOrientedImage(file: File): Promise<RgbaImage> {
  const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  try {
    if (bitmap.width * bitmap.height > MAX_PIXELS) {
      throw new Error(`Image exceeds 20 megapixels: ${file.name}`);
    }
    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) {
      throw new Error("Could not read the uploaded image");
    }
    context.drawImage(bitmap, 0, 0);
    const pixels = context.getImageData(0, 0, bitmap.width, bitmap.height);
    return { width: pixels.width, height: pixels.height, data: pixels.data };
  } finally {
    bitmap.close();
  }
}

export function solidRgba(
  width: number,
  height: number,
  red: number,
  green: number,
  blue: number,
  alpha: number,
): RgbaImage {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let index = 0; index < data.length; index += 4) {
    data[index] = red;
    data[index + 1] = green;
    data[index + 2] = blue;
    data[index + 3] = alpha;
  }
  return { width, height, data };
}
