import { Mat, interp, isClose, linspace } from "./matrix";

export type Gravity = { x: number; y: number };

export type Crop = {
  left: number;
  top: number;
  width: number;
  height: number;
  retained_importance: number;
};

export function cleanMap(heatmap: Mat): Mat {
  const cleaned = Mat.zeros(heatmap.rows, heatmap.cols);
  for (let index = 0; index < heatmap.data.length; index += 1) {
    const value = heatmap.data[index];
    cleaned.data[index] = Number.isFinite(value) && value > 0 ? value : 0;
  }
  return cleaned;
}

export function focalPoint(heatmap: Mat): Gravity {
  const values = cleanMap(heatmap);
  const total = values.sum();
  if (total === 0) {
    return { x: 0.5, y: 0.5 };
  }
  const { rows, cols } = values;
  let x = 0;
  let y = 0;
  for (let col = 0; col < cols; col += 1) {
    let mass = 0;
    for (let row = 0; row < rows; row += 1) {
      mass += values.at(row, col);
    }
    x += mass * ((col + 0.5) / cols);
  }
  for (let row = 0; row < rows; row += 1) {
    let mass = 0;
    for (let col = 0; col < cols; col += 1) {
      mass += values.at(row, col);
    }
    y += mass * ((row + 0.5) / rows);
  }
  return { x: x / total, y: y / total };
}

function bestOffset(mass: Float64Array, extent: number, window: number): [number, number] {
  const edges = linspace(0, extent, mass.length + 1);
  const cumulative = new Float64Array(mass.length + 1);
  for (let index = 0; index < mass.length; index += 1) {
    cumulative[index + 1] = cumulative[index] + mass[index];
  }
  const center = (extent - window) / 2;
  const unique = new Set<number>();
  const consider = (value: number) => {
    const clipped = Math.min(Math.max(value, 0), extent - window);
    unique.add(clipped);
  };
  for (const edge of edges) {
    consider(Math.floor(edge));
    consider(Math.ceil(edge));
    consider(Math.floor(edge - window));
    consider(Math.ceil(edge - window));
  }
  consider(Math.floor(center));
  consider(Math.ceil(center));
  const candidates = [...unique].sort((left, right) => left - right);
  const scores = candidates.map(
    (offset) => interp(offset + window, edges, cumulative) - interp(offset, edges, cumulative),
  );
  const best = Math.max(...scores);
  let chosen = candidates[0];
  let closest = Infinity;
  for (let index = 0; index < candidates.length; index += 1) {
    if (!isClose(scores[index], best, 1e-10, 0)) {
      continue;
    }
    const distance = Math.abs(candidates[index] - center);
    if (distance < closest) {
      closest = distance;
      chosen = candidates[index];
    }
  }
  return [chosen, best];
}

function halfUp(value: number): number {
  return Math.trunc(value + 0.5);
}

export function solveCrop(
  heatmap: Mat,
  imageWidth: number,
  imageHeight: number,
  aspectRatio: number,
): Crop {
  if (Math.min(imageWidth, imageHeight) <= 0) {
    throw new Error("Image dimensions must be positive");
  }
  if (!Number.isFinite(aspectRatio) || aspectRatio <= 0) {
    throw new Error("Aspect ratio must be finite and positive");
  }
  const values = cleanMap(heatmap);
  let width = imageWidth;
  let height = imageHeight;
  if (width / height > aspectRatio) {
    width = Math.max(1, Math.min(width, halfUp(height * aspectRatio)));
  } else {
    height = Math.max(1, Math.min(height, halfUp(width / aspectRatio)));
  }
  let left = 0;
  let top = 0;
  let retained = 0;
  if (width < imageWidth) {
    const columnMass = new Float64Array(values.cols);
    for (let row = 0; row < values.rows; row += 1) {
      for (let col = 0; col < values.cols; col += 1) {
        columnMass[col] += values.at(row, col);
      }
    }
    [left, retained] = bestOffset(columnMass, imageWidth, width);
  } else {
    const rowMass = new Float64Array(values.rows);
    for (let row = 0; row < values.rows; row += 1) {
      for (let col = 0; col < values.cols; col += 1) {
        rowMass[row] += values.at(row, col);
      }
    }
    [top, retained] = bestOffset(rowMass, imageHeight, height);
  }
  const total = values.sum();
  const fraction = total ? Math.min(1, Math.max(0, retained / total)) : 0;
  return { left, top, width, height, retained_importance: fraction };
}

export function scoreCrop(
  heatmap: Mat,
  crop: Crop,
  imageWidth: number,
  imageHeight: number,
): number {
  const values = cleanMap(heatmap);
  const total = values.sum();
  if (total === 0) {
    return 0;
  }
  const xs = linspace(0, imageWidth, values.cols + 1);
  const ys = linspace(0, imageHeight, values.rows + 1);
  const cellWidth = imageWidth / values.cols;
  const cellHeight = imageHeight / values.rows;
  const weightsX = new Float64Array(values.cols);
  const weightsY = new Float64Array(values.rows);
  const right = crop.left + crop.width;
  const bottom = crop.top + crop.height;
  for (let col = 0; col < values.cols; col += 1) {
    weightsX[col] =
      Math.max(0, Math.min(xs[col + 1], right) - Math.max(xs[col], crop.left)) / cellWidth;
  }
  for (let row = 0; row < values.rows; row += 1) {
    weightsY[row] =
      Math.max(0, Math.min(ys[row + 1], bottom) - Math.max(ys[row], crop.top)) / cellHeight;
  }
  let retained = 0;
  for (let row = 0; row < values.rows; row += 1) {
    if (weightsY[row] === 0) {
      continue;
    }
    let rowMass = 0;
    for (let col = 0; col < values.cols; col += 1) {
      rowMass += values.at(row, col) * weightsX[col];
    }
    retained += weightsY[row] * rowMass;
  }
  return Math.min(1, Math.max(0, retained / total));
}
