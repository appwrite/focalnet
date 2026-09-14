/** Row-major float64 matrix used by the crop and imaging ports. */

export class Mat {
  readonly rows: number;
  readonly cols: number;
  readonly data: Float64Array;

  constructor(rows: number, cols: number, data?: Float64Array) {
    if (rows <= 0 || cols <= 0) {
      throw new Error("Expected a nonempty 2D heatmap");
    }
    this.rows = rows;
    this.cols = cols;
    this.data = data ?? new Float64Array(rows * cols);
    if (this.data.length !== rows * cols) {
      throw new Error("Matrix data does not match shape");
    }
  }

  static zeros(rows: number, cols: number): Mat {
    return new Mat(rows, cols);
  }

  static fill(rows: number, cols: number, value: number): Mat {
    const matrix = new Mat(rows, cols);
    if (value !== 0) {
      matrix.data.fill(value);
    }
    return matrix;
  }

  static fromNested(values: number[][]): Mat {
    if (!values.length || !values[0].length) {
      throw new Error("Expected a nonempty 2D heatmap");
    }
    const rows = values.length;
    const cols = values[0].length;
    const matrix = new Mat(rows, cols);
    for (let row = 0; row < rows; row += 1) {
      if (values[row].length !== cols) {
        throw new Error("Expected a rectangular matrix");
      }
      matrix.data.set(values[row], row * cols);
    }
    return matrix;
  }

  at(row: number, col: number): number {
    return this.data[row * this.cols + col];
  }

  set(row: number, col: number, value: number): void {
    this.data[row * this.cols + col] = value;
  }

  max(): number {
    let peak = -Infinity;
    for (const value of this.data) {
      if (value > peak) {
        peak = value;
      }
    }
    return peak;
  }

  min(): number {
    let floor = Infinity;
    for (const value of this.data) {
      if (value < floor) {
        floor = value;
      }
    }
    return floor;
  }

  sum(): number {
    let total = 0;
    for (const value of this.data) {
      total += value;
    }
    return total;
  }

  clone(): Mat {
    return new Mat(this.rows, this.cols, this.data.slice());
  }
}

export function linspace(start: number, stop: number, count: number): Float64Array {
  const values = new Float64Array(count);
  if (count === 1) {
    values[0] = start;
    return values;
  }
  const step = (stop - start) / (count - 1);
  for (let index = 0; index < count; index += 1) {
    values[index] = start + step * index;
  }
  values[count - 1] = stop;
  return values;
}

export function interp(x: number, nodes: Float64Array, values: Float64Array): number {
  const last = nodes.length - 1;
  if (x <= nodes[0]) {
    return values[0];
  }
  if (x >= nodes[last]) {
    return values[last];
  }
  let low = 0;
  let high = last;
  while (high - low > 1) {
    const mid = (low + high) >> 1;
    if (nodes[mid] <= x) {
      low = mid;
    } else {
      high = mid;
    }
  }
  const span = nodes[high] - nodes[low];
  if (span === 0) {
    return values[low];
  }
  const weight = (x - nodes[low]) / span;
  return values[low] + weight * (values[high] - values[low]);
}

export function isClose(left: number, right: number, rtol = 1e-10, atol = 0): boolean {
  return Math.abs(left - right) <= atol + rtol * Math.abs(right);
}
