import { describe, expect, it } from "vitest";
import { generateCandidates, letterboxContent } from "./candidates";
import { Crop, focalPoint, scoreCrop, solveCrop } from "./crop";
import { coverage, fitLetterbox, prepareImage, projectMap, restoreMap, solidRgba } from "./imaging";
import { Mat } from "./matrix";

describe("crop", () => {
  it("uses mass instead of centering on the centroid for multi-subject maps", () => {
    const heatmap = Mat.zeros(64, 64);
    heatmap.set(30, 6, 4);
    heatmap.set(30, 57, 6);
    const point = focalPoint(heatmap);
    expect(point.x).toBeGreaterThan(0.5);
    expect(point.x).toBeLessThan(0.6);
    expect(point.x).toBeCloseTo(0.5796875, 10);
    expect(point.y).toBeCloseTo(0.4765625, 10);
    const crop = solveCrop(heatmap, 1000, 500, 1);
    expect(crop.left).toBeGreaterThanOrEqual(400);
    expect(crop.left).toBe(407);
    expect(crop.retained_importance).toBeCloseTo(0.6, 10);
    const wide = solveCrop(heatmap, 1000, 500, 1.8);
    expect(wide.retained_importance).toBeCloseTo(1, 10);
    expect(wide.left).toBe(50);
  });

  it("chooses the center for uniform and empty maps", () => {
    const uniform = solveCrop(Mat.fill(8, 8, 1), 1000, 600, 1);
    expect(uniform).toMatchObject({ left: 200, top: 0, width: 600, height: 600 });
    expect(uniform.retained_importance).toBeCloseTo(0.6, 10);
    const empty = solveCrop(Mat.zeros(8, 8), 600, 1000, 1);
    expect(empty).toMatchObject({ left: 0, top: 200, retained_importance: 0 });
    expect(focalPoint(Mat.zeros(8, 8))).toEqual({ x: 0.5, y: 0.5 });
  });

  it("keeps crop position invariant to importance scale", () => {
    const heatmap = Mat.zeros(8, 8);
    heatmap.set(2, 6, 1);
    const regular = solveCrop(heatmap, 1000, 500, 1);
    const tiny = heatmap.clone();
    tiny.set(2, 6, 1e-20);
    const scaled = solveCrop(tiny, 1000, 500, 1);
    expect(scaled.left).toBe(regular.left);
    expect(scaled.left).toBe(375);
    expect(scaled.retained_importance).toBeCloseTo(regular.retained_importance, 10);
  });

  it("matches brute-force fractional-cell integration", () => {
    let seed = 12;
    const random = () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return seed / 0x100000000;
    };
    for (let iteration = 0; iteration < 40; iteration += 1) {
      const width = 5 + Math.floor(random() * 35);
      const height = 5 + Math.floor(random() * 35);
      const heatmap = Mat.zeros(5, 7);
      for (let index = 0; index < heatmap.data.length; index += 1) {
        heatmap.data[index] = random();
      }
      const ratio = 0.2 + random() * 3.8;
      const actual = solveCrop(heatmap, width, height, ratio);
      let best = -Infinity;
      for (let left = 0; left <= width - actual.width; left += 1) {
        for (let top = 0; top <= height - actual.height; top += 1) {
          const candidate: Crop = {
            left,
            top,
            width: actual.width,
            height: actual.height,
            retained_importance: 0,
          };
          best = Math.max(best, scoreCrop(heatmap, candidate, width, height));
        }
      }
      expect(actual.retained_importance).toBeCloseTo(best, 10);
      expect(actual.retained_importance).toBeCloseTo(
        scoreCrop(heatmap, actual, width, height),
        10,
      );
    }
  });

  it("ignores invalid activations", () => {
    expect(focalPoint(Mat.fromNested([[Number.NaN, Infinity], [-1, 1]]))).toEqual({
      x: 0.75,
      y: 0.75,
    });
  });

  it.each([0, -1, Number.NaN, Infinity])("rejects invalid aspect ratio %s", (ratio) => {
    expect(() => solveCrop(Mat.fill(2, 2, 1), 100, 100, ratio)).toThrow(/Aspect ratio/);
  });
});

describe("candidates", () => {
  it.each([1, 16 / 9, 4 / 5])(
    "keeps requested pixel ratio and valid bounds for %s",
    (ratio) => {
      const boxes = generateCandidates(1600, 900, ratio);
      expect(boxes.length).toBeGreaterThan(1);
      expect(boxes.length).toBeLessThanOrEqual(125);
      if (ratio === 1) {
        expect(boxes.length).toBe(105);
        expect([...boxes[0]]).toEqual([0, 0, 0.3656249940395355, 0.6499999761581421]);
        expect([...boxes[boxes.length - 1]]).toEqual([
          0.6343749761581421, 0.3499999940395355, 1, 1,
        ]);
      }
      if (Math.abs(ratio - 16 / 9) < 1e-12) {
        expect(boxes.length).toBe(101);
      }
      if (Math.abs(ratio - 4 / 5) < 1e-12) {
        expect(boxes.length).toBe(105);
      }
      const keys = new Set<string>();
      for (const box of boxes) {
        expect(box[0]).toBeGreaterThanOrEqual(0);
        expect(box[1]).toBeGreaterThanOrEqual(0);
        expect(box[2]).toBeLessThanOrEqual(1);
        expect(box[3]).toBeLessThanOrEqual(1);
        expect(box[2]).toBeGreaterThan(box[0]);
        expect(box[3]).toBeGreaterThan(box[1]);
        const pixelRatio = ((box[2] - box[0]) * 1600) / ((box[3] - box[1]) * 900);
        expect(pixelRatio).toBeCloseTo(ratio, 5);
        keys.add([...box].join(","));
      }
      expect(keys.size).toBe(boxes.length);
    },
  );

  it("normalizes letterbox content bounds", () => {
    expect([...letterboxContent(fitLetterbox(1600, 900))]).toEqual([0, 0.21875, 1, 0.78125]);
  });
});

describe("imaging", () => {
  it("rounds letterbox dimensions half-up like Go", () => {
    const tall = fitLetterbox(512, 257);
    expect(tall.height).toBe(129);
    expect(tall).toMatchObject({ size: 256, left: 0, top: 63, width: 256 });
    expect(fitLetterbox(200, 100)).toMatchObject({
      size: 256,
      left: 0,
      top: 64,
      width: 256,
      height: 128,
    });
  });

  it("keeps aspect, alpha, channel order, and neutral padding", () => {
    const { tensor, box } = prepareImage(solidRgba(200, 100, 200, 100, 50, 128));
    expect(box).toMatchObject({ left: 0, top: 64, width: 256, height: 128 });
    expect(tensor.length).toBe(3 * 256 * 256);
    const center = 128 * 256 + 128;
    const expected = [200, 100, 50].map(
      (channel, index) =>
        (((channel * 128) / 255 / 255 - [0.485, 0.456, 0.406][index]) /
          [0.229, 0.224, 0.225][index]),
    );
    expect(tensor[center]).toBeCloseTo(expected[0], 2);
    expect(tensor[256 * 256 + center]).toBeCloseTo(expected[1], 2);
    expect(tensor[2 * 256 * 256 + center]).toBeCloseTo(expected[2], 2);
    for (let row = 0; row < 64; row += 1) {
      for (let col = 0; col < 256; col += 1) {
        const dest = row * 256 + col;
        expect(tensor[dest]).toBe(0);
        expect(tensor[256 * 256 + dest]).toBe(0);
        expect(tensor[2 * 256 * 256 + dest]).toBe(0);
      }
    }
  });

  it.each([
    [199, 301],
    [640, 320],
    [1, 10000],
    [901, 109],
  ] as const)("does not let padding affect restored heatmaps %s×%s", (width, height) => {
    const box = fitLetterbox(width, height);
    const valid = coverage(box);
    expect(valid.sum()).toBeCloseTo((box.width * box.height) / 16, 6);
    const heatmap = Mat.zeros(64, 64);
    for (let index = 0; index < heatmap.data.length; index += 1) {
      heatmap.data[index] = valid.data[index] > 0 ? 0.4 : 1e6;
    }
    const restored = restoreMap(heatmap, box);
    expect(restored.min()).toBeCloseTo(0.4, 5);
    expect(restored.max()).toBeCloseTo(0.4, 5);
  });

  it("round-trips letterboxed source coordinates", () => {
    const source = Mat.zeros(64, 64);
    for (let row = 0; row < 64; row += 1) {
      for (let col = 0; col < 64; col += 1) {
        source.set(row, col, (col + 0.5) / 64);
      }
    }
    const box = fitLetterbox(303, 509);
    const { projected, valid } = projectMap(source, box);
    const restored = restoreMap(projected, box);
    for (let row = 0; row < 64; row += 1) {
      for (let col = 2; col < 62; col += 1) {
        expect(restored.at(row, col)).toBeCloseTo(source.at(row, col), 1);
      }
    }
    for (let index = 0; index < projected.data.length; index += 1) {
      if (valid.data[index] === 0) {
        expect(projected.data[index]).toBe(0);
      }
    }
  });
});
