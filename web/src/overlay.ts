import { type Crop } from "./crop";
import { type RgbaImage } from "./imaging";
import { type Mat } from "./matrix";

export type OverlayKind = "human" | "importance" | "heatmap";

function rgbaToImageData(image: RgbaImage): ImageData {
  return new ImageData(new Uint8ClampedArray(image.data), image.width, image.height);
}

function drawSource(target: HTMLCanvasElement, image: RgbaImage): CanvasRenderingContext2D {
  target.width = image.width;
  target.height = image.height;
  const context = target.getContext("2d");
  if (!context) {
    throw new Error("Could not draw the preview");
  }
  context.putImageData(rgbaToImageData(image), 0, 0);
  return context;
}

function strokeCrop(
  context: CanvasRenderingContext2D,
  crop: Crop,
  color: string,
  label: string,
): void {
  context.save();
  context.strokeStyle = color;
  context.lineWidth = Math.max(2, Math.round(Math.min(context.canvas.width, context.canvas.height) / 180));
  context.strokeRect(crop.left + 0.5, crop.top + 0.5, crop.width, crop.height);
  context.fillStyle = color;
  context.font = `600 ${Math.max(12, Math.round(context.canvas.width / 42))}px "IBM Plex Sans", sans-serif`;
  context.fillText(label, crop.left + 8, Math.max(crop.top - 8, 18));
  context.restore();
}

function strokeGravity(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
): void {
  const px = x * width;
  const py = y * height;
  const arm = Math.max(8, Math.round(Math.min(width, height) / 40));
  context.save();
  context.strokeStyle = "#f3efe4";
  context.lineWidth = 2;
  context.beginPath();
  context.moveTo(px - arm, py);
  context.lineTo(px + arm, py);
  context.moveTo(px, py - arm);
  context.lineTo(px, py + arm);
  context.stroke();
  context.beginPath();
  context.arc(px, py, 4, 0, Math.PI * 2);
  context.fillStyle = "#f3efe4";
  context.fill();
  context.restore();
}

function heatmapColor(value: number): [number, number, number, number] {
  const t = Math.min(1, Math.max(0, value));
  const r = Math.round(255 * Math.min(1, t * 1.4));
  const g = Math.round(80 * t);
  const b = Math.round(255 * (0.2 + 0.6 * (1 - t)));
  return [r, g, b, Math.round(210 * Math.pow(t, 0.7))];
}

export function paintFrame(
  canvas: HTMLCanvasElement,
  image: RgbaImage,
  human: Crop,
  importance: Crop,
  gravity: { x: number; y: number },
  heatmap: Mat,
  mode: OverlayKind,
): void {
  const context = drawSource(canvas, image);
  if (mode === "heatmap") {
    const overlay = context.createImageData(heatmap.cols, heatmap.rows);
    for (let row = 0; row < heatmap.rows; row += 1) {
      for (let col = 0; col < heatmap.cols; col += 1) {
        const [r, g, b, a] = heatmapColor(heatmap.at(row, col));
        const index = (row * heatmap.cols + col) * 4;
        overlay.data[index] = r;
        overlay.data[index + 1] = g;
        overlay.data[index + 2] = b;
        overlay.data[index + 3] = a;
      }
    }
    context.imageSmoothingEnabled = true;
    context.drawImage(imageBitmapSource(overlay), 0, 0, image.width, image.height);
  }
  if (mode !== "importance") {
    strokeCrop(context, human, "#ffb703", "Human");
  }
  if (mode !== "human") {
    strokeCrop(context, importance, "#8ecae6", "Importance");
  }
  strokeGravity(context, gravity.x, gravity.y, image.width, image.height);
}

function imageBitmapSource(data: ImageData): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = data.width;
  canvas.height = data.height;
  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("Could not compose the heatmap overlay");
  }
  context.putImageData(data, 0, 0);
  return canvas;
}

export function paintCrop(canvas: HTMLCanvasElement, image: RgbaImage, crop: Crop): void {
  canvas.width = crop.width;
  canvas.height = crop.height;
  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("Could not draw the crop");
  }
  const source = document.createElement("canvas");
  source.width = image.width;
  source.height = image.height;
  const sourceContext = source.getContext("2d");
  if (!sourceContext) {
    throw new Error("Could not draw the crop");
  }
  sourceContext.putImageData(rgbaToImageData(image), 0, 0);
  context.drawImage(
    source,
    crop.left,
    crop.top,
    crop.width,
    crop.height,
    0,
    0,
    crop.width,
    crop.height,
  );
}
