import "./style.css";
import { type RgbaImage, loadOrientedImage } from "./imaging";
import { type OverlayKind, paintCrop, paintFrame } from "./overlay";
import { HumanCropPredictor, type Prediction } from "./runtime";

const RATIOS: { label: string; value: number }[] = [
  { label: "1:1", value: 1 },
  { label: "4:5", value: 4 / 5 },
  { label: "3:2", value: 3 / 2 },
  { label: "4:3", value: 4 / 3 },
  { label: "16:9", value: 16 / 9 },
  { label: "9:16", value: 9 / 16 },
];

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) {
  throw new Error("Missing #app");
}

app.innerHTML = `
  <div class="shell">
    <header class="hero">
      <div>
        <h1>FocalNet crop lab</h1>
        <p class="lede">
          Drop a photo to compare the human-ranked crop against pure importance
          retention. Everything runs in this tab; the picture never leaves the browser.
        </p>
      </div>
      <div class="badge">CPC → GAICD v2 · ONNX Web</div>
    </header>
    <div class="stage">
      <section class="panel drop" id="drop">
        <input id="file" type="file" accept="image/jpeg,image/png,image/webp" />
        <div class="drop-copy" id="empty">
          <strong>Drop an image</strong>
          JPEG, PNG, or WebP · stays on this device
        </div>
        <canvas id="frame" hidden></canvas>
      </section>
      <aside class="panel">
        <div class="controls" id="ratios"></div>
        <div class="controls" id="modes">
          <button data-mode="human" class="active" type="button">Human crop</button>
          <button data-mode="importance" type="button">Importance crop</button>
          <button data-mode="heatmap" type="button">Heatmap</button>
        </div>
        <div class="crops">
          <figure>
            <canvas id="human-crop"></canvas>
            <figcaption class="human"><span>Human-ranked</span><span id="human-meta">—</span></figcaption>
          </figure>
          <figure>
            <canvas id="importance-crop"></canvas>
            <figcaption class="importance"><span>Importance</span><span id="importance-meta">—</span></figcaption>
          </figure>
        </div>
        <dl class="stats">
          <div class="stat"><dt>Gravity</dt><dd id="gravity">—</dd></div>
          <div class="stat"><dt>Peak</dt><dd id="peak">—</dd></div>
          <div class="stat"><dt>Candidates</dt><dd id="candidates">—</dd></div>
          <div class="stat"><dt>Retention regret</dt><dd id="regret">—</dd></div>
        </dl>
        <p class="status" id="status">Loading the 19 MB ranking model…</p>
      </aside>
    </div>
    <p class="note">
      The ranking head is trained on CPC and GAICD judgments. This preview URL is
      unlisted, not private: anyone with the link can download the ONNX file.
    </p>
  </div>
`;

const drop = $("#drop");
const empty = $("#empty");
const fileInput = $<HTMLInputElement>("#file");
const frame = $<HTMLCanvasElement>("#frame");
const humanCanvas = $<HTMLCanvasElement>("#human-crop");
const importanceCanvas = $<HTMLCanvasElement>("#importance-crop");
const status = $("#status");
const ratios = $("#ratios");
const modes = $("#modes");

let predictor: HumanCropPredictor | null = null;
let image: RgbaImage | null = null;
let prediction: Prediction | null = null;
let aspectRatio = 1;
let overlay: OverlayKind = "human";

for (const ratio of RATIOS) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `chip${ratio.value === aspectRatio ? " active" : ""}`;
  button.textContent = ratio.label;
  button.addEventListener("click", () => {
    aspectRatio = ratio.value;
    for (const node of ratios.querySelectorAll("button")) {
      node.classList.toggle("active", node === button);
    }
    void run();
  });
  ratios.append(button);
}

modes.addEventListener("click", (event) => {
  const button = (event.target as HTMLElement).closest("button[data-mode]");
  if (!(button instanceof HTMLButtonElement)) {
    return;
  }
  overlay = button.dataset.mode as OverlayKind;
  for (const node of modes.querySelectorAll("button")) {
    node.classList.toggle("active", node === button);
  }
  render();
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file) {
    void openFile(file);
  }
});

for (const eventName of ["dragenter", "dragover"] as const) {
  drop.addEventListener(eventName, (event) => {
    event.preventDefault();
    drop.classList.add("active");
  });
}
drop.addEventListener("dragleave", () => drop.classList.remove("active"));
drop.addEventListener("drop", (event) => {
  event.preventDefault();
  drop.classList.remove("active");
  const file = event.dataTransfer?.files[0];
  if (file) {
    void openFile(file);
  }
});

void boot();

async function boot(): Promise<void> {
  try {
    predictor = await HumanCropPredictor.load();
    setStatus("Model ready. Drop a photo to crop it.");
  } catch (error) {
    setStatus(message(error), true);
  }
}

async function openFile(file: File): Promise<void> {
  try {
    image = await loadOrientedImage(file);
    empty.hidden = true;
    frame.hidden = false;
    drop.classList.add("loaded");
    await run();
  } catch (error) {
    setStatus(message(error), true);
  }
}

async function run(): Promise<void> {
  if (!predictor || !image) {
    return;
  }
  setStatus("Scoring crops…");
  const started = performance.now();
  try {
    prediction = await predictor.predict(image, aspectRatio);
    render();
    setStatus(`Cropped in ${Math.round(performance.now() - started)} ms on this device.`);
  } catch (error) {
    setStatus(message(error), true);
  }
}

function render(): void {
  if (!image || !prediction) {
    return;
  }
  paintFrame(
    frame,
    image,
    prediction.crop,
    prediction.importance_crop,
    prediction.gravity,
    prediction.heatmap,
    overlay,
  );
  paintCrop(humanCanvas, image, prediction.crop);
  paintCrop(importanceCanvas, image, prediction.importance_crop);
  $("#human-meta").textContent = `${pct(prediction.crop.retained_importance)} kept`;
  $("#importance-meta").textContent = `${pct(prediction.importance_crop.retained_importance)} kept`;
  $("#gravity").textContent =
    `${prediction.gravity.x.toFixed(3)}, ${prediction.gravity.y.toFixed(3)}`;
  $("#peak").textContent = prediction.importance_peak.toFixed(3);
  $("#candidates").textContent = String(prediction.candidate_count);
  $("#regret").textContent = pct(prediction.retention_regret);
}

function pct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function setStatus(text: string, error = false): void {
  status.textContent = text;
  status.classList.toggle("error", error);
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : "Something went wrong.";
}

function $<T extends HTMLElement>(selector: string): T {
  const node = document.querySelector(selector);
  if (!node) {
    throw new Error(`Missing ${selector}`);
  }
  return node as T;
}
