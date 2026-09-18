"""Render labeled FocalNet comparison figures for the Hub model card."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from focalnet.human_runtime import HumanCropPredictor

FONT_MEDIUM = Path("/usr/share/fonts/truetype/macos/Inter-Medium.ttf")
FONT_SEMIBOLD = Path("/usr/share/fonts/truetype/macos/Inter-SemiBold.ttf")

BG = (246, 247, 249, 255)
INK = (17, 18, 23, 255)
WHITE = (255, 255, 255, 255)
CENTER_COLOR = (232, 168, 12, 255)
FOCAL_COLOR = (240, 46, 101, 255)
PAD = 24
GAP = 16
LABEL_H = 40
RADIUS = 20
CROP_SIZE = 460


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


def center_crop(image: Image.Image, aspect: float) -> tuple[int, int, int, int]:
    width, height = image.size
    if width / height >= aspect:
        crop_h = height
        crop_w = max(1, min(width, int(round(height * aspect))))
    else:
        crop_w = width
        crop_h = max(1, min(height, int(round(width / aspect))))
    left = (width - crop_w) // 2
    top = (height - crop_h) // 2
    return left, top, crop_w, crop_h


def take_crop(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    left, top, width, height = box
    return image.crop((left, top, left + width, top + height))


def scale_to_width(image: Image.Image, width: int) -> Image.Image:
    height = max(1, round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def scale_to_fit(image: Image.Image, size: int) -> Image.Image:
    fitted = image.copy()
    fitted.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (246, 247, 249))
    canvas.paste(fitted, ((size - fitted.width) // 2, (size - fitted.height) // 2))
    return canvas


def round_panel(image: Image.Image) -> Image.Image:
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, *image.size), RADIUS, fill=255)
    out = Image.new("RGBA", image.size, (0, 0, 0, 0))
    out.paste(image.convert("RGBA"), mask=mask)
    return out


def draw_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    color: tuple[int, ...],
    width: int,
) -> None:
    left, top, crop_w, crop_h = box
    for inset in range(width):
        draw.rounded_rectangle(
            (
                left + inset,
                top + inset,
                left + crop_w - 1 - inset,
                top + crop_h - 1 - inset,
            ),
            radius=max(4, 14 - inset),
            outline=color,
        )


def draw_pill(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fill: tuple[int, ...],
    text_font: ImageFont.FreeTypeFont,
    text_fill: tuple[int, ...] = WHITE,
) -> None:
    x, y = xy
    left, top, right, bottom = draw.textbbox((x, y), text, font=text_font)
    pad_x, pad_y = 10, 5
    draw.rounded_rectangle(
        (left - pad_x, top - pad_y, right + pad_x, bottom + pad_y),
        8,
        fill=fill,
    )
    draw.text((x, y), text, font=text_font, fill=text_fill)


def annotate_source(
    image: Image.Image,
    center: tuple[int, int, int, int],
    focal: tuple[int, int, int, int],
) -> Image.Image:
    canvas = image.convert("RGBA")
    shade = Image.new("RGBA", canvas.size, (10, 12, 18, 96))
    hole = Image.new("L", canvas.size, 0)
    left, top, width, height = focal
    ImageDraw.Draw(hole).rounded_rectangle((left, top, left + width, top + height), 16, fill=255)
    canvas = Image.composite(canvas, Image.alpha_composite(canvas, shade), hole)
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw_box(draw, center, CENTER_COLOR, 5)
    draw_box(draw, focal, FOCAL_COLOR, 7)
    caption = font(FONT_SEMIBOLD, max(20, image.width // 36))
    draw_pill(draw, (focal[0] + 18, focal[1] + 16), "FocalNet", FOCAL_COLOR, caption)
    draw_pill(draw, (center[0] + 18, center[1] + 16), "Center", CENTER_COLOR, caption, INK)
    return Image.alpha_composite(canvas, overlay)


def colorize_heatmap(heatmap: np.ndarray, size: tuple[int, int]) -> Image.Image:
    values = np.clip(heatmap.astype(np.float32), 0, 1)
    values = (values - values.min()) / (float(np.ptp(values)) + 1e-6)
    heat = (
        np.asarray(
            Image.fromarray((values * 255).astype(np.uint8), "L").resize(
                size, Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
        / 255.0
    )
    rgb = np.zeros((*heat.shape, 4), dtype=np.uint8)
    rgb[..., 0] = np.clip(80 + 175 * heat + 80 * np.power(heat, 4), 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip(20 + 40 * heat + 180 * np.power(heat, 3), 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip(60 + 40 * (1 - heat) + 80 * heat, 0, 255).astype(np.uint8)
    rgb[..., 3] = np.clip(50 + 160 * heat, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, "RGBA")


def labeled_panel(image: Image.Image, title: str) -> Image.Image:
    photo = round_panel(image)
    canvas = Image.new("RGBA", (photo.width, photo.height + LABEL_H), (0, 0, 0, 0))
    canvas.paste(photo, (0, 0), photo)
    draw = ImageDraw.Draw(canvas)
    draw.text((6, photo.height + 10), title, font=font(FONT_MEDIUM, 22), fill=INK)
    return canvas


def compose_story(source: Image.Image, left: Image.Image, right: Image.Image) -> Image.Image:
    width = PAD * 2 + max(source.width, left.width + GAP + right.width)
    height = PAD * 2 + source.height + GAP + max(left.height, right.height)
    canvas = Image.new("RGBA", (width, height), BG)
    canvas.paste(source, ((width - source.width) // 2, PAD), source)
    y = PAD + source.height + GAP
    x = (width - (left.width + GAP + right.width)) // 2
    canvas.paste(left, (x, y), left)
    canvas.paste(right, (x + left.width + GAP, y), right)
    return canvas.convert("RGB")


def render_comparison(
    source: Path,
    human: HumanCropPredictor,
    aspect: float,
    aspect_label: str,
    output: Path,
    *,
    heatmap: bool = False,
) -> None:
    image = Image.open(source).convert("RGB")
    result, importance = human.predict(source, aspect)
    crop = result["crop"]
    focal = (crop["left"], crop["top"], crop["width"], crop["height"])
    center = center_crop(image, aspect)
    annotated = annotate_source(image, center, focal)
    if heatmap:
        annotated = Image.alpha_composite(
            annotated.convert("RGBA"), colorize_heatmap(importance, image.size)
        )
    source_w = CROP_SIZE * 2 + GAP
    source_panel = labeled_panel(scale_to_width(annotated, source_w), "Source")
    center_panel = labeled_panel(
        scale_to_fit(take_crop(image, center), CROP_SIZE), f"Center {aspect_label}"
    )
    focal_panel = labeled_panel(
        scale_to_fit(take_crop(image, focal), CROP_SIZE),
        f"FocalNet {aspect_label}",
    )
    figure = compose_story(source_panel, center_panel, focal_panel)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.save(output, "PNG", optimize=True)
    print(f"wrote {output} {figure.size} {output.stat().st_size / 1024:.0f} KiB")


def main() -> None:
    human = HumanCropPredictor("/tmp/focalnet-hub-staging/focalnet-human.onnx")
    photos = Path("/tmp/autogravity-photos")
    out = Path("/tmp/hub-comparisons")
    jobs = [
        (photos / "golden-retriever-original.jpg", 1.0, "1:1", "comparison-retriever.png", False),
        (photos / "multiple-faces.jpg", 1.0, "1:1", "comparison-faces.png", False),
    ]
    for source, aspect, label, name, heat in jobs:
        render_comparison(source, human, aspect, label, out / name, heatmap=heat)


if __name__ == "__main__":
    main()
