from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import cv2
import fitz
import numpy as np
from PIL import Image


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def estimate_skew_angle(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    points = np.column_stack(np.where(thresh > 0))
    if len(points) < 10:
        return 0.0
    rect = cv2.minAreaRect(points)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    return -angle


def deskew_image(image: np.ndarray) -> np.ndarray:
    angle = estimate_skew_angle(image)
    if math.isclose(angle, 0.0, abs_tol=0.3):
        return image
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def preprocess_image(
    image: Image.Image,
    grayscale: bool = True,
    denoise: bool = True,
    deskew: bool = True,
    max_side: int = 1536,
) -> Image.Image:
    frame = pil_to_bgr(image)

    if deskew:
        frame = deskew_image(frame)

    if denoise:
        frame = cv2.fastNlMeansDenoisingColored(frame, None, 5, 5, 7, 21)

    if grayscale:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    height, width = frame.shape[:2]
    current_max = max(height, width)
    if current_max > max_side:
        scale = max_side / current_max
        new_size = (int(width * scale), int(height * scale))
        frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

    return bgr_to_pil(frame)


def convert_pdf_to_images(pdf_path: str | Path, dpi: int = 200) -> list[Image.Image]:
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    scale = dpi / 72
    matrix = fitz.Matrix(scale, scale)
    pages: list[Image.Image] = []
    for page in doc:
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
        pages.append(image)
    return pages


def load_document_pages(input_path: str | Path, dpi: int = 200) -> list[Image.Image]:
    input_path = Path(input_path)
    if input_path.suffix.lower() == ".pdf":
        return convert_pdf_to_images(input_path, dpi=dpi)
    return [Image.open(input_path).convert("RGB")]


def save_pages(
    pages: Iterable[Image.Image],
    output_dir: str | Path,
    stem: str,
) -> list[Path]:
    output_dir = ensure_dir(output_dir)
    output_paths: list[Path] = []
    for index, page in enumerate(pages, start=1):
        output_path = output_dir / f"{stem}_page_{index}.png"
        page.save(output_path)
        output_paths.append(output_path)
    return output_paths
