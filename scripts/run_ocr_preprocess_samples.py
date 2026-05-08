"""ocr_preprocess 디렉터리의 샘플 이미지로 전처리/OCR을 로컬 실험한다."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pytesseract

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.services.ocr.id_card_ocr_service import (  # noqa: E402
    _NAME_RE,
    _RRN_RE,
    _ocr_confidence,
)

DEFAULT_INPUT_DIR = ROOT_DIR / "ocr_preprocess"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
logger = logging.getLogger("ocr_preprocess_samples")


def _remove_red_annotations(img_bgr: np.ndarray) -> np.ndarray:
    """디버그용 빨간 박스/라인은 OCR 노이즈라 흰 배경으로 제거한다."""
    cleaned = img_bgr.copy()
    blue, green, red = cv2.split(cleaned)
    red_mask = (red > 150) & (green < 120) & (blue < 120)
    cleaned[red_mask] = (255, 255, 255)
    return cleaned


def _to_binary_sample(img_bgr: np.ndarray) -> np.ndarray:
    """ocr_preprocess 샘플은 이미 전처리된 이미지라 재전처리 대신 이진화만 정규화한다."""
    cleaned = _remove_red_annotations(img_bgr)
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _crop_by_ratio(
    img: np.ndarray, x1: float, y1: float, x2: float, y2: float
) -> np.ndarray:
    h, w = img.shape[:2]
    ix1 = max(0, min(w - 1, int(w * x1)))
    ix2 = max(ix1 + 1, min(w, int(w * x2)))
    iy1 = max(0, min(h - 1, int(h * y1)))
    iy2 = max(iy1 + 1, min(h, int(h * y2)))
    return img[iy1:iy2, ix1:ix2]


def _resize_for_ocr(img: np.ndarray, min_height: int = 80) -> np.ndarray:
    h = img.shape[0]
    if h >= min_height:
        return img

    scale = min_height / max(h, 1)
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _split_sample_name_rrn(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # 현재 ocr_preprocess 샘플은 이름/주민번호가 이미 ROI 안에 들어온 디버그 이미지다.
    # 재탐색보다 고정 비율 crop이 빨간 박스/사선 패턴 노이즈 영향을 덜 받는다.
    name_roi = _crop_by_ratio(binary, 0.10, 0.23, 0.88, 0.58)
    rrn_roi = _crop_by_ratio(binary, 0.10, 0.63, 0.88, 0.87)
    return _resize_for_ocr(name_roi), _resize_for_ocr(rrn_roi)


def _parse_name(*texts: str) -> str:
    for text in texts:
        normalized = re.sub(r"(?<=[가-힣])\s+(?=[가-힣])", "", text)
        match = _NAME_RE.search(normalized)
        if match:
            return match.group(0)
    return ""


def _parse_rrn(*texts: str) -> str:
    normalized_texts = [
        text.replace(" ", "")
        .replace("－", "-")
        .replace("–", "-")
        .replace("—", "-")
        for text in texts
    ]

    for normalized in normalized_texts:
        match = _RRN_RE.search(normalized)
        if match:
            rrn = match.group(0)
            return rrn if "-" in rrn else f"{rrn[:6]}-{rrn[6:]}"

    # Tesseract가 선행 0을 누락해 12자리로 읽는 경우가 있어 샘플 검증용으로 보정한다.
    for normalized in normalized_texts:
        digits = re.sub(r"\D", "", normalized)
        if len(digits) == 12:
            rrn = f"0{digits}"
            return f"{rrn[:6]}-{rrn[6:]}"
    return ""


def _iter_images(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _run_one(image_path: Path) -> dict[str, object]:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return {
            "image": str(image_path),
            "status": "fail",
            "reason": "invalid_image",
        }

    preprocessed = _to_binary_sample(img)
    name_roi, rrn_roi = _split_sample_name_rrn(preprocessed)

    full_text = pytesseract.image_to_string(
        _resize_for_ocr(preprocessed), lang="kor+eng", config="--psm 6"
    )
    rrn_text = pytesseract.image_to_string(
        rrn_roi,
        lang="eng",
        config="--psm 7 -c tessedit_char_whitelist=0123456789-",
    )
    name_split_text = pytesseract.image_to_string(name_roi, lang="kor", config="--psm 7")
    rrn_split_text = pytesseract.image_to_string(
        rrn_roi,
        lang="eng",
        config="--psm 13 -c tessedit_char_whitelist=0123456789-",
    )

    rrn = _parse_rrn(full_text, rrn_text, rrn_split_text)
    name = _parse_name(name_split_text, full_text)

    data = pytesseract.image_to_data(
        _resize_for_ocr(preprocessed),
        lang="kor",
        config="--psm 6",
        output_type=pytesseract.Output.DICT,
    )
    tokens = [str(t).strip() for t in data.get("text", []) if str(t).strip()]

    return {
        "image": str(image_path),
        "status": "success",
        "shape": img.shape,
        "preprocessed_shape": preprocessed.shape,
        "name_roi_shape": name_roi.shape,
        "rrn_roi_shape": rrn_roi.shape,
        "confidence": _ocr_confidence(preprocessed),
        "name": name,
        "rrn": rrn,
        "full_text": full_text.strip(),
        "rrn_text": rrn_text.strip(),
        "name_split_text": name_split_text.strip(),
        "rrn_split_text": rrn_split_text.strip(),
        "tokens": tokens,
        "kor_tokens": [t for t in tokens if re.search(r"[가-힣]", t)],
        "num_tokens": [t for t in tokens if re.search(r"\d", t)],
    }


def _print_result(result: dict[str, object]) -> None:
    print("=" * 80)
    print(f"image: {result['image']}")
    print(f"status: {result['status']}")
    if result["status"] != "success":
        print(f"reason: {result.get('reason', '')}")
        return

    logger.info(
        "recognized image=%s name=%r rrn=%r confidence=%.2f",
        result["image"],
        result["name"],
        result["rrn"],
        float(result["confidence"]),
    )
    print(f"shape: {result['shape']} -> {result['preprocessed_shape']}")
    print(f"name_roi_shape: {result['name_roi_shape']}")
    print(f"rrn_roi_shape: {result['rrn_roi_shape']}")
    print(f"confidence: {float(result['confidence']):.2f}")
    print(f"parsed_name: {result['name']!r}")
    print(f"parsed_rrn: {result['rrn']!r}")
    print(f"tokens: {result['tokens']}")
    print(f"kor_tokens: {result['kor_tokens']}")
    print(f"num_tokens: {result['num_tokens']}")
    print("--- full_text ---")
    print(result["full_text"])
    print("--- rrn_text ---")
    print(result["rrn_text"])
    print("--- split_name_text ---")
    print(result["name_split_text"])
    print("--- split_rrn_text ---")
    print(result["rrn_split_text"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(
        description="ocr_preprocess 이미지들을 OpenCV로 전처리한 뒤 Tesseract OCR을 실행합니다."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"이미지를 읽을 디렉터리. 기본값: {DEFAULT_INPUT_DIR}",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.exists():
        raise SystemExit(f"input dir not found: {input_dir}")

    image_paths = _iter_images(input_dir)
    if not image_paths:
        raise SystemExit(f"image not found: {input_dir}")

    print(f"[OCR SAMPLE] input_dir={input_dir} images={len(image_paths)}")
    for image_path in image_paths:
        _print_result(_run_one(image_path))


if __name__ == "__main__":
    main()
