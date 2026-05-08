"""주민등록증 스타일 이미지 OCR 서비스."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_RRN_RE = re.compile(r"\d{6}-?\d{7}")
_NAME_RE = re.compile(r"[가-힣]{2,4}")
_ROI_PROFILE_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "ocr_roi_profile.json"
)


def _parse_korean_name(*texts: str) -> str:
    for text in texts:
        # 한국 신분증 이름 줄은 "김 주 미 (金...)"처럼 OCR되는 경우가 많다.
        before_paren = re.split(r"[\(\[（]", text, maxsplit=1)[0]
        compact = re.sub(r"(?<=[가-힣])\s+(?=[가-힣])", "", before_paren)
        match = _NAME_RE.search(compact)
        if match:
            return match.group(0)
    return ""


def _filter_small_strokes(binary_img: np.ndarray) -> np.ndarray:
    """작은 점 노이즈를 제거하고 일정 크기 이상의 획만 남긴다."""
    # THRESH_BINARY 기준: 글자(검정)=0, 배경(흰색)=255 -> 연결요소 분석을 위해 반전
    inv = cv2.bitwise_not(binary_img)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)

    h, w = binary_img.shape[:2]
    min_area = max(24, int((h * w) * 0.00012))
    min_stroke_len = max(6, int(min(h, w) * 0.015))

    kept = np.zeros_like(inv)
    for idx in range(1, num_labels):  # 0은 배경
        x = stats[idx, cv2.CC_STAT_LEFT]
        y = stats[idx, cv2.CC_STAT_TOP]
        ww = stats[idx, cv2.CC_STAT_WIDTH]
        hh = stats[idx, cv2.CC_STAT_HEIGHT]
        area = stats[idx, cv2.CC_STAT_AREA]

        # 아주 작은 점/짧은 찌꺼기 제거
        if area < min_area:
            continue
        if max(ww, hh) < min_stroke_len:
            continue

        kept[labels == idx] = 255

    # 다시 OCR 입력 포맷(검정 글자 / 흰 배경)으로 복원
    return cv2.bitwise_not(kept)


def _preprocess(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # 1) 그레이스케일
    gray = (
        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        if len(img_bgr.shape) == 3
        else img_bgr.copy()
    )

    # 2) 작은 글자 인식 향상을 위한 3배 업스케일
    scaled = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

    # 3) CLAHE로 조명 불균일 보정
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(scaled)

    # 4) 가우시안 블러로 미세 노이즈 제거
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # 5) Adaptive Threshold로 이진화
    th = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        8,
    )
    th = _filter_small_strokes(th)
    return scaled, th


def _load_saved_roi_profile() -> dict[str, float] | None:
    if not _ROI_PROFILE_PATH.exists():
        return None
    try:
        raw = json.loads(_ROI_PROFILE_PATH.read_text(encoding="utf-8"))
        return {
            "roi_x1": float(raw["roi_x1"]),
            "roi_x2": float(raw["roi_x2"]),
            "roi_y1": float(raw["roi_y1"]),
            "roi_y2": float(raw["roi_y2"]),
        }
    except Exception:
        return None


def _roi_bounds(
    img: np.ndarray, roi_override: dict[str, float] | None = None
) -> tuple[int, int, int, int]:
    h, w = img.shape[:2]
    roi_data = roi_override or _load_saved_roi_profile()
    if roi_data:
        x1 = max(0.0, min(1.0, float(roi_data.get("roi_x1", 0.14))))
        x2 = max(0.0, min(1.0, float(roi_data.get("roi_x2", 0.57))))
        y1 = max(0.0, min(1.0, float(roi_data.get("roi_y1", 0.18))))
        y2 = max(0.0, min(1.0, float(roi_data.get("roi_y2", 0.63))))
    else:
        x1, x2, y1, y2 = 0.14, 0.57, 0.18, 0.63

    if x2 <= x1:
        x2 = min(1.0, x1 + 0.05)
    if y2 <= y1:
        y2 = min(1.0, y1 + 0.05)

    ix1, ix2 = int(w * x1), int(w * x2)
    iy1, iy2 = int(h * y1), int(h * y2)
    return ix1, ix2, iy1, iy2


def _ocr_confidence(th: np.ndarray) -> float:
    import pytesseract

    data = pytesseract.image_to_data(
        th,
        lang="kor",
        config="--psm 6",
        output_type=pytesseract.Output.DICT,
    )
    vals: list[float] = []
    for c in data.get("conf", []):
        try:
            v = float(c)
        except (TypeError, ValueError):
            continue
        if v >= 0:
            vals.append(v)
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _split_name_rrn_roi(preprocessed_roi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """단일 ROI를 이름/주민번호 하위 영역으로 분리한다."""
    h = preprocessed_roi.shape[0]
    name_end = max(1, int(h * 0.40))
    rrn_start = min(h - 1, int(h * 0.45))
    name_roi = preprocessed_roi[:name_end, :]
    rrn_roi = preprocessed_roi[rrn_start:, :]
    return name_roi, rrn_roi


def _crop_to_text_region(preprocessed_roi: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    """이진화 ROI에서 글자(검정 픽셀) 영역만 동적으로 감싸서 crop."""
    if preprocessed_roi.size == 0:
        return preprocessed_roi, None

    # binary(흰 배경/검은 글자) -> 글자를 foreground로 뒤집기
    inv = cv2.bitwise_not(preprocessed_roi)

    # 잔점 제거 + 글자 연결
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel, iterations=1)
    merged = cv2.dilate(cleaned, kernel, iterations=1)

    ys, xs = np.where(merged > 0)
    if len(xs) == 0 or len(ys) == 0:
        return preprocessed_roi, None

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    # 너무 타이트하면 OCR이 깨지므로 여백 추가
    h, w = preprocessed_roi.shape[:2]
    pad_x = max(6, int(w * 0.02))
    pad_y = max(6, int(h * 0.02))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w - 1, x2 + pad_x)
    y2 = min(h - 1, y2 + pad_y)

    cropped = preprocessed_roi[y1 : y2 + 1, x1 : x2 + 1]
    return cropped, (x1, y1, x2, y2)


class IdCardOCRService:
    def process_image_sync(
        self, image_bytes: bytes, roi_override: dict[str, float] | None = None
    ) -> dict[str, Any]:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return {
                "status": "fail",
                "reason": "invalid_image",
                "data": {"name": "", "rrn": ""},
                "raw_text": "",
            }

        import pytesseract

        x1, x2, y1, y2 = _roi_bounds(img, roi_override)
        roi_img = img[y1:y2, x1:x2]
        _, merged_roi_raw = _preprocess(roi_img)
        merged_roi, text_bbox = _crop_to_text_region(merged_roi_raw)
        name_roi, rrn_roi = _split_name_rrn_roi(merged_roi)
        if roi_override:
            print(f"[OCR] roi_override={roi_override}")
        if text_bbox:
            print(f"[OCR] dynamic_text_bbox={text_bbox}")
        else:
            print("[OCR] dynamic_text_bbox=None (원본 ROI 사용)")

        full_text = pytesseract.image_to_string(
            merged_roi, lang="kor+eng", config="--psm 6"
        )
        roi_text = pytesseract.image_to_string(
            merged_roi,
            lang="kor+eng",
            config="--psm 6",
        )
        rrn_text = pytesseract.image_to_string(
            merged_roi,
            lang="eng",
            config="--psm 6 -c tessedit_char_whitelist=0123456789-",
        )
        name_split_text = pytesseract.image_to_string(
            name_roi,
            lang="kor",
            config="--psm 7",
        )
        rrn_split_text = pytesseract.image_to_string(
            rrn_roi,
            lang="eng",
            config="--psm 7 -c tessedit_char_whitelist=0123456789-",
        )

        rrn_m = (
            _RRN_RE.search(rrn_split_text)
            or _RRN_RE.search(rrn_text)
            or _RRN_RE.search(roi_text)
        )
        rrn = rrn_m.group(0) if rrn_m else ""
        if rrn and "-" not in rrn and len(rrn) >= 13:
            rrn = f"{rrn[:6]}-{rrn[6:]}"

        name = _parse_korean_name(name_split_text, roi_text, full_text)

        ocr_data = pytesseract.image_to_data(
            merged_roi,
            lang="kor",
            config="--psm 6",
            output_type=pytesseract.Output.DICT,
        )
        tokens = [str(t).strip() for t in ocr_data.get("text", []) if str(t).strip()]
        kor_tokens = [t for t in tokens if re.search(r"[가-힣]", t)]
        num_tokens = [t for t in tokens if re.search(r"\d", t)]
        rrn_like_tokens = [t for t in tokens if re.search(r"\d{6}-?\d{1,7}", t)]

        print("[OCR] full_text:\n" + full_text)
        print("[OCR] merged_roi_text:\n" + roi_text)
        print("[OCR] merged_roi_rrn_text:\n" + rrn_text)
        print("[OCR] split_name_text:\n" + name_split_text)
        print("[OCR] split_rrn_text:\n" + rrn_split_text)
        print(f"[OCR] tokens_all({len(tokens)}): {tokens}")
        print(f"[OCR] tokens_kor({len(kor_tokens)}): {kor_tokens}")
        print(f"[OCR] tokens_num({len(num_tokens)}): {num_tokens}")
        print(f"[OCR] tokens_rrn_like({len(rrn_like_tokens)}): {rrn_like_tokens}")
        print(f"[OCR] parsed_name={name!r} parsed_rrn={rrn!r}")

        conf = _ocr_confidence(merged_roi)
        print(f"[OCR] confidence={conf:.2f}")
        if not rrn:
            return {
                "status": "retry",
                "reason": "rrn_not_found",
                "data": {"name": name, "rrn": ""},
                "raw_text": full_text,
            }
        if conf < 45.0:
            return {
                "status": "retry",
                "reason": "low_confidence",
                "data": {"name": name, "rrn": rrn},
                "raw_text": full_text,
            }

        print(f"[OCR] CONFIRMED name={name!r} rrn={rrn!r}")
        return {
            "status": "success",
            "data": {"name": name, "rrn": rrn},
            "raw_text": full_text,
        }

    async def process_image(
        self, image_bytes: bytes, roi_override: dict[str, float] | None = None
    ) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self.process_image_sync, image_bytes, roi_override
        )


_id_card_ocr_service: IdCardOCRService | None = None


def get_id_card_ocr_service() -> IdCardOCRService:
    global _id_card_ocr_service
    if _id_card_ocr_service is None:
        _id_card_ocr_service = IdCardOCRService()
    return _id_card_ocr_service
