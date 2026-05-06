import re

import cv2
import numpy as np
import pytesseract
from fastapi import APIRouter, File, UploadFile

router = APIRouter()

_RRN_RE = re.compile(r"\d{6}-?\d{7}")
_NAME_RE = re.compile(r"[가-힣]{2,4}")


def _preprocess(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(eq, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray, th


def _roi(th: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = th.shape[:2]
    # 고정 템플릿 기반 ROI (일반적인 주민등록증 정방향 촬영 기준)
    name = th[int(h * 0.28): int(h * 0.43), int(w * 0.16): int(w * 0.56)]
    rrn = th[int(h * 0.50): int(h * 0.66), int(w * 0.16): int(w * 0.84)]
    return name, rrn


def _ocr_confidence(th: np.ndarray) -> float:
    data = pytesseract.image_to_data(
        th,
        config="--psm 6",
        output_type=pytesseract.Output.DICT,
    )
    vals: list[float] = []
    for c in data.get("conf", []):
        try:
            v = float(c)
        except Exception:
            continue
        if v >= 0:
            vals.append(v)
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


@router.post("/ocr")
async def ocr_card(file: UploadFile = File(...)):
    raw = await file.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {
            "status": "fail",
            "reason": "invalid_image",
            "data": {"name": "", "rrn": ""},
            "raw_text": "",
        }

    gray, th = _preprocess(img)
    name_roi, rrn_roi = _roi(th)

    # 3단계 OCR 수행
    full_text = pytesseract.image_to_string(th, config="--psm 6")
    name_text = pytesseract.image_to_string(name_roi, config="--psm 7")
    rrn_text = pytesseract.image_to_string(
        rrn_roi,
        config="--psm 7 -c tessedit_char_whitelist=0123456789-",
    )

    # 4단계 파싱
    rrn_m = _RRN_RE.search(rrn_text) or _RRN_RE.search(full_text)
    rrn = rrn_m.group(0) if rrn_m else ""
    if rrn and "-" not in rrn:
        rrn = f"{rrn[:6]}-{rrn[6:]}"

    name_m = _NAME_RE.search(name_text) or _NAME_RE.search(full_text)
    name = name_m.group(0) if name_m else ""

    # 5단계 검증
    conf = _ocr_confidence(th)
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

    return {
        "status": "success",
        "data": {"name": name, "rrn": rrn},
        "raw_text": full_text,
    }
