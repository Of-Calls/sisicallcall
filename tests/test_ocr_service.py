"""IdCardOCRService 단위 테스트 — Tesseract/OpenCV 실호출 없이 목으로 검증."""

import numpy as np
import pytest
from unittest.mock import patch

from app.services.ocr.id_card_ocr_service import IdCardOCRService


@pytest.fixture
def svc() -> IdCardOCRService:
    return IdCardOCRService()


def test_process_image_sync_invalid_image(svc: IdCardOCRService) -> None:
    with patch(
        "app.services.ocr.id_card_ocr_service.cv2.imdecode",
        return_value=None,
    ):
        out = svc.process_image_sync(b"not-an-image")
    assert out["status"] == "fail"
    assert out["reason"] == "invalid_image"


def test_process_image_sync_rrn_not_found(svc: IdCardOCRService) -> None:
    fake_bgr = np.zeros((120, 120, 3), dtype=np.uint8)
    with patch(
        "app.services.ocr.id_card_ocr_service.cv2.imdecode",
        return_value=fake_bgr,
    ):
        with patch(
            "pytesseract.image_to_string",
            return_value="텍스트만 있고 숫자 패턴 없음",
        ):
            with patch(
                "pytesseract.image_to_data",
                return_value={"conf": ["55"]},
            ):
                out = svc.process_image_sync(b"jpeg-bytes")
    assert out["status"] == "retry"
    assert out["reason"] == "rrn_not_found"


def test_process_image_sync_low_confidence(svc: IdCardOCRService) -> None:
    fake_bgr = np.zeros((120, 120, 3), dtype=np.uint8)

    def fake_to_string(image, **kwargs):
        cfg = kwargs.get("config") or ""
        if "whitelist" in cfg:
            return "900101-1234567"
        if "--psm 7" in cfg:
            return "홍길동"
        return "noise"

    with patch(
        "app.services.ocr.id_card_ocr_service.cv2.imdecode",
        return_value=fake_bgr,
    ):
        with patch(
            "pytesseract.image_to_string",
            side_effect=fake_to_string,
        ):
            with patch(
                "pytesseract.image_to_data",
                return_value={"conf": ["10", "20"]},
            ):
                out = svc.process_image_sync(b"jpeg-bytes")
    assert out["status"] == "retry"
    assert out["reason"] == "low_confidence"


def test_process_image_sync_success(svc: IdCardOCRService) -> None:
    fake_bgr = np.zeros((120, 120, 3), dtype=np.uint8)

    def fake_to_string(image, **kwargs):
        cfg = kwargs.get("config") or ""
        if "whitelist" in cfg:
            return "900101-1234567"
        if "--psm 7" in cfg:
            return "홍길동"
        return "noise"

    with patch(
        "app.services.ocr.id_card_ocr_service.cv2.imdecode",
        return_value=fake_bgr,
    ):
        with patch(
            "pytesseract.image_to_string",
            side_effect=fake_to_string,
        ):
            with patch(
                "pytesseract.image_to_data",
                return_value={"conf": ["60", "70", "80"]},
            ):
                out = svc.process_image_sync(b"jpeg-bytes")
    assert out["status"] == "success"
    assert out["data"]["name"] == "홍길동"
    assert out["data"]["rrn"] == "900101-1234567"


@pytest.mark.asyncio
async def test_process_image_async_delegates_to_sync(svc: IdCardOCRService) -> None:
    with patch.object(IdCardOCRService, "process_image_sync") as m:
        m.return_value = {
            "status": "fail",
            "reason": "invalid_image",
            "data": {"name": "", "rrn": ""},
            "raw_text": "",
        }
        out = await svc.process_image(b"x")
    m.assert_called_once_with(b"x")
    assert out["status"] == "fail"
