import asyncio
import re
from pathlib import Path
from uuid import uuid4


class VisionImageStorageError(Exception):
    pass


def _safe_filename_prefix(call_id: str) -> str:
    safe_call_id = re.sub(r"[^A-Za-z0-9_.-]", "_", call_id)
    return safe_call_id or "call"


def _write_file(path: Path, content: bytes) -> None:
    path.write_bytes(content)


async def save_vision_image(
    content: bytes,
    extension: str,
    call_id: str,
    upload_dir: str,
) -> str:
    try:
        directory = Path(upload_dir)
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"{_safe_filename_prefix(call_id)}_{uuid4().hex}{extension}"
        path = directory / filename
        await asyncio.to_thread(_write_file, path, content)
        return str(path)
    except Exception as e:
        raise VisionImageStorageError() from e
