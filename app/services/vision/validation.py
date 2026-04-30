from dataclasses import dataclass

from fastapi import UploadFile

try:
    import magic
except ImportError:  # pragma: no cover - exercised only when dependency is absent
    magic = None


ALLOWED_IMAGE_MIME_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


@dataclass(frozen=True)
class ValidatedImage:
    content: bytes
    mime_type: str
    extension: str
    size_bytes: int


class EmptyImageFileError(Exception):
    pass


class ImageFileTooLargeError(Exception):
    pass


class UnsupportedImageMimeTypeError(Exception):
    pass


def detect_mime_type(content: bytes) -> str:
    if magic is None:
        raise RuntimeError("python-magic is not installed")
    return magic.from_buffer(content, mime=True)


async def validate_image_upload(file: UploadFile, max_bytes: int) -> ValidatedImage:
    content = await file.read()
    size_bytes = len(content)

    if size_bytes == 0:
        raise EmptyImageFileError()
    if size_bytes > max_bytes:
        raise ImageFileTooLargeError()

    mime_type = detect_mime_type(content)
    if isinstance(mime_type, bytes):
        mime_type = mime_type.decode("utf-8")

    extension = ALLOWED_IMAGE_MIME_TYPES.get(mime_type)
    if extension is None:
        raise UnsupportedImageMimeTypeError()

    return ValidatedImage(
        content=content,
        mime_type=mime_type,
        extension=extension,
        size_bytes=size_bytes,
    )
