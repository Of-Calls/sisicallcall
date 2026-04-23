from __future__ import annotations

from typing import Optional, List
from pydantic import BaseModel


class AuthInitiateRequest(BaseModel):
    call_id: str
    tenant_id: str
    customer_ref: str
    customer_phone: str


class AuthInitiateResponse(BaseModel):
    auth_id: str
    status: str
    message: str


class LivenessInstructionsResponse(BaseModel):
    auth_id: str
    instructions: List[str]
    token: str


class LivenessCompleteRequest(BaseModel):
    token: str


class LivenessCompleteResponse(BaseModel):
    auth_id: str
    liveness_passed: bool


class FaceVerifyResponse(BaseModel):
    auth_id: str
    verified: bool
    similarity_score: float
    attempts_remaining: int


class AuthStatusResponse(BaseModel):
    auth_id: str
    status: str
    liveness_passed: bool
    face_verified: bool
    created_at: Optional[str] = None


class FaceRegisterResponse(BaseModel):
    tenant_id: str
    customer_ref: str
    registered_at: str
