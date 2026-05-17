"""Pydantic schemas for ML inputs and outputs."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TrainingSummary(BaseModel):
    """Training summary schema."""

    accuracy: float = Field(..., ge=0, le=1)
    precision: float = Field(..., ge=0, le=1)
    recall: float = Field(..., ge=0, le=1)
    f1: float = Field(..., ge=0, le=1)
