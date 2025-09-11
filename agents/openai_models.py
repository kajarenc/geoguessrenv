from __future__ import annotations

from typing import Optional, Literal

from pydantic import BaseModel, Field, model_validator


class ClickParams(BaseModel):
    x: int = Field(..., description="X screen coordinate in pixels")
    y: int = Field(..., description="Y screen coordinate in pixels")


class AnswerParams(BaseModel):
    lat: float = Field(..., description="Latitude in degrees")
    lon: float = Field(..., description="Longitude in degrees")


class SubmitAction(BaseModel):
    op: Literal["click", "answer"]
    click: Optional[ClickParams] = None
    answer: Optional[AnswerParams] = None

    @model_validator(mode="after")
    def _validate_required_by_op(self) -> "SubmitAction":
        if self.op == "click" and self.click is None:
            raise ValueError("'click' object must be provided when op=='click'")
        if self.op == "answer" and self.answer is None:
            raise ValueError("'answer' object must be provided when op=='answer'")
        return self


