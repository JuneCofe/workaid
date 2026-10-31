from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    conversation_id: str | None = None
    user_id: str = Field(min_length=1, max_length=120)
    message: str = Field(min_length=1, max_length=8000)


class CaseSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=5, ge=1, le=20)


class CaseAnswersRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=120)
    answers: dict[str, Any] = Field(min_length=1, max_length=20)


class ProfileRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=120)
    facts: list[str] = Field(default_factory=list, max_length=30)
    preferences: dict[str, str] = Field(default_factory=dict)


class ProfileResponse(BaseModel):
    user_id: str
    facts: list[str]
    preferences: dict[str, str]


class ConversationSummary(BaseModel):
    id: str
    title: str
    updated_at: str
    message_count: int


class SourceItem(BaseModel):
    title: str
    category: str
    article_count: int
    path: str
    kind: Literal["statute", "case"]

