"""Agent models for the DataUp SDK."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from dataup.models.enums import (
    AgentProvider,
    AgentType,
    ComputeTier,
    LabelSource,
)


class AgentBase(BaseModel):
    """Base agent schema."""

    name: str = Field(max_length=256, description="Agent name")
    endpoint: HttpUrl = Field(description="Model endpoint URL")
    auth_token: str = Field(max_length=256, description="Authentication token")
    timeout: int = Field(default=30, ge=1, le=300, description="Request timeout in seconds")
    rate_limit: int = Field(default=100, ge=1, description="Rate limit per hour")
    provider: AgentProvider = Field(description="External service provider")
    agent_type: AgentType = Field(default=AgentType.DETECTOR, description="Agent type")
    is_public: bool = Field(default=False, description="If true, the Agent is available to all users")
    labels: list[str] | None = Field(default=None, description="List of class names")
    label_source: LabelSource = Field(
        default=LabelSource.COCO,
        description="Source of the labels: standard like COCO, or custom input",
    )
    tags: list[str] = Field(default_factory=list)


class AgentCreate(AgentBase):
    """Schema for creating an agent."""

    compute_tier: ComputeTier = Field(default=ComputeTier.PRIVATE, description="Compute tier")


class AgentUpdate(BaseModel):
    """Schema for updating an agent (all fields optional)."""

    name: str | None = Field(None, max_length=256)
    endpoint: HttpUrl | None = None
    auth_token: str | None = Field(None, max_length=256)
    timeout: int | None = Field(None, ge=1, le=300)
    rate_limit: int | None = Field(None, ge=1)
    provider: AgentProvider | None = None
    agent_type: AgentType | None = None
    is_public: bool | None = None
    labels: list[str] | None = None
    label_source: LabelSource | None = None
    is_active: bool | None = None
    compute_tier: ComputeTier | None = None
    tags: list[str] | None = None


class Agent(AgentBase):
    """Full agent schema with all fields."""

    model_config = ConfigDict(from_attributes=True, extra="ignore")

    id: str
    owner_id: str
    is_active: bool = False
    created_date: datetime
    updated_date: datetime
    last_used: datetime | None = None
    total_usage_count: int = 0
    error_count: int = 0
    last_error: str | None = None
    resolved_labels: list[str] | None = None
    has_publication_grant: bool = False
    compute_tier: ComputeTier
    tags: list[str]


class AgentRead(BaseModel):
    """Agent read schema (excludes sensitive fields like auth_token and owner_id)."""

    model_config = ConfigDict(from_attributes=True, extra="ignore")

    id: str
    name: str
    endpoint: HttpUrl
    timeout: int
    rate_limit: int
    provider: AgentProvider
    agent_type: AgentType
    is_public: bool
    labels: list[str] | None = None
    label_source: LabelSource
    tags: list[str]
    is_active: bool = False
    created_date: datetime
    updated_date: datetime
    last_used: datetime | None = None
    total_usage_count: int = 0
    error_count: int = 0
    last_error: str | None = None
    resolved_labels: list[str] | None = None
    has_publication_grant: bool = False
    compute_tier: ComputeTier


class AgentUsageMonthly(BaseModel):
    """Monthly usage statistics for an agent."""

    month: date
    agent_id: str
    user_id: str
    api_key_id: str | None = None
    requests_total: int = Field(default=0, ge=0)
    requests_success: int = Field(default=0, ge=0)
    requests_error: int = Field(default=0, ge=0)
