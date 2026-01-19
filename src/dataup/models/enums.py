"""Enum definitions for the DataUp SDK."""

from enum import Enum


class ComputeTier(str, Enum):
    """Compute tier for agent inference."""

    STANDARD = "standard"
    HEAVY = "heavy"
    HEAVYPLUS = "heavy_plus"
    PRIVATE = "private"


class LabelSource(str, Enum):
    """Source of label definitions."""

    CUSTOM = "custom"
    COCO = "coco"


class AgentType(str, Enum):
    """Type of agent."""

    DETECTOR = "detector"
    INTERACTOR = "interactor"
    REID = "reid"
    TRACKER = "tracker"


class AgentProvider(str, Enum):
    """External service provider for agents."""

    ULTRALYTICS = "ultralytics"
    HUGGINGFACE = "huggingface"
    ROBOFLOW = "roboflow"
    LANDINGAI = "landingai"
    DATAUP = "dataup"
    ZINKIAI = "zinkiai"
    CUSTOM = "custom"


class AgentRequestStatus(str, Enum):
    """Status of an agent request."""

    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"
