"""DataUp Python SDK - Official client library for the DataUp API."""

from dataup._pagination import paginate, paginate_async
from dataup._version import __version__
from dataup.async_client import AsyncDataUpClient
from dataup.client import DataUpClient
from dataup.exceptions import (
    AuthenticationError,
    ConflictError,
    ConnectionError,
    DataUpAPIError,
    DataUpError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    TimeoutError,
    ValidationError,
)
from dataup.models.agents import (
    Agent,
    AgentCreate,
    AgentRead,
    AgentUpdate,
    AgentUsageMonthly,
)
from dataup.models.evaluations import (
    BatchIngestRequest,
    BatchIngestResponse,
    COCOSummaryMetrics,
    EvaluationCreate,
    EvaluationCreateResponse,
    EvaluationFrameRead,
    EvaluationRead,
    EvaluationStatus,
    FinalizeResponse,
    FrameData,
    JobMetrics,
    JobMetricsResponse,
    PerClassCOCOMetrics,
    PredictionSource,
    ThresholdMetrics,
)
from dataup.models.common import CursorPage
from dataup.models.enums import (
    AgentProvider,
    AgentType,
    ComputeTier,
    LabelSource,
)
from dataup.models.inference import (
    DetectionResults,
    DetectorParams,
    InferenceRequest,
    InferenceResponse,
    SAM3Params,
)

# Evaluation module (high-level classes)
from dataup.evaluation import (
    EvaluationRunner,
    InferenceProvider,
    RoboflowProvider,
    UltralyticsProvider,
)

__all__ = [
    # Version
    "__version__",
    # Clients
    "DataUpClient",
    "AsyncDataUpClient",
    # Exceptions
    "DataUpError",
    "DataUpAPIError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "ConflictError",
    "ValidationError",
    "RateLimitError",
    "ConnectionError",
    "TimeoutError",
    # Pagination
    "paginate",
    "paginate_async",
    # Enums
    "AgentProvider",
    "AgentType",
    "ComputeTier",
    "LabelSource",
    # Agent models
    "Agent",
    "AgentCreate",
    "AgentUpdate",
    "AgentRead",
    "AgentUsageMonthly",
    # Evaluation models
    "EvaluationCreate",
    "EvaluationCreateResponse",
    "EvaluationRead",
    "EvaluationFrameRead",
    "EvaluationStatus",
    "PredictionSource",
    "FrameData",
    "BatchIngestRequest",
    "BatchIngestResponse",
    "FinalizeResponse",
    "COCOSummaryMetrics",
    "PerClassCOCOMetrics",
    "ThresholdMetrics",
    "JobMetrics",
    "JobMetricsResponse",
    # Inference models
    "InferenceRequest",
    "InferenceResponse",
    "DetectorParams",
    "SAM3Params",
    "DetectionResults",
    # Common
    "CursorPage",
    # Evaluation module
    "EvaluationRunner",
    "InferenceProvider",
    "UltralyticsProvider",
    "RoboflowProvider",
]
