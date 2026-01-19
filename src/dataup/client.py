"""Synchronous DataUp API client."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx

from dataup._base import DEFAULT_BASE_URL, DEFAULT_TIMEOUT, BaseClient
from dataup.exceptions import (
    AuthenticationError,
    ConflictError,
    DataUpAPIError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    ValidationError,
)
from dataup.models.agents import Agent, AgentCreate, AgentRead, AgentUpdate, AgentUsageMonthly
from dataup.models.evaluations import (
    BatchIngestRequest,
    BatchIngestResponse,
    EvaluationCreate,
    EvaluationCreateResponse,
    EvaluationFrameRead,
    EvaluationRead,
    FinalizeResponse,
    JobMetricsResponse,
)
from dataup.models.common import CursorPage
from dataup.models.inference import InferenceRequest, InferenceResponse

if TYPE_CHECKING:
    from dataup.models.enums import AgentProvider, AgentType


class AgentsResource:
    """Agents API resource."""

    def __init__(self, client: DataUpClient) -> None:
        self._client = client

    def list(
        self,
        *,
        provider: AgentProvider | str | None = None,
        agent_type: AgentType | str | None = None,
        is_active: bool | None = None,
        is_public: bool | None = None,
        search: str | None = None,
        cursor: str | None = None,
        size: int = 10,
    ) -> CursorPage[AgentRead]:
        """List agents with optional filters and pagination."""
        params: dict[str, Any] = {"size": size}
        if provider is not None:
            params["provider"] = provider.value if hasattr(provider, "value") else provider
        if agent_type is not None:
            params["agent_type"] = agent_type.value if hasattr(agent_type, "value") else agent_type
        if is_active is not None:
            params["is_active"] = is_active
        if is_public is not None:
            params["is_public"] = is_public
        if search is not None:
            params["search"] = search
        if cursor is not None:
            params["cursor"] = cursor

        response = self._client._request("GET", "/agents/", params=params)
        return CursorPage[AgentRead].model_validate(response.json())

    def get(self, agent_id: str) -> AgentRead:
        """Get a specific agent by ID."""
        response = self._client._request("GET", f"/agents/{agent_id}")
        return AgentRead.model_validate(response.json())

    def create(self, agent: AgentCreate) -> AgentRead:
        """Create a new agent."""
        response = self._client._request(
            "POST",
            "/agents/",
            json=agent.model_dump(mode="json", exclude_unset=True),
        )
        return AgentRead.model_validate(response.json())

    def update(self, agent_id: str, agent: AgentUpdate) -> Agent:
        """Update an existing agent."""
        response = self._client._request(
            "PATCH",
            f"/agents/{agent_id}",
            json=agent.model_dump(mode="json", exclude_unset=True),
        )
        return Agent.model_validate(response.json())

    def delete(self, agent_id: str) -> None:
        """Delete an agent."""
        self._client._request("DELETE", f"/agents/{agent_id}")

    def infer(self, agent_id: str, request: InferenceRequest) -> InferenceResponse:
        """Run inference on an agent."""
        response = self._client._request(
            "POST",
            f"/agents/{agent_id}/infer",
            json=request.model_dump(mode="json", exclude_unset=True),
        )
        return InferenceResponse.model_validate(response.json())

    def activate(self, agent_id: str) -> dict[str, str]:
        """Activate an agent."""
        response = self._client._request("POST", f"/agents/{agent_id}/activate")
        return response.json()

    def deactivate(self, agent_id: str) -> dict[str, str]:
        """Deactivate an agent."""
        response = self._client._request("POST", f"/agents/{agent_id}/deactivate")
        return response.json()

    def get_monthly_usage(self, agent_id: str) -> AgentUsageMonthly:
        """Get monthly usage statistics for an agent."""
        response = self._client._request("GET", f"/agents/{agent_id}/monthly-usage")
        return AgentUsageMonthly.model_validate(response.json())


class EvaluationsResource:
    """Evaluations API resource."""

    def __init__(self, client: DataUpClient) -> None:
        self._client = client

    def list(
        self,
        *,
        cursor: str | None = None,
        size: int = 10,
    ) -> CursorPage[EvaluationRead]:
        """List evaluations with pagination."""
        params: dict[str, Any] = {"size": size}
        if cursor is not None:
            params["cursor"] = cursor
        response = self._client._request("GET", "/evaluations/", params=params)
        return CursorPage[EvaluationRead].model_validate(response.json())

    def get(self, evaluation_id: str) -> EvaluationRead:
        """Get a specific evaluation by ID."""
        response = self._client._request("GET", f"/evaluations/{evaluation_id}")
        return EvaluationRead.model_validate(response.json())

    def create(self, evaluation: EvaluationCreate) -> EvaluationCreateResponse:
        """Create a new evaluation.

        Args:
            evaluation: EvaluationCreate object with evaluation parameters.

        Returns:
            EvaluationCreateResponse with the new evaluation ID.
        """
        response = self._client._request(
            "POST",
            "/evaluations/",
            json=evaluation.model_dump(mode="json", exclude_unset=True),
        )
        return EvaluationCreateResponse.model_validate(response.json())

    def delete(self, evaluation_id: str) -> None:
        """Delete an evaluation."""
        self._client._request("DELETE", f"/evaluations/{evaluation_id}")

    def ingest_batch(
        self,
        evaluation_id: str,
        batch: BatchIngestRequest,
    ) -> BatchIngestResponse:
        """Ingest a batch of frames into an evaluation.

        Args:
            evaluation_id: ID of the evaluation.
            batch: BatchIngestRequest with frame data.

        Returns:
            BatchIngestResponse with processing status.
        """
        response = self._client._request(
            "POST",
            f"/evaluations/{evaluation_id}/batches",
            json=batch.model_dump(mode="json", exclude_unset=True),
        )
        return BatchIngestResponse.model_validate(response.json())

    def finalize(self, evaluation_id: str) -> FinalizeResponse:
        """Finalize an evaluation and compute COCO metrics.

        Must be called after all batches have been ingested.

        Args:
            evaluation_id: ID of the evaluation to finalize.

        Returns:
            FinalizeResponse with computed metrics.
        """
        response = self._client._request(
            "POST",
            f"/evaluations/{evaluation_id}/finalize",
        )
        return FinalizeResponse.model_validate(response.json())

    def get_frames(
        self,
        evaluation_id: str,
        *,
        job_id: int | None = None,
        cursor: str | None = None,
        size: int = 10,
    ) -> CursorPage[EvaluationFrameRead]:
        """Get frames for an evaluation, optionally filtered by job_id.

        Args:
            evaluation_id: ID of the evaluation.
            job_id: Optional job ID to filter frames.
            cursor: Pagination cursor.
            size: Number of frames per page.

        Returns:
            CursorPage of EvaluationFrameRead objects.
        """
        params: dict[str, Any] = {"size": size}
        if job_id is not None:
            params["job_id"] = job_id
        if cursor is not None:
            params["cursor"] = cursor
        response = self._client._request(
            "GET",
            f"/evaluations/{evaluation_id}/frames",
            params=params,
        )
        return CursorPage[EvaluationFrameRead].model_validate(response.json())

    def get_job_metrics(
        self,
        evaluation_id: str,
        *,
        confidence_threshold: float = 0.5,
    ) -> JobMetricsResponse:
        """Get job-level metrics for an evaluation.

        Args:
            evaluation_id: ID of the evaluation.
            confidence_threshold: Confidence threshold for computing metrics.

        Returns:
            JobMetricsResponse with per-job metrics.
        """
        params: dict[str, Any] = {"confidence_threshold": confidence_threshold}
        response = self._client._request(
            "GET",
            f"/evaluations/{evaluation_id}/job_metrics",
            params=params,
        )
        return JobMetricsResponse.model_validate(response.json())


class DataUpClient(BaseClient):
    """Synchronous DataUp API client."""

    agents: AgentsResource
    evaluations: EvaluationsResource

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        http_client: httpx.Client | None = None,
    ) -> None:
        super().__init__(api_key, base_url=base_url, timeout=timeout)

        self._client = http_client or httpx.Client(
            timeout=self.timeout,
            headers=self._headers,
        )

        # Initialize resources
        self.agents = AgentsResource(self)
        self.evaluations = EvaluationsResource(self)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """Make HTTP request and handle errors."""
        url = self._build_url(path)
        response = self._client.request(method, url, params=params, json=json)
        self._handle_response(response)
        return response

    def _handle_response(self, response: httpx.Response) -> None:
        """Handle HTTP response and raise appropriate exceptions."""
        if response.is_success:
            return

        status_code = response.status_code
        try:
            error_data = response.json()
            message = error_data.get("detail", error_data.get("message", response.text))
        except Exception:
            message = response.text

        if status_code == 400:
            raise ValidationError(message, status_code=status_code)
        elif status_code == 401:
            raise AuthenticationError(message, status_code=status_code)
        elif status_code == 403:
            raise PermissionDeniedError(message, status_code=status_code)
        elif status_code == 404:
            raise NotFoundError(message, status_code=status_code)
        elif status_code == 409:
            raise ConflictError(message, status_code=status_code)
        elif status_code == 429:
            raise RateLimitError(message, status_code=status_code)
        else:
            raise DataUpAPIError(message, status_code=status_code)

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> DataUpClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
