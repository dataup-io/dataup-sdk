"""Roboflow inference provider."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from dataup.evaluation.providers.base import InferenceProvider
from dataup_models.geom import BoundingBox
from dataup_models.labels import Label

if TYPE_CHECKING:
    from PIL import Image


class RoboflowProvider(InferenceProvider):
    """Inference provider for Roboflow models.

    Requires the `roboflow` package to be installed:
        pip install roboflow

    Also requires ROBOFLOW_API_KEY environment variable to be set.

    Example:
        >>> provider = RoboflowProvider()
        >>> provider.load_model("project-name/version")  # e.g., "coco/1"
        >>> labels = provider.predict(image, conf=0.25)
    """

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the Roboflow provider.

        Args:
            api_key: Roboflow API key. If not provided, will read from
                    ROBOFLOW_API_KEY environment variable.
        """
        self._api_key = api_key or os.environ.get("ROBOFLOW_API_KEY")
        self._model: Any | None = None
        self._class_names: list[str] = []
        self._project_name: str = ""

    def load_model(self, weights: str) -> None:
        """Load a Roboflow model by project/version.

        Args:
            weights: Model identifier in format "workspace/project/version"
                    or "project/version". Examples:
                    - "my-workspace/my-project/1"
                    - "coco-dataset/1"

        Raises:
            ImportError: If roboflow package is not installed.
            ValueError: If ROBOFLOW_API_KEY is not set.
            ValueError: If weights format is invalid.
        """
        try:
            from roboflow import Roboflow
        except ImportError as e:
            raise ImportError(
                "roboflow package is required for RoboflowProvider. "
                "Install it with: pip install roboflow"
            ) from e

        if not self._api_key:
            raise ValueError(
                "Roboflow API key is required. Set ROBOFLOW_API_KEY environment "
                "variable or pass api_key to RoboflowProvider constructor."
            )

        # Parse weights format
        parts = weights.split("/")
        if len(parts) == 2:
            # Format: project/version
            project_name, version = parts
            workspace = None
        elif len(parts) == 3:
            # Format: workspace/project/version
            workspace, project_name, version = parts
        else:
            raise ValueError(
                f"Invalid weights format: {weights}. "
                "Expected 'project/version' or 'workspace/project/version'."
            )

        # Initialize Roboflow client
        rf = Roboflow(api_key=self._api_key)

        # Get project and model
        if workspace:
            project = rf.workspace(workspace).project(project_name)
        else:
            project = rf.project(project_name)

        self._model = project.version(int(version)).model
        self._project_name = project_name

        # Try to get class names from the model
        # Note: Roboflow models may have class names in different places
        if hasattr(self._model, "classes"):
            self._class_names = list(self._model.classes)
        else:
            self._class_names = []

    def predict(
        self,
        image: Image.Image,
        *,
        conf: float = 0.25,
        iou: float = 0.5,
    ) -> list[Label]:
        """Run inference on a single image.

        Args:
            image: PIL Image to run inference on.
            conf: Confidence threshold for detections.
            iou: IoU threshold for NMS (note: Roboflow may not support this).

        Returns:
            List of Label objects representing detections.

        Raises:
            RuntimeError: If no model is loaded.
        """
        if self._model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        # Roboflow expects a file path or numpy array
        # Convert PIL Image to a format Roboflow accepts
        import tempfile

        import numpy as np

        # Convert PIL to numpy array
        img_array = np.array(image)

        # Save temporarily for Roboflow (it works better with file paths)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            image.save(tmp_file.name)
            tmp_path = tmp_file.name

        try:
            # Run inference
            result = self._model.predict(tmp_path, confidence=int(conf * 100)).json()
        finally:
            # Clean up temp file
            os.unlink(tmp_path)

        labels: list[Label] = []

        # Process predictions
        predictions = result.get("predictions", [])
        for pred in predictions:
            # Roboflow returns center coordinates and dimensions
            x_center = pred.get("x", 0)
            y_center = pred.get("y", 0)
            width = pred.get("width", 0)
            height = pred.get("height", 0)

            # Convert to top-left corner format
            x = int(x_center - width / 2)
            y = int(y_center - height / 2)
            width = int(width)
            height = int(height)

            class_name = pred.get("class", "unknown")
            confidence = pred.get("confidence", 0.0)

            # Update class names if we find new ones
            if class_name not in self._class_names:
                self._class_names.append(class_name)

            labels.append(
                Label(
                    label=class_name,
                    score=confidence,
                    bbox=BoundingBox(x=x, y=y, width=width, height=height),
                )
            )

        return labels

    @property
    def class_names(self) -> list[str]:
        """Return list of class names the model can detect.

        Returns:
            List of class name strings.
        """
        return self._class_names

    @property
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded.

        Returns:
            True if a model is loaded, False otherwise.
        """
        return self._model is not None
