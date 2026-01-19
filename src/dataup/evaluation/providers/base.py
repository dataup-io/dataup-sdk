"""Abstract base class for inference providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

    from dataup_models.labels import Label


class InferenceProvider(ABC):
    """Abstract base class for inference providers.

    Inference providers implement model loading and prediction for specific
    ML frameworks (e.g., Ultralytics, Roboflow).

    Example:
        >>> provider = UltralyticsProvider()
        >>> provider.load_model("yolov8n.pt")
        >>> labels = provider.predict(image, conf=0.25, iou=0.5)
    """

    @abstractmethod
    def load_model(self, weights: str) -> None:
        """Load a model from weights path or model name.

        Args:
            weights: Path to model weights file or model identifier.
        """

    @abstractmethod
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
            iou: IoU threshold for NMS.

        Returns:
            List of Label objects representing detections.
        """

    @property
    @abstractmethod
    def class_names(self) -> list[str]:
        """Return list of class names the model can detect.

        Returns:
            List of class name strings.
        """

    @property
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded.

        Returns:
            True if a model is loaded, False otherwise.
        """
        return False
