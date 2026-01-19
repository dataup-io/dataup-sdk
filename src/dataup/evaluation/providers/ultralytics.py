"""Ultralytics YOLO inference provider."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dataup_models.geom import BoundingBox
from dataup_models.labels import Label

from dataup.evaluation.providers.base import InferenceProvider

if TYPE_CHECKING:
    from PIL import Image


class UltralyticsProvider(InferenceProvider):
    """Inference provider for Ultralytics YOLO models.

    Requires the `ultralytics` package to be installed:
        pip install ultralytics

    Example:
        >>> provider = UltralyticsProvider()
        >>> provider.load_model("yolov8n.pt")
        >>> labels = provider.predict(image, conf=0.25, iou=0.5)
    """

    def __init__(self) -> None:
        """Initialize the Ultralytics provider."""
        self._model: Any | None = None
        self._class_names: list[str] = []

    def load_model(self, weights: str) -> None:
        """Load a YOLO model from weights path.

        Args:
            weights: Path to model weights file (e.g., "yolov8n.pt")
                    or a model name that ultralytics can download.

        Raises:
            ImportError: If ultralytics package is not installed.
        """
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics package is required for UltralyticsProvider. "
                "Install it with: pip install ultralytics"
            ) from e

        self._model = YOLO(weights)
        # Extract class names from the model
        self._class_names = list(self._model.names.values())

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

        Raises:
            RuntimeError: If no model is loaded.
        """
        if self._model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        # Run inference
        results = self._model.predict(
            source=image,
            conf=conf,
            iou=iou,
            verbose=False,
        )

        labels: list[Label] = []

        # Process results (first result since we pass a single image)
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None:
                for box in boxes:
                    # Get box coordinates (xyxy format)
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy

                    # Convert to x, y, width, height format
                    x = int(x1)
                    y = int(y1)
                    width = int(x2 - x1)
                    height = int(y2 - y1)

                    # Get confidence and class
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self._class_names[class_id]

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
