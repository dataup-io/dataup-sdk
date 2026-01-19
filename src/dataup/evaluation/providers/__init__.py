"""Inference providers for evaluations."""

from dataup.evaluation.providers.base import InferenceProvider
from dataup.evaluation.providers.roboflow import RoboflowProvider
from dataup.evaluation.providers.ultralytics import UltralyticsProvider

__all__ = [
    "InferenceProvider",
    "RoboflowProvider",
    "UltralyticsProvider",
]
