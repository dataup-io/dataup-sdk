"""Utility functions for CVAT client."""

from __future__ import annotations

from typing import Any

from dataup_models.geom import BoundingBox, Polygon
from dataup_models.labels import Label, LabelAttribute

from dataup.cvat.models.annotations import AttributeValue, Shape
from dataup.cvat.models.enums import ShapeType
from dataup.cvat.models.tasks import CVATLabel, CVATLabelAttribute


def shape_to_label(
    shape: dict[str, Any] | Shape,
    cvat_labels: list[CVATLabel],
    *,
    score: float = 1.0,
) -> Label:
    # Convert dict to Shape if needed
    if isinstance(shape, dict):
        shape_type = ShapeType(shape.get("type", "rectangle"))
        points = shape.get("points", [])
        attributes = shape.get("attributes", [])
        label_id = shape.get("label_id")
    else:
        shape_type = shape.type
        points = shape.points
        attributes = shape.attributes
        label_id = shape.label_id

    # Resolve label name from CVATLabel
    label_name = f"label_{label_id}"  # Default fallback
    label_attr_map: dict[int, CVATLabelAttribute] = {}  # spec_id -> CVATLabelAttribute

    if label_id is not None:
        for cvat_label in cvat_labels:
            if cvat_label.id == label_id:
                label_name = cvat_label.name
                # Build attribute spec_id -> name mapping
                label_attr_map = {attr.id: attr for attr in cvat_label.attributes}
                break

    # Convert attributes using CVATLabelAttribute to get proper names
    label_attributes = []
    for attr in attributes:
        if isinstance(attr, dict):
            spec_id = attr.get("spec_id")
            attr_value = attr.get("value", "")
        elif isinstance(attr, AttributeValue):
            spec_id = attr.spec_id
            attr_value = attr.value
        else:
            continue

        if not attr_value:
            continue

        # Resolve attribute name from CVATLabelAttribute
        attr_key = label_attr_map[spec_id].name if spec_id in label_attr_map else str(spec_id)

        label_attributes.append(LabelAttribute(key=attr_key, value=attr_value))

    # Convert points to bbox/polygon based on shape type
    bbox: BoundingBox | None = None
    polygon: Polygon | None = None

    if shape_type == ShapeType.RECTANGLE:
        # Rectangle: points = [x1, y1, x2, y2]
        if len(points) >= 4:
            x1, y1, x2, y2 = points[0], points[1], points[2], points[3]
            x_min = min(x1, x2)
            y_min = min(y1, y2)
            x_max = max(x1, x2)
            y_max = max(y1, y2)
            bbox = BoundingBox(
                x=int(x_min),
                y=int(y_min),
                width=int(x_max - x_min),
                height=int(y_max - y_min),
            )
    elif shape_type == ShapeType.POLYGON:
        # Polygon: points = [x1, y1, x2, y2, x3, y3, ...]
        if len(points) >= 6:  # At least 3 points (6 coordinates)
            polygon_points = [
                (int(points[i]), int(points[i + 1])) for i in range(0, len(points), 2)
            ]
            polygon = Polygon(points=polygon_points)
            # Also create bbox from polygon
            bbox = polygon.to_bbox()
    elif shape_type == ShapeType.POLYLINE:
        # Polyline: similar to polygon but open
        if len(points) >= 4:  # At least 2 points
            polygon_points = [
                (int(points[i]), int(points[i + 1])) for i in range(0, len(points), 2)
            ]
            polygon = Polygon(points=polygon_points)
            bbox = polygon.to_bbox()
    elif shape_type == ShapeType.POINTS:
        # Points: create bbox from point cloud
        if len(points) >= 2:
            x_coords = [points[i] for i in range(0, len(points), 2)]
            y_coords = [points[i + 1] for i in range(0, len(points), 2)]
            if x_coords and y_coords:
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                bbox = BoundingBox(
                    x=int(x_min),
                    y=int(y_min),
                    width=int(x_max - x_min) if x_max > x_min else 1,
                    height=int(y_max - y_min) if y_max > y_min else 1,
                )
    elif shape_type == ShapeType.ELLIPSE:
        # Ellipse: points = [cx, cy, rx, ry] or [x1, y1, x2, y2]
        if len(points) >= 4:
            if len(points) == 4:
                # Assume center-radius format
                cx, cy, rx, ry = points[0], points[1], points[2], points[3]
                bbox = BoundingBox(
                    x=int(cx - rx),
                    y=int(cy - ry),
                    width=int(2 * rx),
                    height=int(2 * ry),
                )
            else:
                # Bounding box format
                x1, y1, x2, y2 = points[0], points[1], points[2], points[3]
                x_min = min(x1, x2)
                y_min = min(y1, y2)
                x_max = max(x1, x2)
                y_max = max(y1, y2)
                bbox = BoundingBox(
                    x=int(x_min),
                    y=int(y_min),
                    width=int(x_max - x_min),
                    height=int(y_max - y_min),
                )
    else:
        # For other types (cuboid, skeleton, mask), try to create bbox from points
        if len(points) >= 4:
            x_coords = [points[i] for i in range(0, len(points), 2) if i < len(points)]
            y_coords = [points[i + 1] for i in range(0, len(points), 2) if i + 1 < len(points)]
            if x_coords and y_coords:
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                bbox = BoundingBox(
                    x=int(x_min),
                    y=int(y_min),
                    width=int(x_max - x_min) if x_max > x_min else 1,
                    height=int(y_max - y_min) if y_max > y_min else 1,
                )

    # Ensure we have at least a bbox
    if bbox is None:
        # Fallback: create a minimal bbox
        bbox = BoundingBox(x=0, y=0, width=1, height=1)

    return Label(
        label=label_name,
        score=score,
        bbox=bbox,
        polygon=polygon,
        attributes=label_attributes,
    )
