"""Models package — unified DETR decoder for multi-task biometry.

Upstream: data/types.py (type contracts), data/task_registry.py (task definitions).
Downstream: train/ (FUBioModule wraps FUBioModel with matching + loss).
"""

from fubio.models.backbone import (
    BackboneOutput,
    ConvNeXtV2Backbone,
    DINOv2Backbone,
    build_backbone,
)
from fubio.models.model import FUBioModel

__all__ = [
    "BackboneOutput",
    "ConvNeXtV2Backbone",
    "DINOv2Backbone",
    "FUBioModel",
    "build_backbone",
]
