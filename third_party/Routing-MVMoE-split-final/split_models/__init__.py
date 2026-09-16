"""Strict XY-only giant-tour versions of the three multitask MVMoE models."""

from .xy_models import MVMoE4ELSplit, MVMoE4ESplit, POMOMTLSplit, get_split_model

__all__ = ["POMOMTLSplit", "MVMoE4ESplit", "MVMoE4ELSplit", "get_split_model"]
