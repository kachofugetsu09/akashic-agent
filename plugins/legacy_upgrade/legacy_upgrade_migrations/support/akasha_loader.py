"""Frozen Akasha index loader used by historical migrations."""

from .akasha.infrastructure.loader import load_turn_suffix, load_turns

__all__ = ["load_turn_suffix", "load_turns"]
