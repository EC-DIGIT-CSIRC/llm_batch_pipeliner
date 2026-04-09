"""Plugin discovery and registration.

Plugins are registered by name.  Built-in example plugins are auto-discovered
from :mod:`llm_batch_pipeline.examples`.  External plugins can be registered
programmatically via :func:`register_plugin`.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_batch_pipeline.plugins.base import (
        FileReader,
        Filter,
        OutputTransformer,
        Transformer,
    )

logger = logging.getLogger("llm_batch_pipeline.registry")


@dataclass(slots=True)
class PluginSpec:
    """Fully describes a plugin's capabilities."""

    name: str
    reader: FileReader
    pre_filters: list[Filter] = field(default_factory=list)
    transformers: list[Transformer] = field(default_factory=list)
    post_filters: list[Filter] = field(default_factory=list)
    output_transformer: OutputTransformer | None = None
    default_prompt: str | None = None
    default_schema_module: str | None = None  # dotted module path to schema .py


# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, PluginSpec] = {}


def register_plugin(spec: PluginSpec) -> None:
    """Register a plugin spec by name.  Overwrites existing entries."""
    _REGISTRY[spec.name] = spec
    logger.debug("Registered plugin: %s", spec.name)


def get_plugin(name: str) -> PluginSpec:
    """Look up a registered plugin.  Raises :class:`KeyError` if not found."""
    _auto_discover()
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys())) or "(none)"
        msg = f"Unknown plugin {name!r}.  Available: {available}"
        raise KeyError(msg)
    return _REGISTRY[name]


def list_plugins() -> list[str]:
    """Return sorted names of all registered plugins."""
    _auto_discover()
    return sorted(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Auto-discovery of built-in example plugins
# ---------------------------------------------------------------------------

_DISCOVERED = False

_BUILTIN_MODULES = (
    "llm_batch_pipeline.examples.spam_detection.plugin",
    "llm_batch_pipeline.examples.gdpr_detection.plugin",
)


def _auto_discover() -> None:
    global _DISCOVERED  # noqa: PLW0603
    if _DISCOVERED:
        return
    _DISCOVERED = True
    for mod_path in _BUILTIN_MODULES:
        try:
            mod = importlib.import_module(mod_path)
            if hasattr(mod, "register"):
                mod.register()
        except Exception:  # noqa: BLE001
            logger.debug("Could not auto-discover plugin %s", mod_path, exc_info=True)
