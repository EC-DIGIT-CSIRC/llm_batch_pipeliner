"""Pydantic schema loading, strict JSON Schema enforcement, and field inference.

Loads a Python file containing a ``mySchema`` class that inherits from
:class:`pydantic.BaseModel`.  Converts it to a JSON Schema suitable for
OpenAI's structured-output mode (``strict: true``).

Also provides heuristic functions to auto-detect which schema fields hold the
classification label and confidence score.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Schema loading
# ---------------------------------------------------------------------------


def load_schema_class(schema_path: Path) -> type:
    """Dynamically import *schema_path* and return the ``mySchema`` class.

    Raises :class:`ValueError` if the file does not define ``mySchema``.
    """
    module_name = f"_llm_batch_schema_{schema_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, schema_path)
    if spec is None or spec.loader is None:
        msg = f"Cannot load schema from {schema_path}"
        raise ValueError(msg)

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    schema_cls = getattr(module, "mySchema", None)
    if schema_cls is None:
        msg = f"Schema file {schema_path} must define a class named 'mySchema'"
        raise ValueError(msg)

    return schema_cls


def load_schema_format(schema_path: Path) -> dict[str, Any]:
    """Load a schema file and return the OpenAI ``text.format`` dict.

    Result structure::

        {
            "format": {
                "type": "json_schema",
                "name": "<class_name>",
                "schema": { ... },
                "strict": True,
            }
        }
    """
    cls = load_schema_class(schema_path)
    raw_schema = cls.model_json_schema()
    _ensure_strict_json_schema(raw_schema)

    return {
        "format": {
            "type": "json_schema",
            "name": cls.__name__,
            "schema": raw_schema,
            "strict": True,
        }
    }


# ---------------------------------------------------------------------------
# Strict JSON Schema enforcement
# ---------------------------------------------------------------------------


def _ensure_strict_json_schema(schema: dict[str, Any]) -> None:
    """Mutate *schema* in place to satisfy OpenAI strict mode.

    * Sets ``additionalProperties: false`` on every object.
    * Makes all properties ``required``.
    * Recurses into ``$defs``, ``properties``, ``items``, etc.
    """
    if not isinstance(schema, dict):
        return

    if schema.get("type") == "object" or "properties" in schema:
        schema["additionalProperties"] = False
        props = schema.get("properties", {})
        if props:
            schema["required"] = list(props.keys())
        for prop_schema in props.values():
            _ensure_strict_json_schema(prop_schema)

    for key in ("items", "additionalProperties"):
        if isinstance(schema.get(key), dict):
            _ensure_strict_json_schema(schema[key])

    for ref_schema in schema.get("$defs", {}).values():
        _ensure_strict_json_schema(ref_schema)

    # anyOf / oneOf / allOf
    for combo_key in ("anyOf", "oneOf", "allOf"):
        for variant in schema.get(combo_key, []):
            _ensure_strict_json_schema(variant)


# ---------------------------------------------------------------------------
# Schema field inference
# ---------------------------------------------------------------------------

_LABEL_NAMES = ("label", "classification", "category", "class_", "verdict", "prediction", "result")
_CONFIDENCE_NAMES = ("confidence", "certainty", "score", "probability", "spam_score")


def infer_label_field(schema_path: Path) -> str | None:
    """Auto-detect the label field name from a schema file.

    Strategy:
      1. Look for string fields with an ``enum`` (Literal types).
      2. Fall back to well-known field names.
    """
    cls = load_schema_class(schema_path)
    json_schema = cls.model_json_schema()
    props = json_schema.get("properties", {})

    # Tier 1: enum-bearing string fields
    for name, prop in props.items():
        if prop.get("type") == "string" and "enum" in prop:
            return name

    # Tier 2: well-known names
    for candidate in _LABEL_NAMES:
        if candidate in props:
            return candidate

    return None


def infer_confidence_field(schema_path: Path) -> str | None:
    """Auto-detect the confidence/score field from a schema file.

    Looks for numeric fields matching well-known names.
    """
    cls = load_schema_class(schema_path)
    json_schema = cls.model_json_schema()
    props = json_schema.get("properties", {})

    for candidate in _CONFIDENCE_NAMES:
        if candidate in props:
            prop_type = props[candidate].get("type", "")
            if prop_type in ("number", "integer"):
                return candidate

    return None
