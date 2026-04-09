"""Transform chain execution.

Applies a list of :class:`~llm_batch_pipeline.plugins.base.Transformer`
instances sequentially to each :class:`~llm_batch_pipeline.plugins.base.ParsedFile`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from llm_batch_pipeline.logging_utils import log_event
from llm_batch_pipeline.metrics import timed
from llm_batch_pipeline.plugins.base import ParsedFile, Transformer

logger = logging.getLogger("llm_batch_pipeline.transforms")


def run_transform_chain(
    files: list[ParsedFile],
    transformers: list[Transformer],
    *,
    chain_name: str = "transform",
    on_progress: Callable[[], None] | None = None,
) -> list[ParsedFile]:
    """Apply *transformers* sequentially to every file in *files*.

    Returns a new list of (possibly modified) :class:`ParsedFile` objects.
    Transformers that raise are logged and the file is passed through
    unchanged.
    """
    if not transformers:
        return list(files)

    results: list[ParsedFile] = []
    for pf in files:
        current = pf
        for tx in transformers:
            try:
                with timed() as t:
                    current = tx.apply(current)
                log_event(
                    logger,
                    f"Applied {tx.name}",
                    file_id=current.filename,
                    step=chain_name,
                    transformer=tx.name,
                    status="ok",
                    duration_ms=t["duration_ms"],
                )
            except Exception as exc:  # noqa: BLE001
                log_event(
                    logger,
                    f"{tx.name} failed: {exc}",
                    file_id=pf.filename,
                    step=chain_name,
                    transformer=tx.name,
                    status="error",
                    error=str(exc),
                    level=logging.WARNING,
                )
                # Keep the file in its pre-transform state
        results.append(current)

        if on_progress:
            on_progress()

    log_event(
        logger,
        f"{chain_name}: transformed {len(results)} files",
        step=chain_name,
        status="ok",
        count=len(results),
    )
    return results
