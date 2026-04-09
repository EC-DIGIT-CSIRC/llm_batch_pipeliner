"""Filter chain execution.

Applies a list of :class:`~llm_batch_pipeline.plugins.base.Filter` instances
to a list of :class:`~llm_batch_pipeline.plugins.base.ParsedFile` objects.
Files that fail any filter are excluded and the drop reason is recorded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from llm_batch_pipeline.logging_utils import log_event
from llm_batch_pipeline.metrics import timed
from llm_batch_pipeline.plugins.base import Filter, ParsedFile

logger = logging.getLogger("llm_batch_pipeline.filters")


@dataclass(slots=True)
class FilterResult:
    """Outcome of running a filter chain over a file set."""

    kept: list[ParsedFile] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)
    total_input: int = 0

    @property
    def kept_count(self) -> int:
        return len(self.kept)

    @property
    def dropped_count(self) -> int:
        return len(self.dropped)


def run_filter_chain(
    files: list[ParsedFile],
    filters: list[Filter],
    *,
    chain_name: str = "filter",
) -> FilterResult:
    """Apply *filters* sequentially to *files*.

    A file is dropped on the first filter that rejects it.  Drop reasons
    are recorded in :attr:`FilterResult.dropped`.
    """
    result = FilterResult(total_input=len(files))
    if not filters:
        result.kept = list(files)
        return result

    for pf in files:
        keep = True
        reason = ""
        rejecting_filter = ""
        for flt in filters:
            with timed() as t:
                keep, reason = flt.apply(pf)
            if not keep:
                rejecting_filter = flt.name
                log_event(
                    logger,
                    f"Dropped by {flt.name}: {reason}",
                    file_id=pf.filename,
                    step=chain_name,
                    filter=flt.name,
                    status="dropped",
                    reason=reason,
                    duration_ms=t["duration_ms"],
                )
                break

        if keep:
            result.kept.append(pf)
        else:
            result.dropped.append(
                {
                    "filename": pf.filename,
                    "filter": rejecting_filter,
                    "reason": reason,
                }
            )

    log_event(
        logger,
        f"{chain_name}: kept {result.kept_count}/{result.total_input}",
        step=chain_name,
        status="ok",
        kept=result.kept_count,
        dropped=result.dropped_count,
    )
    return result
