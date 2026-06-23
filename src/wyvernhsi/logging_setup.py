"""Logging configuration for pipeline runs."""
from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_dir: Path | None = None, level: int = logging.INFO) -> None:
    """Configure root logging to console, plus a file under log_dir if given."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )