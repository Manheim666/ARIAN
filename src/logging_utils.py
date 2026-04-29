from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone


class _UTCFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat(timespec="seconds")


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("arian")
    logger.setLevel(level)

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(_UTCFormatter("%(asctime)sZ | %(levelname)s | %(message)s"))
        logger.addHandler(handler)

    logger.propagate = False
    return logger
