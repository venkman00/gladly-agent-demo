"""Structured JSON logs plus New Relic custom events.

One wide event per processed ticket (intent, latency, tokens, escalation) is
the minimum that answers the questions that matter in production. The NRQL
queries are in the README.
"""

import json
import logging
import sys

try:
    import newrelic.agent as newrelic
except ImportError:  # optional: runs fine without the agent installed
    newrelic = None


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {"level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        entry.update(getattr(record, "fields", {}))
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=level, handlers=[handler], force=True)


def record_event(event_type: str, attributes: dict) -> None:
    logging.getLogger("events").info(event_type, extra={"fields": attributes})
    if newrelic:
        newrelic.record_custom_event(event_type, attributes, application=newrelic.application())
