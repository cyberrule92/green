"""OpenTelemetry tracing + logging setup.

Exports spans and log records over OTLP/gRPC to the collector at
``OTEL_EXPORTER_OTLP_ENDPOINT`` (default: ``host.docker.internal:9090``).
Call ``setup_tracing(app)`` once after the FastAPI app is constructed.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_initialised = False


def setup_tracing(app) -> None:
    global _initialised
    if _initialised:
        return

    try:
        from opentelemetry import trace
        from opentelemetry._logs import set_logger_provider
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
            OTLPLogExporter,
        )
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.logging import LoggingInstrumentor
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        logger.warning("OpenTelemetry packages unavailable, tracing disabled: %s", exc)
        return

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "host.docker.internal:9090")
    service_name = os.getenv("OTEL_SERVICE_NAME", "adaptive-green-ai")
    insecure = os.getenv("OTEL_EXPORTER_OTLP_INSECURE", "true").strip().lower() in {
        "1", "true", "yes", "on",
    }

    resource = Resource.create({"service.name": service_name})

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=insecure))
    )
    trace.set_tracer_provider(tracer_provider)

    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(
        BatchLogRecordProcessor(OTLPLogExporter(endpoint=endpoint, insecure=insecure))
    )
    set_logger_provider(logger_provider)

    otlp_log_handler = LoggingHandler(level=logging.NOTSET, logger_provider=logger_provider)
    logging.getLogger().addHandler(otlp_log_handler)

    LoggingInstrumentor().instrument(set_logging_format=False)
    FastAPIInstrumentor.instrument_app(app)

    logger.info(
        "OpenTelemetry → %s (service=%s, traces+logs over gRPC)",
        endpoint, service_name,
    )
    _initialised = True
