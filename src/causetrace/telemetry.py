"""OpenTelemetry tracing for the API, the agent, retrieval and LLM calls.

Modules create spans through ``tracer``; until ``setup_tracing`` installs a provider they
are no-ops, so evaluations and tests pay nothing. ``CAUSETRACE_TRACING`` picks the
exporter: ``console`` (default), ``otlp`` (``OTEL_EXPORTER_OTLP_ENDPOINT``, for example
the Jaeger service of the ``tracing`` compose profile) or ``off``.
"""

import os

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)

tracer = trace.get_tracer("causetrace")


def setup_tracing(
    service_name: str = "causetrace-api", exporter: SpanExporter | None = None
) -> TracerProvider | None:
    """Install the global tracer provider once; return it (None when tracing is off).

    ``exporter`` overrides the environment (tests pass an in-memory exporter).
    """
    current = trace.get_tracer_provider()
    if isinstance(current, TracerProvider):
        return current  # already installed; OpenTelemetry allows one global provider
    mode = (os.environ.get("CAUSETRACE_TRACING") or "console").lower()
    if exporter is None and mode == "off":
        return None
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    elif mode == "otlp":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    else:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    return provider
