FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends bash bc procps && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /app/data && \
    chown -R appuser:appuser /app

COPY advanced_rag.py budgets.py coding_agent.py conversation_store.py \
     csrd_reporting.py \
     decision_engine.py deferred_queue.py host_metrics_service.py \
     model_zoo.py model_zoo_updater.py monitoring_layer.py multimodal.py \
     nemo_guardrails.py quality_latency_estimator.py \
     rl_controller.py routing_policies.py semantic_cache.py system_metrics.sh \
     tenancy.py tracing.py workflows.py workflow_templates.py ./
COPY config ./config

RUN sed -i 's/\r$//' /app/system_metrics.sh && \
    chmod +x /app/system_metrics.sh && \
    chown -R appuser:appuser /app

EXPOSE 8100 9000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8100/health', timeout=3).read()"

USER appuser

CMD ["uvicorn", "decision_engine:app", "--host", "0.0.0.0", "--port", "8100"]
