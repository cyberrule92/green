# AWQ calibration runner — the image behind MODEL_QUANT_IMAGE.
#
# Built FROM the vLLM image on purpose. That image is already pulled on this host
# (32.9 GB), and its torch is the one proven to initialise CUDA against this
# 12.8-era vGPU driver. Installing a second torch to get a calibration toolchain
# would cost several GB of disk and a real risk of landing a build that cannot
# open a CUDA context — which on a licence-restricted vGPU is a failure you only
# discover at run time. autoawq resolves cleanly on top: it pulls datasets and
# pyarrow and leaves torch alone.
#
# Build (context must be the repo root):
#   docker build -f docker/quantize/Dockerfile -t green-quantize:latest .
#
# The pinned version matters. autoawq's quantize() keyword arguments have moved
# between releases, and scripts/quantize_awq.py is written against 0.2.9.

FROM vllm/vllm-openai:latest

RUN pip install --no-cache-dir autoawq==0.2.9

COPY scripts/quantize_awq.py /opt/green/quantize_awq.py

# The base image's ENTRYPOINT starts the vLLM API server, which would swallow
# our command as arguments to itself. LocalQuantizer launches this container with
# Cmd=["sh","-lc", <MODEL_QUANT_COMMAND>] and sets no entrypoint, so it has to be
# cleared here or every quantization run silently starts a web server instead.
ENTRYPOINT []
CMD ["python3", "/opt/green/quantize_awq.py", "--help"]
