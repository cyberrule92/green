# LoRA / QLoRA trainer — the image behind MODEL_FINETUNE_IMAGE.
#
# Built FROM the vLLM image for the same reason as the quantizer: its torch is
# the build proven to open a CUDA context on this host's vGPU driver, and it is
# already on disk. peft/trl/datasets resolve on top without touching torch, and
# bitsandbytes (0.49) and accelerate ship in the base already — 4-bit quantize
# was verified working on this GPU before pinning QLoRA to it.
#
# The API image cannot do this job. Its torch targets CUDA 13 against a 12.8
# driver and fails to initialise CUDA at all, which is why training was carved
# out into a container in the first place.
#
# Build (context must be the repo root):
#   docker build -f docker/finetune/Dockerfile -t green-finetune:latest .

FROM vllm/vllm-openai:latest

RUN pip install --no-cache-dir \
      peft==0.20.0 \
      trl==1.9.2 \
      datasets==5.0.1 \
      kernels==0.11.1

COPY scripts/finetune_train.py /opt/green/finetune_train.py

# Cleared for the same reason as the quantizer image: the base ENTRYPOINT starts
# the vLLM API server, and FineTuningService launches this with
# Cmd=["sh","-lc", <MODEL_FINETUNE_COMMAND>] and no entrypoint override. Left in
# place, every training job would quietly start a web server and time out.
ENTRYPOINT []
CMD ["python3", "/opt/green/finetune_train.py", "--help"]
