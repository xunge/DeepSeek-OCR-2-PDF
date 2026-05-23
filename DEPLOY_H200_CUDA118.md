# H200 CUDA 11.8 Docker Deployment

This project is deployed with the official DeepSeek-OCR-2 dependency stack:

```text
CUDA 11.8 + Python 3.10 + torch 2.6.0 + vLLM 0.8.5 + flash-attn 2.7.3
```

The H200 server can keep its newer host driver/CUDA display version. The
backend container uses `nvidia/cuda:11.8.0-base-ubuntu22.04`.

## Prerequisites

- NVIDIA H200 is visible in `nvidia-smi`
- Docker Engine and Docker Compose are installed
- NVIDIA Container Toolkit is installed
- The DeepSeek-OCR-2 model is downloaded into the project-local model directory

Verify GPU access from a CUDA 11.8 container:

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## Model Path And Download

The compose file expects the model at:

```text
./models/DeepSeek-OCR-2
```

Download or resume the model weights:

```bash
python3 scripts/download_model.py
```

Inside the backend container, the same directory is mounted read-only at:

```text
/app/models/DeepSeek-OCR-2
```

If your model is stored elsewhere, update the volume and `MODEL_PATH` in
`docker-compose.yml`.

## Build And Run

```bash
docker compose build
docker compose up -d
```

Backend health check:

```bash
curl http://localhost:5000/api/health
```

Frontend:

```text
http://SERVER_IP/
```

## GPU And Concurrency Tuning

Edit `docker-compose.yml`:

```yaml
environment:
  CUDA_VISIBLE_DEVICES: "0"
  MAX_CONCURRENCY: "16"
  NUM_WORKERS: "16"
```

For a single H200, start conservatively with `MAX_CONCURRENCY=16`. Increase it
after checking GPU memory and latency under real PDF workloads.

## Notes

The Dockerfile starts from the CUDA 11.8 `base` image and installs the minimal
CUDA development packages needed by `flash-attn` and vLLM/Triton:

```text
cuda-nvcc-11-8
cuda-cudart-dev-11-8
```

This keeps the required base image while still providing `ptxas` at
`/usr/local/cuda-11.8/bin/ptxas`.

The image uses Ubuntu 22.04's system Python 3.10 instead of conda, so the build
does not need to download Miniconda or Miniforge from external release hosts.

If the server cannot access GitHub, put the vLLM wheel in the project-local
`wheels/` directory before building:

```text
wheels/vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
```

Also put the flash-attn wheel there:

```text
wheels/flash_attn-2.7.3+cu11torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

The Dockerfile installs these local wheels when present. If a wheel is absent,
it falls back to the official remote package URL/PyPI.
