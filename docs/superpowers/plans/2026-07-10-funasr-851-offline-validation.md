# FunASR 8.5.1 Offline Validation Image Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a parallel ARM64 CANN 8.5.1 image that verifies mounted offline Fun-ASR-Nano-2512, FSMN VAD, and CAM++ models on an Ascend NPU.

**Architecture:** A dedicated Dockerfile uses the candidate CANN 8.5.1 Python 3.11 base, pins the NPU Python runtime, and clones the two upstream validation repositories at fixed revisions. A Python entrypoint loads the offline Nano model first, then probes Nano with offline VAD and CAM++ paths attached. A separate GitHub Actions workflow validates the base image's ARM64 manifest before publishing an independent `8.5.1-validate` tag to GHCR and ACR.

**Tech Stack:** Docker Buildx, GitHub Actions, CANN 8.5.1, Python 3.11, PyTorch 2.9.0, torch-npu 2.9.0.post1, FunASR 1.3.1, ModelScope.

## Global Constraints

- Keep `Dockerfile`, `.github/workflows/build.yml`, and tag `8.0-rc1` unchanged.
- Use `ascendai/cann:8.5.1-910b-ubuntu22.04-py3.11` only after the workflow's `linux/arm64` manifest preflight passes.
- Because GitHub-hosted x86 runners execute ARM64 code under QEMU, Docker build validation uses `pip check` and script compilation only; `torch` and `torch_npu` imports run exclusively in the native NPU-server validator.
- Do not copy, download, or bake model weights into the image.
- The runtime accepts only local `NANO_MODEL_DIR`, `VAD_MODEL_DIR`, and `SPK_MODEL_DIR` paths.
- Keep ModelScope and Hugging Face in offline mode at runtime.
- Use `disable_update=True` for every FunASR model load.
- Do not add SenseVoice-Small or start an HTTP API in this version.

---

### Task 1: Create the parallel ARM64 image definition

**Files:**
- Create: `Dockerfile.8.5.1-validate`
- Create: `requirements-8.5.1-validate.txt`

**Interfaces:**
- Consumes: Docker build argument `CANN_BASE`, defaulting to `ascendai/cann:8.5.1-910b-ubuntu22.04-py3.11`.
- Produces: an image with Python 3.11, `torch==2.9.0`, `torch-npu==2.9.0.post1`, `funasr==1.3.1`, and fixed upstream source trees at `/opt/src/funasr-nano` and `/opt/src/funasr-vad-campp`.

- [ ] **Step 1: Add the pinned Python dependency manifest**

Create `requirements-8.5.1-validate.txt`:

```text
funasr==1.3.1
modelscope
soundfile
scipy
librosa
editdistance
fastapi
uvicorn
python-multipart
```

- [ ] **Step 2: Add the Dockerfile with fixed upstream revisions**

Create `Dockerfile.8.5.1-validate` with:

```dockerfile
ARG CANN_BASE=ascendai/cann:8.5.1-910b-ubuntu22.04-py3.11
FROM ${CANN_BASE}

ARG NANO_REV=2a2778765678ab5f15239b9dc181839e892d7006
ARG VAD_CAMPP_REV=eba9c2ed4fed2adc838322376d8df6278b82c2bc

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        git \
        libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-8.5.1-validate.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
        torch==2.9.0 \
        torchaudio==2.9.0 \
        torch-npu==2.9.0.post1 && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

RUN git clone https://atomgit.com/gyccc/FunAudioLLM-Fun-ASR-Nano-2512-NPU.git /opt/src/funasr-nano && \
    git -C /opt/src/funasr-nano checkout "$NANO_REV" && \
    git clone https://atomgit.com/Ascend-SACT/FunASR_VAD-SenseVoiceSmall-CAMPPlus.git /opt/src/funasr-vad-campp && \
    git -C /opt/src/funasr-vad-campp checkout "$VAD_CAMPP_REV"

RUN python3 -c "import funasr, modelscope, torch, torch_npu; print(torch.__version__)"
```

Then add the verification scripts and runtime environment in Task 2.

- [ ] **Step 3: Verify the Dockerfile's static shape**

Run:

```bash
git diff --check -- Dockerfile.8.5.1-validate requirements-8.5.1-validate.txt
```

Expected: no output and exit code 0.

- [ ] **Step 4: Commit the image definition**

```bash
git add Dockerfile.8.5.1-validate requirements-8.5.1-validate.txt
git commit -m "Add CANN 8.5.1 FunASR validation image"
```

### Task 2: Add offline NPU model validator

**Files:**
- Create: `verify/verify_npu_models.py`
- Create: `verify/entrypoint.sh`
- Modify: `Dockerfile.8.5.1-validate`

**Interfaces:**
- Consumes: `NANO_MODEL_DIR`, `VAD_MODEL_DIR`, `SPK_MODEL_DIR`, optional `TEST_AUDIO`, and optional `NPU_DEVICE` environment variables.
- Produces: exit code 0 only when NPU allocation, Nano-only transcription, and Nano plus VAD/CAM++ transcription each return a non-empty result.

- [ ] **Step 1: Write static validation checks**

Create a test command that compiles the runtime validator without executing NPU code:

```bash
python3 -m py_compile verify/verify_npu_models.py
```

Expected before implementation: non-zero exit because the file does not exist.

- [ ] **Step 2: Implement `verify/verify_npu_models.py`**

Implement a Python script that:

```python
import os
from pathlib import Path

NANO_MODEL_DIR = Path(os.environ["NANO_MODEL_DIR"])
VAD_MODEL_DIR = Path(os.environ["VAD_MODEL_DIR"])
SPK_MODEL_DIR = Path(os.environ["SPK_MODEL_DIR"])
NPU_DEVICE = os.environ.get("NPU_DEVICE", "npu:0")
TEST_AUDIO = Path(os.environ.get("TEST_AUDIO", "/opt/src/funasr-nano/assets/test.wav"))
```

It must reject missing paths; import `torch` and `torch_npu`; require `torch.npu.is_available()`; allocate `torch.tensor([1], device=NPU_DEVICE)`; apply the upstream Nano registration patch; use `AutoModel(model=str(NANO_MODEL_DIR), device=NPU_DEVICE, disable_update=True, trust_remote_code=True)` for Nano-only inference; then use the same call with `vad_model=str(VAD_MODEL_DIR)` and `spk_model=str(SPK_MODEL_DIR)` for the combined integration probe. Each `generate()` call receives `input=[str(TEST_AUDIO)]`, `cache={}`, `batch_size=1`, `language="中文"`, and `itn=True`, and must produce non-empty `result[0]["text"]`.

- [ ] **Step 3: Implement the entrypoint**

Create `verify/entrypoint.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
exec python3 /opt/verify/verify_npu_models.py "$@"
```

- [ ] **Step 4: Wire the scripts into the Dockerfile**

Append the following to `Dockerfile.8.5.1-validate`:

```dockerfile
COPY verify/verify_npu_models.py /opt/verify/verify_npu_models.py
COPY verify/entrypoint.sh /usr/local/bin/verify-npu-models

ENV ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
ENV LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}
ENV PYTHONPATH=/opt/src/funasr-nano:/opt/src/funasr-vad-campp:${PYTHONPATH}
ENV HF_HUB_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV MODELSCOPE_OFFLINE=1

RUN chmod 0755 /usr/local/bin/verify-npu-models && \
    python3 -m py_compile /opt/verify/verify_npu_models.py

WORKDIR /workspace
ENTRYPOINT ["verify-npu-models"]
```

- [ ] **Step 5: Run static validator verification**

Run:

```bash
python3 -m py_compile verify/verify_npu_models.py
git diff --check
```

Expected: both commands exit code 0.

- [ ] **Step 6: Commit the validator**

```bash
git add Dockerfile.8.5.1-validate verify/verify_npu_models.py verify/entrypoint.sh
git commit -m "Add offline NPU model validation entrypoint"
```

### Task 3: Publish the parallel validation image through Actions

**Files:**
- Create: `.github/workflows/build-8.5.1-validate.yml`

**Interfaces:**
- Consumes: `ALIYUN_ACR_PWD` repository secret and workflow-dispatch `tag`, default `8.5.1-validate`.
- Produces: matching ARM64 tags at GHCR and ACR only after CANN base `linux/arm64` manifest preflight and successful Buildx build.

- [ ] **Step 1: Add the base-image preflight workflow**

Create a workflow that checks out code, installs QEMU and Buildx, then runs:

```yaml
- name: Verify CANN base image supports ARM64
  run: |
    docker buildx imagetools inspect \
      ascendai/cann:8.5.1-910b-ubuntu22.04-py3.11 \
      --format '{{json .Manifest}}' \
      | grep -q '"architecture":"arm64"'
```

- [ ] **Step 2: Add registry logins and Buildx publish**

Use the established ACR login username `hopemyl` and `secrets.ALIYUN_ACR_PWD`. Configure Buildx with `file: ./Dockerfile.8.5.1-validate`, `platforms: linux/arm64`, and tags:

```text
ghcr.io/${{ github.repository_owner }}/funasr-cann:${{ github.event.inputs.tag || '8.5.1-validate' }}
crpi-fs24haezdztsodhc.cn-guangzhou.personal.cr.aliyuncs.com/hopemyl/funasr:${{ github.event.inputs.tag || '8.5.1-validate' }}
```

- [ ] **Step 3: Verify workflow syntax and changed files**

Run:

```bash
git diff --check
git diff -- .github/workflows/build-8.5.1-validate.yml
```

Expected: no whitespace errors; the workflow must use `Dockerfile.8.5.1-validate` and preserve the existing workflow unchanged.

- [ ] **Step 4: Commit and push**

```bash
git add .github/workflows/build-8.5.1-validate.yml
git commit -m "Build and publish CANN 8.5.1 validation image"
git push origin main
```

- [ ] **Step 5: Verify GitHub Actions output**

Run:

```bash
gh run list --repo lim12137/funasr-cann-arm64 --workflow build-8.5.1-validate.yml --limit 1
gh run view "$(gh run list --repo lim12137/funasr-cann-arm64 --workflow build-8.5.1-validate.yml --limit 1 --json databaseId --jq '.[0].databaseId')" --repo lim12137/funasr-cann-arm64 --log-failed
```

Expected: base manifest preflight and Build and push steps complete successfully. If the base tag does not exist or lacks ARM64, report that exact preflight failure without modifying the `8.0-rc1` workflow.

### Task 4: Validate mounted models on the NPU server

**Files:**
- No repository changes.

**Interfaces:**
- Consumes: the ACR `8.5.1-validate` image and read-only offline model mount.
- Produces: validator output containing `NPU_AVAILABLE`, `NANO_ONLY_TEXT`, and `NANO_VAD_SPK_TEXT` with exit code 0.

- [ ] **Step 1: Pull the validation image**

```bash
docker pull crpi-fs24haezdztsodhc.cn-guangzhou.personal.cr.aliyuncs.com/hopemyl/funasr:8.5.1-validate
```

- [ ] **Step 2: Run NPU validation with offline models**

```bash
docker run --rm \
  --device /dev/davinci0 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /data/models:/models:ro \
  -e NANO_MODEL_DIR=/models/Fun-ASR-Nano-2512 \
  -e VAD_MODEL_DIR=/models/speech_fsmn_vad_zh-cn-16k-common-pytorch \
  -e SPK_MODEL_DIR=/models/speech_campplus_sv_zh-cn_16k-common \
  crpi-fs24haezdztsodhc.cn-guangzhou.personal.cr.aliyuncs.com/hopemyl/funasr:8.5.1-validate
```

Expected: exit code 0 after NPU allocation, Nano-only transcription, and Nano plus VAD/CAM++ integration probe.

- [ ] **Step 3: Record result before business API work**

Use the returned integration result to choose the next design: an API chain using Nano plus VAD/CAM++ only if the combined probe passes; otherwise expose Nano separately and use VAD/CAM++ according to a validated composition.
