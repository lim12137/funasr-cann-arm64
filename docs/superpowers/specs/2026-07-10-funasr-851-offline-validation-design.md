# FunASR 8.5.1 Offline Validation Image Design

## Goal

Build a parallel ARM64 validation image for an Ascend NPU server. It must validate offline Fun-ASR-Nano-2512, FSMN VAD, and CAM++ model directories on `npu:0` without downloading or embedding model weights.

## Scope

- Keep the existing `Dockerfile` and `8.0-rc1` image path unchanged.
- Add a separate `8.5.1-validate` image path based on `ascendai/cann:8.5.1-910b-ubuntu22.04-py3.11`.
- Pin the runtime to Python 3.11, `torch==2.9.0`, `torch-npu==2.9.0.post1`, and `funasr==1.3.1` to match the two upstream NPU validation projects.
- Bundle source revisions `2a2778765678ab5f15239b9dc181839e892d7006` from `gyccc/FunAudioLLM-Fun-ASR-Nano-2512-NPU` and `eba9c2ed4fed2adc838322376d8df6278b82c2bc` from `Ascend-SACT/FunASR_VAD-SenseVoiceSmall-CAMPPlus`.
- Provide runtime commands that verify real NPU availability, load each mounted offline model, and run a one-file transcription smoke test.
- Do not start an HTTP service or add business API model routing in this version.

## Runtime Contract

The NPU server mounts all model folders read-only under `/models`. The validation command accepts the following environment variables:

- `NANO_MODEL_DIR`: required local Fun-ASR-Nano-2512 folder.
- `VAD_MODEL_DIR`: required local FSMN VAD folder.
- `SPK_MODEL_DIR`: required local CAM++ folder.
- `TEST_AUDIO`: optional local 16 kHz audio file. Defaults to the Nano project's bundled `assets/test.wav`.
- `NPU_DEVICE`: optional device selector. Defaults to `npu:0`.

The validator must reject missing directories before importing a model. It uses only local paths with `disable_update=True`; it does not invoke ModelScope download APIs.

## Validation Behavior

`verify-npu-models` performs these ordered checks:

1. Imports `torch` and `torch_npu`, checks `torch.npu.is_available()`, selects `NPU_DEVICE`, and allocates a scalar tensor on the NPU.
2. Applies the Fun-ASR-Nano registration and audio-loader compatibility patch used by its upstream project, loads `NANO_MODEL_DIR` through `AutoModel`, and transcribes `TEST_AUDIO`.
3. Loads the Nano model again with `VAD_MODEL_DIR` and `SPK_MODEL_DIR` attached to `AutoModel`, then transcribes `TEST_AUDIO`.
4. Prints the resolved model paths, NPU metadata, elapsed time, and non-empty transcription text. Any failure returns a non-zero status.

The third check is deliberately an integration probe: it establishes whether the current FunASR and NPU runtime can attach FSMN VAD and CAM++ to Fun-ASR-Nano-2512. Future API design depends on that real hardware result.

## Build and Publishing

An independent `build-8.5.1-validate.yml` workflow runs on GitHub-hosted Ubuntu with QEMU and Buildx. It first runs `docker buildx imagetools inspect` for the candidate CANN base and requires a `linux/arm64` manifest. It then builds `linux/arm64`, runs import-only checks during the Docker build, and pushes matching tags to:

- `ghcr.io/lim12137/funasr-cann:8.5.1-validate`
- `crpi-fs24haezdztsodhc.cn-guangzhou.personal.cr.aliyuncs.com/hopemyl/funasr:8.5.1-validate`

The GitHub runner has no Ascend device. It verifies the base image, dependency installation, and Python imports only. The NPU server executes the runtime validator.

## Non-Goals

- No model weight downloads during image build or runtime.
- No SenseVoice-Small dependency or model path.
- No modification to the existing `8.0-rc1` Dockerfile or workflow.
- No claim that VAD/CAM++ composition is production-ready before the NPU server validator succeeds.
