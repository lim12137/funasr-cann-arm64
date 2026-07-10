import importlib
import os
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_AUDIO = Path("/opt/src/funasr-nano/assets/test.wav")


def required_directory(name: str, value: str) -> Path:
    path = Path(value)
    if not path.is_dir():
        raise RuntimeError(f"{name} must be an existing directory: {path}")
    return path


def required_file(name: str, value: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise RuntimeError(f"{name} must be an existing file: {path}")
    return path


def transcription_text(result: list[dict[str, Any]]) -> str:
    if result:
        text = str(result[0].get("text", "")).strip()
        if text:
            return text
    raise RuntimeError("FunASR returned an empty transcription")


def configure_nano_runtime() -> None:
    import torchaudio

    original_load = torchaudio.load

    def load_with_soundfile_fallback(filepath: str | Path, **kwargs: Any) -> Any:
        try:
            return original_load(filepath, **kwargs)
        except Exception:
            import soundfile as soundfile
            import torch

            data, sample_rate = soundfile.read(filepath, dtype="float32")
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            return torch.from_numpy(data).unsqueeze(0), sample_rate

    torchaudio.load = load_with_soundfile_fallback

    funasr_dir = Path(importlib.import_module("funasr").__file__).parent
    nano_module_dir = funasr_dir / "models" / "fun_asr_nano"
    if str(nano_module_dir) not in sys.path:
        sys.path.insert(0, str(nano_module_dir))
    importlib.import_module("funasr.models.fun_asr_nano.model")


def load_model(
    nano_model_dir: Path,
    device: str,
    vad_model_dir: Path | None = None,
    spk_model_dir: Path | None = None,
) -> Any:
    configure_nano_runtime()
    from funasr import AutoModel

    kwargs: dict[str, Any] = {
        "model": str(nano_model_dir),
        "device": device,
        "disable_update": True,
        "trust_remote_code": True,
    }
    if vad_model_dir is not None:
        kwargs["vad_model"] = str(vad_model_dir)
    if spk_model_dir is not None:
        kwargs["spk_model"] = str(spk_model_dir)
    return AutoModel(**kwargs)


def transcribe(label: str, model: Any, audio_path: Path) -> str:
    started_at = time.monotonic()
    result = model.generate(
        input=[str(audio_path)],
        cache={},
        batch_size=1,
        language="中文",
        itn=True,
    )
    text = transcription_text(result)
    elapsed_seconds = time.monotonic() - started_at
    print(f"{label}_TEXT={text}")
    print(f"{label}_SECONDS={elapsed_seconds:.3f}")
    return text


def main() -> None:
    nano_model_dir = required_directory("NANO_MODEL_DIR", os.environ.get("NANO_MODEL_DIR", ""))
    vad_model_dir = required_directory("VAD_MODEL_DIR", os.environ.get("VAD_MODEL_DIR", ""))
    spk_model_dir = required_directory("SPK_MODEL_DIR", os.environ.get("SPK_MODEL_DIR", ""))
    audio_path = required_file("TEST_AUDIO", os.environ.get("TEST_AUDIO", str(DEFAULT_AUDIO)))
    device = os.environ.get("NPU_DEVICE", "npu:0")

    import torch
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError("NPU is unavailable; verify device and Ascend driver mounts")
    device_count = torch.npu.device_count()
    torch.tensor([1], device=device).cpu()

    print(f"NPU_AVAILABLE=true")
    print(f"NPU_DEVICE={device}")
    print(f"NPU_DEVICE_COUNT={device_count}")
    print(f"NANO_MODEL_DIR={nano_model_dir}")
    print(f"VAD_MODEL_DIR={vad_model_dir}")
    print(f"SPK_MODEL_DIR={spk_model_dir}")
    print(f"TEST_AUDIO={audio_path}")

    nano_model = load_model(nano_model_dir, device)
    transcribe("NANO_ONLY", nano_model, audio_path)
    del nano_model

    combined_model = load_model(nano_model_dir, device, vad_model_dir, spk_model_dir)
    transcribe("NANO_VAD_SPK", combined_model, audio_path)


if __name__ == "__main__":
    main()
