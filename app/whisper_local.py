import os
from .pipeline import run

_model = None


def transcribe(video, wd):
    global _model
    from faster_whisper import WhisperModel
    audio = wd / "audio.mp3"
    run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k", str(audio)])
    if _model is None:
        try:
            import ctranslate2
            gpu = ctranslate2.get_cuda_device_count() > 0
        except Exception:
            gpu = False
        size = os.getenv("WHISPER_MODEL", "large-v3" if gpu else "small")
        _model = WhisperModel(size, device="cuda" if gpu else "cpu",
                              compute_type="float16" if gpu else "int8")
    segs, _ = _model.transcribe(str(audio), vad_filter=True)
    out = [(s.start, s.end, s.text.strip()) for s in segs if s.text.strip()]
    if not out:
        raise RuntimeError("စကားပြောမတွေ့ပါ (no speech found)")
    return out
