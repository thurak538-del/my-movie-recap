"""Script-first movie recap: transcript -> Burmese recap script -> TTS -> cut video to fit voice."""
import asyncio, json, subprocess
import edge_tts, requests

GAP = 0.2  # seconds of silence after each narration line


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-800:])
    return p.stdout


def dur(path):
    return float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                      "-of", "csv=p=0", str(path)]).strip())


def transcribe(video, wd, key):
    audio = wd / "audio.mp3"
    run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k", str(audio)])
    with open(audio, "rb") as f:
        r = requests.post("https://api.groq.com/openai/v1/audio/transcriptions",
                          headers={"Authorization": f"Bearer {key}"}, files={"file": ("audio.mp3", f)},
                          data={"model": "whisper-large-v3", "response_format": "verbose_json",
                                "timestamp_granularities[]": "segment"}, timeout=900)
    r.raise_for_status()
    segs = [(s["start"], s["end"], s["text"].strip()) for s in r.json().get("segments", [])]
    if not segs:
        raise RuntimeError("စကားပြောမတွေ့ပါ (no speech found)")
    return segs


PROMPT = """You are a Burmese (Myanmar Unicode) movie-recap narrator in the style of popular YouTube movie recap channels.
Below is the timed transcript of a video ({total:.0f} seconds long). Retell the whole story in spoken Burmese:
- Natural spoken recap voice (hook at the start, flowing storytelling), NOT bookish or word-for-word translation.
- Do not invent plot events that are not supported by the transcript.
- About {n} lines. Each line = 1-2 SHORT sentences (max ~25 words), easy to read aloud.
- For each line give "start" and "end" (seconds): the scene in the video that matches that line.
  Keep lines in chronological order, each span 3-20 seconds, within 0-{total:.0f}.
Return ONLY JSON: {{"lines":[{{"text":"...","start":0.0,"end":8.0}}]}}

TRANSCRIPT:
{tr}"""


def write_script(segs, total, minutes, key, model):
    tr = "\n".join(f"[{a:.0f}-{b:.0f}] {t}" for a, b, t in segs)
    prompt = PROMPT.format(total=total, n=max(5, int(minutes * 60 / 8)), tr=tr)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    last = None
    for _ in range(2):
        r = requests.post(url, params={"key": key}, timeout=300, json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.7}})
        r.raise_for_status()
        try:
            raw = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            out = []
            for l in json.loads(raw)["lines"]:
                s = min(max(float(l["start"]), 0), total - 1)
                e = min(max(float(l["end"]), s + 3), total)
                if l["text"].strip():
                    out.append({"text": l["text"].strip(), "start": s, "end": e})
            if out:
                return out
        except Exception as ex:
            last = ex
    raise RuntimeError(f"Gemini script error: {last}")


def tts(text, voice, rate, out):
    for _ in range(3):
        try:
            asyncio.run(edge_tts.Communicate(text, voice, rate=rate).save(str(out)))
            return
        except Exception:
            pass
    raise RuntimeError("Edge TTS failed")


def ts(t):
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    return f"{h:02}:{m:02}:{s:06.3f}".replace(".", ",")


def build(video, lines, voice, rate, wd, progress):
    vl, al, srt, t = [], [], [], 0.0
    for i, l in enumerate(lines):
        progress(35 + int(55 * i / len(lines)), f"ဖန်တီးနေသည် {i + 1}/{len(lines)}")
        mp3, clip, wav = wd / f"n{i}.mp3", wd / f"c{i}.mp4", wd / f"a{i}.wav"
        tts(l["text"], voice, rate, mp3)
        d = dur(mp3)
        T = d + GAP
        L = l["end"] - l["start"]
        speed = min(1.25, max(0.8, L / d))   # video speed limited to 0.8x-1.25x
        src = min(L, d * speed)
        run(["ffmpeg", "-y", "-ss", f"{l['start']:.3f}", "-t", f"{src:.3f}", "-i", str(video),
             "-vf", f"setpts=PTS/{speed:.4f},tpad=stop_mode=clone:stop_duration={T:.2f},"
                    "scale=1280:720:force_original_aspect_ratio=decrease,"
                    "pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=30,format=yuv420p",
             "-t", f"{T:.3f}", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", str(clip)])
        run(["ffmpeg", "-y", "-i", str(mp3), "-af", f"apad=whole_dur={T:.3f}",
             "-ar", "44100", "-ac", "1", str(wav)])
        vl.append(f"file '{clip.resolve()}'")
        al.append(f"file '{wav.resolve()}'")
        srt.append(f"{i + 1}\n{ts(t)} --> {ts(t + d)}\n{l['text']}\n")
        t += T
    (wd / "v.txt").write_text("\n".join(vl))
    (wd / "a.txt").write_text("\n".join(al))
    (wd / "subtitles.srt").write_text("\n".join(srt), encoding="utf-8")
    progress(92, "ပေါင်းစပ်နေသည်")
    # original audio is dropped; narration loudness normalised once over the whole track (no ducking)
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(wd / "v.txt"),
         "-f", "concat", "-safe", "0", "-i", str(wd / "a.txt"), "-map", "0:v", "-map", "1:a",
         "-c:v", "copy", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-c:a", "aac", "-b:a", "192k",
         "-shortest", str(wd / "final.mp4")])
