import json, os, shutil, threading, uuid
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .pipeline import build, dur, transcribe, write_script

DATA = Path(os.getenv("DATA_DIR", "data"))
MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
JOBS = {}
app = FastAPI(title="My Movie Recap")


def worker(jid, video, groq, gem, voice, rate, minutes):
    j, wd = JOBS[jid], DATA / jid
    st = lambda p, m: j.update(progress=p, message=m)
    try:
        total = dur(video)
        st(5, "စာထုတ်နေသည် (Whisper)")
        segs = transcribe(video, wd, groq)
        st(20, "Recap script ရေးနေသည် (Gemini)")
        lines = write_script(segs, total, minutes, gem, MODEL)
        (wd / "script.json").write_text(json.dumps(lines, ensure_ascii=False, indent=2), encoding="utf-8")
        build(video, lines, voice, rate, wd, st)
        j.update(status="done", progress=100, message="ပြီးပါပြီ")
    except Exception as e:
        j.update(status="error", message=str(e)[:500])


@app.post("/api/jobs")
def create(file: UploadFile = File(...), groq_key: str = Form(""), gemini_key: str = Form(""),
           voice: str = Form("my-MM-ThihaNeural"), rate: str = Form("+0%"), minutes: float = Form(5)):
    groq = groq_key or os.getenv("GROQ_API_KEY", "")
    gem = gemini_key or os.getenv("GEMINI_API_KEY", "")
    if not groq or not gem:
        raise HTTPException(400, "Groq key နဲ့ Gemini key လိုအပ်ပါတယ်")
    jid = uuid.uuid4().hex[:10]
    wd = DATA / jid
    wd.mkdir(parents=True)
    video = wd / ("source" + Path(file.filename or "v.mp4").suffix)
    with open(video, "wb") as f:
        shutil.copyfileobj(file.file, f)
    JOBS[jid] = {"status": "running", "progress": 1, "message": "စတင်နေသည်"}
    threading.Thread(target=worker, args=(jid, video, groq, gem, voice, rate, minutes), daemon=True).start()
    return {"id": jid}


@app.get("/api/jobs/{jid}")
def status(jid: str):
    if jid not in JOBS:
        raise HTTPException(404)
    return JOBS[jid]


@app.get("/api/jobs/{jid}/download/{kind}")
def download(jid: str, kind: str):
    names = {"video": "final.mp4", "srt": "subtitles.srt", "script": "script.json"}
    f = DATA / jid / names.get(kind, "?")
    if jid not in JOBS or not f.exists():
        raise HTTPException(404)
    return FileResponse(f, filename=f"recap_{jid}_{f.name}")


app.mount("/", StaticFiles(directory="static", html=True), name="static")
