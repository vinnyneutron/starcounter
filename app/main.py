import base64
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from .star_counter import analyze_image

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Starcount")


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    image_bytes = await file.read()
    try:
        n_stars, score, annotated_png = analyze_image(image_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not process that image.")

    return JSONResponse({
        "n_stars": n_stars,
        "score": round(score),
        "annotated_image": "data:image/png;base64," + base64.b64encode(annotated_png).decode("ascii"),
    })


@app.get("/api/health")
def health():
    return {"status": "ok"}
