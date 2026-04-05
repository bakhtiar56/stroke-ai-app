from fastapi import FastAPI

app = FastAPI(title="Stroke AI API", version="0.1.0")


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_loaded": False, "model_version": "0.1.0"}