from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import json, pickle
from Traitement import *
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
POINTER = MODEL_DIR / "production_model.txt"  
THRESHOLD_FILE = MODEL_DIR / "threshold.json"

def load_model_and_threshold():
    if POINTER.exists():
        name = POINTER.read_text().strip()
        model_path = MODEL_DIR / name
    else:
        pkl_files = list(MODEL_DIR.glob("*.pkl"))
        if not pkl_files:
            raise FileNotFoundError(f"Aucun fichier .pkl trouvé dans {MODEL_DIR}")
        model_path = pkl_files[0]
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable à l’emplacement : {model_path}")
    with open(model_path, "rb") as f:
        mdl = pickle.load(f)
    thr = 0.5
    if THRESHOLD_FILE.exists():
        try:
            thr = float(json.loads(THRESHOLD_FILE.read_text()).get("threshold", thr))
        except Exception:
            pass
    return mdl, thr, model_path.name
model, BEST_THRESHOLD, model_name = load_model_and_threshold()
class PredictRequest(BaseModel):
    text: str
class PredictResponse(BaseModel):
    prediction: str
    probability_spam: float
    threshold: float
    model: str
app = FastAPI(title="Spam Classifier API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)
@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "model": model_name, "threshold": BEST_THRESHOLD}
@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(req: PredictRequest):
    text_brut = (req.text or "").strip()
    if not text_brut:
        raise HTTPException(status_code=400, detail="Le texte ne doit pas être vide.")
    text_propre = nettoyage(text_brut)
    prob_spam = float(model.predict_proba([text_propre])[0][1])
    pred = 1 if prob_spam >= BEST_THRESHOLD else 0
    label = "Spam" if pred == 1 else "Not spam"
    return PredictResponse(
        prediction=label,
        probability_spam=prob_spam,
        threshold=BEST_THRESHOLD,
        model=model_name,
    )
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)