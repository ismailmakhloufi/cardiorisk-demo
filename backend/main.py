import json
import hashlib
import io
import base64
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import uuid
import shutil

import numpy as np
# compat numpy 2.0 + chromadb 0.4.18 (np.float_ removed)
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "int_"):
    np.int_ = np.int64
if not hasattr(np, "uint"):
    np.uint = np.uint64
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
sys.path.append(str(SRC_DIR))

from generer_rapport_patient import (  # noqa: E402
    build_dashboard_recommendations,
    build_prompt,
    build_rag_query,
    call_ollama,
    predict_patient,
    retrieve_context,
)


PATIENTS_JSON = DATA_DIR / "patients.json"
DOCTORS_JSON = DATA_DIR / "doctors.json"
PATIENTS_AUTH_JSON = DATA_DIR / "patients_auth.json"
MESSAGES_JSON = DATA_DIR / "messages.json"
BILANS_JSON = DATA_DIR / "bilans.json"
UPLOADS_DIR = DATA_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

ORDONNANCE_FIELDS = {
    "lasilix",
    "betabloquants",
    "dose_BB_pourcentage",
    "plavix_cardiocine",
    "sintrome_aod",
    "plavix",
    "cardiocine100",
    "sintrom",
    "AOD",
    "ARM",
    "INH_SGLT2",
    "Ivabradine",
}

MODEL_INPUT_FIELDS = [
    "espace_PR",
    "cause_valvulaire",
    "PAD",
    "PAS",
    "OG",
    "Uree",
    "statine",
    "ATCD_d_hospitalisation",
    "IMC",
    "IEC_dose",
    "FQ_ECG_sortie",
    "HTAP",
    "TP",
    "lymphocyte",
    "ARM",
    "QT_corrige",
    "Glycemie_a_jeun",
    "ProBNP",
    "dose_lasilix_sup120",
]

def _load_patients() -> list:
    if PATIENTS_JSON.exists():
        with open(PATIENTS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def _save_patients(patients: list):
    with open(PATIENTS_JSON, "w", encoding="utf-8") as f:
        json.dump(patients, f, ensure_ascii=False, indent=2)

def _next_patient_id(patients: list) -> int:
    return max((p.get("id", 0) for p in patients), default=0) + 1


def _load_doctors() -> list:
    if DOCTORS_JSON.exists():
        with open(DOCTORS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_doctors(doctors: list):
    with open(DOCTORS_JSON, "w", encoding="utf-8") as f:
        json.dump(doctors, f, ensure_ascii=False, indent=2)


def _doctor_id(name: str) -> str:
    slug = "-".join(name.strip().lower().split())
    return "".join(ch for ch in slug if ch.isalnum() or ch == "-") or "default"


def _password_hash(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _load_patients_auth() -> list:
    if PATIENTS_AUTH_JSON.exists():
        with open(PATIENTS_AUTH_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_patients_auth(data: list):
    with open(PATIENTS_AUTH_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_messages() -> list:
    if MESSAGES_JSON.exists():
        with open(MESSAGES_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_messages(data: list):
    with open(MESSAGES_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_bilans() -> list:
    if BILANS_JSON.exists():
        with open(BILANS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_bilans(data: list):
    with open(BILANS_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class PatientInput(BaseModel):
    espace_PR: float = Field(160, ge=50, le=600)
    cause_valvulaire: int = Field(0, ge=0, le=1)
    PAD: float = Field(65, ge=20, le=200)
    PAS: float
    OG: float
    Uree: float = Field(8.5, ge=0.5, le=600)
    statine: int = Field(0, ge=0, le=1)
    ATCD_d_hospitalisation: int = Field(0, ge=0)
    IMC: float
    IEC_dose: float
    FQ_ECG_sortie: float
    HTAP: int = Field(0, ge=0, le=2)
    TP: float = Field(75, ge=5, le=600)
    lymphocyte: float
    ARM: int = Field(0, ge=0, le=1)
    QT_corrige: float
    Glycemie_a_jeun: float = Field(1.1, ge=0.2, le=10)
    ProBNP: float = Field(1500, ge=0, le=50000)
    dose_lasilix_sup120: Optional[int] = Field(None, ge=0, le=1)
    dose_de_lasilix: Optional[float] = None
    lasilix: Optional[int] = Field(None, ge=0, le=1)
    betabloquants: Optional[int] = Field(None, ge=0, le=1)
    dose_BB_pourcentage: Optional[float] = None
    plavix_cardiocine: Optional[int] = Field(None, ge=0, le=2)
    sintrome_aod: Optional[int] = Field(None, ge=0, le=2)
    plavix: Optional[int] = Field(None, ge=0, le=1)
    cardiocine100: Optional[int] = Field(None, ge=0, le=1)
    sintrom: Optional[int] = Field(None, ge=0, le=1)
    AOD: Optional[int] = Field(None, ge=0, le=1)
    ARM: Optional[int] = Field(None, ge=0, le=1)
    INH_SGLT2: Optional[int] = Field(None, ge=0, le=1)
    Ivabradine: Optional[int] = Field(None, ge=0, le=1)


class PatientSave(BaseModel):
    patient_id: Optional[int] = None
    doctor_id: str = "default"
    nom: str = ""
    prenom: str = ""
    date_naissance: str = ""
    sexe: str = ""
    espace_PR: float = Field(160, ge=50, le=600)
    cause_valvulaire: int = Field(0, ge=0, le=1)
    PAD: float = Field(60, ge=20, le=200)
    PAS: float
    OG: float
    Uree: float = Field(8.5, ge=0.5, le=600)
    statine: int = Field(0, ge=0, le=1)
    ATCD_d_hospitalisation: int = Field(0, ge=0)
    IMC: float
    IEC_dose: float
    FQ_ECG_sortie: float
    HTAP: int = Field(0, ge=0, le=2)
    TP: float = Field(75, ge=5, le=600)
    lymphocyte: float
    ARM: int = Field(0, ge=0, le=1)
    QT_corrige: float
    Glycemie_a_jeun: float = Field(1.1, ge=0.2, le=10)
    ProBNP: float = Field(1500, ge=0, le=50000)
    dose_lasilix_sup120: Optional[int] = Field(None, ge=0, le=1)
    dose_de_lasilix: Optional[float] = None
    lasilix: Optional[int] = Field(None, ge=0, le=1)
    betabloquants: Optional[int] = Field(None, ge=0, le=1)
    dose_BB_pourcentage: Optional[float] = None
    plavix_cardiocine: Optional[int] = Field(None, ge=0, le=2)
    sintrome_aod: Optional[int] = Field(None, ge=0, le=2)
    plavix: Optional[int] = Field(None, ge=0, le=1)
    cardiocine100: Optional[int] = Field(None, ge=0, le=1)
    sintrom: Optional[int] = Field(None, ge=0, le=1)
    AOD: Optional[int] = Field(None, ge=0, le=1)
    ARM: Optional[int] = Field(None, ge=0, le=1)
    INH_SGLT2: Optional[int] = Field(None, ge=0, le=1)
    Ivabradine: Optional[int] = Field(None, ge=0, le=1)


class DoctorAuth(BaseModel):
    name: str = Field(..., min_length=2)
    password: str = Field(..., min_length=3)


class DoctorUpdate(BaseModel):
    doctor_id: str
    name: str = Field(..., min_length=2)
    password: str = ""
    photo_url: str = ""


class PatientAuthSignup(BaseModel):
    email: str = Field(..., min_length=4)
    password: str = Field(..., min_length=3)
    nom: str = Field(..., min_length=1)
    prenom: str = Field(..., min_length=1)
    doctor_id: str = Field(..., min_length=2)
    date_naissance: str = ""
    sexe: str = ""


class PatientAuthLogin(BaseModel):
    email: str
    password: str


class MessageCreate(BaseModel):
    patient_auth_id: str
    doctor_id: str
    sender_role: str = Field(..., pattern="^(patient|doctor)$")
    content: str = Field(..., min_length=1)


class BilanReply(BaseModel):
    doctor_id: str
    reply: str = Field(..., min_length=1)


class ImportPayload(BaseModel):
    filename: str
    content_base64: str


class PredictResponse(BaseModel):
    proba: float
    score_percent: float
    threshold: float
    risk_level: str
    technical_class: str
    major_factors: List[Dict]
    dashboard: Dict


class ReportResponse(PredictResponse):
    rag_context: List[Dict]
    report_markdown: str


app = FastAPI(title="IC-FEr Risk Assistant API")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    print("=== 422 Validation Error ===")
    print(f"URL: {request.url}")
    print(f"Body: {body.decode('utf-8', errors='ignore')[:2000]}")
    print(f"Errors: {exc.errors()}")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _patient_dict(patient: PatientInput) -> Dict:
    values = patient.model_dump(exclude_none=True)
    if values.get("plavix_cardiocine") is not None:
        values["plavix"] = 1 if values["plavix_cardiocine"] == 1 else 0
        values["cardiocine100"] = 1 if values["plavix_cardiocine"] == 2 else 0
    if values.get("sintrome_aod") is not None:
        values["sintrom"] = 1 if values["sintrome_aod"] == 1 else 0
        values["AOD"] = 1 if values["sintrome_aod"] == 2 else 0
    htap = values.get("HTAP")
    if htap is not None:
        try:
            hv = float(htap)
            if hv >= 3:
                values["HTAP"] = 0 if hv < 40 else 1 if hv < 60 else 2
        except:
            pass
    if values.get("dose_lasilix_sup120") is None:
        dose = values.get("dose_de_lasilix", 0) or 0
        try:
            dose = float(dose)
        except:
            dose = 0
        values["dose_lasilix_sup120"] = 1 if dose >= 120 else 0
    return values


def _prediction_values(values: Dict) -> Dict:
    return {key: values[key] for key in MODEL_INPUT_FIELDS}


def _prediction_payload(patient_dict: Dict) -> Dict:
    prediction_dict = _prediction_values(patient_dict)
    prediction = predict_patient(prediction_dict)
    recommendations = build_dashboard_recommendations(patient_dict, prediction)
    major = prediction["contributions"][:8]
    return {
        "prediction": prediction,
        "recommendations": recommendations,
        "response": {
            "proba": prediction["proba"],
            "score_percent": round(prediction["proba"] * 100, 1),
            "threshold": prediction["threshold"],
            "risk_level": prediction["niveau_risque"],
            "technical_class": prediction["classe"],
            "major_factors": major,
            "dashboard": recommendations,
        },
    }


def _medical_values(patient: PatientSave) -> Dict:
    values = patient.model_dump()
    if values.get("plavix_cardiocine") is not None:
        values["plavix"] = 1 if values["plavix_cardiocine"] == 1 else 0
        values["cardiocine100"] = 1 if values["plavix_cardiocine"] == 2 else 0
    if values.get("sintrome_aod") is not None:
        values["sintrom"] = 1 if values["sintrome_aod"] == 1 else 0
        values["AOD"] = 1 if values["sintrome_aod"] == 2 else 0
    htap = values.get("HTAP")
    if htap is not None:
        try:
            hv = float(htap)
            if hv >= 3:
                values["HTAP"] = 0 if hv < 40 else 1 if hv < 60 else 2
        except:
            pass
    if values.get("dose_lasilix_sup120") is None:
        dose = values.get("dose_de_lasilix", 0) or 0
        try:
            dose = float(dose)
        except:
            dose = 0
        values["dose_lasilix_sup120"] = 1 if dose >= 120 else 0
    ignored = {"patient_id", "doctor_id", "nom", "prenom", "date_naissance", "sexe"}
    return {k: v for k, v in values.items() if k not in ignored and v is not None}


def _consultation_record(patient: PatientSave, prediction: Dict) -> Dict:
    return {
        "date_consultation": datetime.now().isoformat(timespec="seconds"),
        "variables": _medical_values(patient),
        "resultat": {
            "proba": prediction["proba"],
            "score_percent": round(prediction["proba"] * 100, 1),
            "risk_level": prediction["niveau_risque"],
            "technical_class": prediction["classe"],
        },
    }


def _matches_doctor(patient: Dict, doctor_id: Optional[str]) -> bool:
    return not doctor_id or patient.get("doctor_id", "default") == doctor_id


def _save_patient_record(patient: PatientSave) -> Dict:
    patients = _load_patients()
    medical = _medical_values(patient)
    prediction = predict_patient(_prediction_values(medical))
    consultation = _consultation_record(patient, prediction)

    record = None
    if patient.patient_id is not None:
        for existing in patients:
            if existing.get("id") == patient.patient_id and existing.get("doctor_id", "default") == patient.doctor_id:
                record = existing
                break

    if record is None:
        record = {
            "id": _next_patient_id(patients),
            "doctor_id": patient.doctor_id,
            "nom": patient.nom,
            "prenom": patient.prenom,
            "date_naissance": patient.date_naissance,
            "sexe": patient.sexe,
            "created_at": date.today().isoformat(),
            "consultations": [],
        }
        patients.append(record)
    else:
        record["nom"] = patient.nom or record.get("nom", "")
        record["prenom"] = patient.prenom or record.get("prenom", "")
        record["date_naissance"] = patient.date_naissance or record.get("date_naissance", "")
        record["sexe"] = patient.sexe or record.get("sexe", "")

    record.setdefault("consultations", []).append(consultation)
    record["last_result"] = consultation["resultat"]
    record["last_consultation"] = consultation["date_consultation"]
    _save_patients(patients)
    return record


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/doctors/signup")
def signup_doctor(payload: DoctorAuth):
    doctors = _load_doctors()
    did = _doctor_id(payload.name)
    if any(d.get("id") == did for d in doctors):
        raise HTTPException(status_code=409, detail="Ce cardiologue existe déjà")
    doctor = {
        "id": did,
        "name": payload.name.strip(),
        "role": "Cardiologue",
        "password_hash": _password_hash(payload.password),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    doctors.append(doctor)
    _save_doctors(doctors)
    return {"doctor": {"id": doctor["id"], "name": doctor["name"], "role": doctor["role"], "photo_url": doctor.get("photo_url", "")}}


@app.post("/api/doctors/login")
def login_doctor(payload: DoctorAuth):
    doctors = _load_doctors()
    did = _doctor_id(payload.name)
    password_hash = _password_hash(payload.password)
    for doctor in doctors:
        if doctor.get("id") == did and doctor.get("password_hash") == password_hash:
            return {"doctor": {"id": doctor["id"], "name": doctor["name"], "role": doctor.get("role", "Cardiologue"), "photo_url": doctor.get("photo_url", "")}}
    raise HTTPException(status_code=401, detail="Nom ou mot de passe incorrect")


@app.put("/api/doctors/profile")
def update_doctor_profile(payload: DoctorUpdate):
    doctors = _load_doctors()
    for doctor in doctors:
        if doctor.get("id") == payload.doctor_id:
            doctor["name"] = payload.name.strip()
            doctor["photo_url"] = payload.photo_url
            if payload.password:
                doctor["password_hash"] = _password_hash(payload.password)
            _save_doctors(doctors)
            return {"doctor": {"id": doctor["id"], "name": doctor["name"], "role": doctor.get("role", "Cardiologue"), "photo_url": doctor.get("photo_url", "")}}
    raise HTTPException(status_code=404, detail="Cardiologue non trouvé")


@app.post("/api/predict", response_model=PredictResponse)
def predict(patient: PatientInput):
    patient_dict = _patient_dict(patient)
    payload = _prediction_payload(patient_dict)
    return payload["response"]


@app.post("/api/report", response_model=ReportResponse)
def report(patient: PatientInput):
    patient_dict = _patient_dict(patient)
    payload = _prediction_payload(patient_dict)
    prediction = payload["prediction"]
    recommendations = payload["recommendations"]
    query = build_rag_query(patient_dict, prediction)
    context = retrieve_context(query)
    prompt = build_prompt(patient_dict, prediction, context, recommendations)
    report_markdown = call_ollama(prompt)
    return {**payload["response"], "rag_context": context, "report_markdown": report_markdown}


@app.post("/api/patients")
def save_patient(patient: PatientSave):
    record = _save_patient_record(patient)
    return {"id": record["id"], "message": "Patient enregistré", "record": record}


@app.post("/api/patients/import")
def import_patients(doctor_id: str, payload: ImportPayload):
    raw_base64 = payload.content_base64.split(",")[-1]
    content = base64.b64decode(raw_base64)
    suffix = Path(payload.filename or "").suffix.lower()
    try:
        if suffix == ".csv":
            df = pd.read_csv(io.BytesIO(content))
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Format non supporté. Utilisez CSV ou Excel.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Fichier illisible: {exc}") from exc

    import_required = ["nom", "prenom", "date_naissance", "sexe", "espace_PR", "cause_valvulaire", "PAD", "PAS", "OG", "Uree", "statine", "ATCD_d_hospitalisation", "IMC", "IEC_dose", "FQ_ECG_sortie", "HTAP", "TP", "lymphocyte", "ARM", "QT_corrige", "Glycemie_a_jeun", "ProBNP", "dose_de_lasilix"]
    missing = [col for col in import_required if col not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Colonnes manquantes: {', '.join(missing)}")

    imported = 0
    errors = []
    for idx, row in df.iterrows():
        try:
            date_naissance = row["date_naissance"]
            if pd.notna(date_naissance) and hasattr(date_naissance, "date"):
                date_naissance = date_naissance.date().isoformat()
            data = {
                "doctor_id": doctor_id,
                "nom": str(row["nom"]),
                "prenom": str(row["prenom"]),
                "date_naissance": "" if pd.isna(date_naissance) else str(date_naissance),
                "sexe": str(row["sexe"]),
            }
            for key in MODEL_INPUT_FIELDS:
                if key == "dose_lasilix_sup120":
                    dose = row.get("dose_de_lasilix", 0) or 0
                    try:
                        dose = float(dose)
                    except:
                        dose = 0
                    data[key] = 1 if dose >= 120 else 0
                    data["dose_de_lasilix"] = row["dose_de_lasilix"]
                else:
                    data[key] = row[key]
            for key in ORDONNANCE_FIELDS:
                if key in df.columns and pd.notna(row[key]):
                    data[key] = row[key]
            _save_patient_record(PatientSave(**data))
            imported += 1
        except Exception as exc:
            errors.append({"ligne": int(idx) + 2, "erreur": str(exc)})
    return {"imported": imported, "errors": errors}


@app.get("/api/patients/export")
def export_patients(doctor_id: Optional[str] = None):
    rows = []
    for patient in _load_patients():
        if not _matches_doctor(patient, doctor_id):
            continue
        for consultation in patient.get("consultations", []):
            row = {
                "patient_id": patient.get("id"),
                "nom": patient.get("nom", ""),
                "prenom": patient.get("prenom", ""),
                "date_naissance": patient.get("date_naissance", ""),
                "sexe": patient.get("sexe", ""),
                "date_consultation": consultation.get("date_consultation", ""),
                **consultation.get("variables", {}),
                **consultation.get("resultat", {}),
            }
            rows.append(row)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, index=False, sheet_name="patients")
    output.seek(0)
    headers = {"Content-Disposition": "attachment; filename=patients_icfer.xlsx"}
    return StreamingResponse(output, headers=headers, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@app.get("/api/patients")
def list_patients(doctor_id: Optional[str] = None):
    patients = _load_patients()
    filtered = [p for p in patients if _matches_doctor(p, doctor_id)]
    return {"patients": filtered}


@app.get("/api/patients/critical")
def list_critical_patients(doctor_id: Optional[str] = None):
    patients = _load_patients()
    critical = []
    for patient in patients:
        if not _matches_doctor(patient, doctor_id):
            continue
        result = patient.get("last_result", {})
        if result.get("risk_level") in ("eleve", "tres eleve") or result.get("score_percent", 0) >= 70:
            critical.append(patient)
    return {"patients": critical}


@app.get("/api/patients/{patient_id}")
def get_patient(patient_id: int, doctor_id: Optional[str] = None):
    patients = _load_patients()
    for p in patients:
        if p["id"] == patient_id and _matches_doctor(p, doctor_id):
            return p
    raise HTTPException(status_code=404, detail="Patient non trouvé")


# ── Patient portal & communication (non-destructive extension) ──

@app.get("/api/doctors/list")
def list_doctors_public():
    doctors = _load_doctors()
    return {"doctors": [{"id": d["id"], "name": d["name"], "role": d.get("role", "Cardiologue")} for d in doctors]}


@app.post("/api/patient/signup")
def signup_patient(payload: PatientAuthSignup):
    auths = _load_patients_auth()
    email_norm = payload.email.strip().lower()
    if any(a.get("email", "").lower() == email_norm for a in auths):
        raise HTTPException(status_code=409, detail="Email déjà utilisé")
    doctors = _load_doctors()
    if not any(d.get("id") == payload.doctor_id for d in doctors):
        raise HTTPException(status_code=404, detail="Médecin non trouvé")
    pid = f"pat-{uuid.uuid4().hex[:8]}"
    record = {
        "id": pid,
        "email": email_norm,
        "nom": payload.nom.strip(),
        "prenom": payload.prenom.strip(),
        "doctor_id": payload.doctor_id,
        "date_naissance": payload.date_naissance,
        "sexe": payload.sexe,
        "password_hash": _password_hash(payload.password),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    auths.append(record)
    _save_patients_auth(auths)
    public = {k: v for k, v in record.items() if k != "password_hash"}
    return {"patient": public}


@app.post("/api/patient/login")
def login_patient(payload: PatientAuthLogin):
    auths = _load_patients_auth()
    email_norm = payload.email.strip().lower()
    h = _password_hash(payload.password)
    for a in auths:
        if a.get("email", "").lower() == email_norm and a.get("password_hash") == h:
            public = {k: v for k, v in a.items() if k != "password_hash"}
            return {"patient": public}
    raise HTTPException(status_code=401, detail="Email ou mot de passe incorrect")


@app.get("/api/patient/me")
def get_patient_auth(patient_auth_id: str):
    auths = _load_patients_auth()
    for a in auths:
        if a.get("id") == patient_auth_id:
            public = {k: v for k, v in a.items() if k != "password_hash"}
            return {"patient": public}
    raise HTTPException(status_code=404, detail="Patient non trouvé")


@app.post("/api/messages")
def post_message(payload: MessageCreate):
    # strict isolation: patient can only talk to his assigned doctor
    auths = _load_patients_auth()
    patient_auth = next((a for a in auths if a.get("id") == payload.patient_auth_id), None)
    if not patient_auth:
        raise HTTPException(status_code=404, detail="Patient non trouvé")
    if patient_auth.get("doctor_id") != payload.doctor_id:
        raise HTTPException(status_code=403, detail="Ce patient n'est pas rattaché à ce médecin")
    # doctor sending must also match (doctor_id must be the patient's doctor)
    # already checked above
    messages = _load_messages()
    msg = {
        "id": f"msg-{uuid.uuid4().hex[:8]}",
        "patient_auth_id": payload.patient_auth_id,
        "doctor_id": payload.doctor_id,
        "sender_role": payload.sender_role,
        "content": payload.content.strip(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    messages.append(msg)
    _save_messages(messages)
    return {"message": msg}


@app.get("/api/messages")
def list_messages(patient_auth_id: str, doctor_id: str):
    # isolation check
    auths = _load_patients_auth()
    pa = next((a for a in auths if a.get("id") == patient_auth_id), None)
    if pa and pa.get("doctor_id") != doctor_id:
        raise HTTPException(status_code=403, detail="Accès non autorisé à ce médecin")
    messages = _load_messages()
    filtered = [m for m in messages if m.get("patient_auth_id") == patient_auth_id and m.get("doctor_id") == doctor_id]
    filtered.sort(key=lambda x: x.get("created_at", ""))
    return {"messages": filtered}


@app.get("/api/messages/doctor")
def list_messages_for_doctor(doctor_id: str):
    messages = _load_messages()
    bilans = _load_bilans()
    auths = _load_patients_auth()
    # group by patient_auth_id
    grouped = {}
    for m in messages:
        if m.get("doctor_id") != doctor_id:
            continue
        pid = m.get("patient_auth_id")
        grouped.setdefault(pid, []).append(m)
    # also include patients with bilans but no messages
    for b in bilans:
        if b.get("doctor_id") != doctor_id:
            continue
        pid = b.get("patient_auth_id")
        grouped.setdefault(pid, [])
    result = []
    for pid, msgs in grouped.items():
        auth = next((a for a in auths if a.get("id") == pid), None)
        msgs_sorted = sorted(msgs, key=lambda x: x.get("created_at", ""))
        last = msgs_sorted[-1] if msgs_sorted else None
        patient_bilans = [b for b in bilans if b.get("patient_auth_id") == pid and b.get("doctor_id") == doctor_id]
        # non-lus = messages patient après le dernier message médecin + bilans en_attente non répondus
        last_doctor_idx = -1
        for idx, m in enumerate(msgs_sorted):
            if m.get("sender_role") == "doctor":
                last_doctor_idx = idx
        unread_msgs = sum(1 for idx, m in enumerate(msgs_sorted) if idx > last_doctor_idx and m.get("sender_role") == "patient")
        unread_bilans = sum(1 for b in patient_bilans if b.get("status") == "en_attente")
        # badge = 1 par conversation ayant au moins un non-lu (message ou bilan), sinon 0
        has_unread = 1 if (unread_msgs > 0 or unread_bilans > 0) else 0
        result.append({
            "patient_auth_id": pid,
            "patient": {k: v for k, v in (auth or {}).items() if k != "password_hash"},
            "messages": msgs_sorted,
            "last_message": last,
            "bilans": sorted(patient_bilans, key=lambda x: x.get("created_at", ""), reverse=True),
            "unread_count": has_unread,
            "unread_detail": {"messages": unread_msgs, "bilans": unread_bilans},
        })
    # sort by last activity
    result.sort(key=lambda x: (x["last_message"] or {}).get("created_at", "") if x["last_message"] else (x["bilans"][0].get("created_at", "") if x["bilans"] else ""), reverse=True)
    return {"conversations": result}


@app.post("/api/bilans/upload")
async def upload_bilan(
    patient_auth_id: str = Form(...),
    doctor_id: str = Form(...),
    description: str = Form(""),
    file: UploadFile = File(...),
):
    auths = _load_patients_auth()
    patient_auth = next((a for a in auths if a.get("id") == patient_auth_id), None)
    if not patient_auth:
        raise HTTPException(status_code=404, detail="Patient non trouvé")
    if patient_auth.get("doctor_id") != doctor_id:
        raise HTTPException(status_code=403, detail="Ce patient n'est pas rattaché à ce médecin")
    doctors = _load_doctors()
    if not any(d.get("id") == doctor_id for d in doctors):
        raise HTTPException(status_code=404, detail="Médecin non trouvé")
    # validate file
    allowed = {".pdf", ".png", ".jpg", ".jpeg"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail="Format non supporté. PDF, PNG, JPG uniquement.")
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Fichier trop volumineux (max 10 Mo)")
    bid = f"bilan-{uuid.uuid4().hex[:8]}"
    safe_name = f"{bid}{suffix}"
    patient_dir = UPLOADS_DIR / patient_auth_id
    patient_dir.mkdir(parents=True, exist_ok=True)
    dest = patient_dir / safe_name
    with open(dest, "wb") as f:
        f.write(content)
    bilans = _load_bilans()
    record = {
        "id": bid,
        "patient_auth_id": patient_auth_id,
        "doctor_id": doctor_id,
        "filename": file.filename,
        "stored_filename": safe_name,
        "description": description.strip(),
        "suffix": suffix,
        "size": len(content),
        "status": "en_attente",
        "reply": "",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    bilans.append(record)
    _save_bilans(bilans)
    # also create a message to notify doctor
    messages = _load_messages()
    messages.append({
        "id": f"msg-{uuid.uuid4().hex[:8]}",
        "patient_auth_id": patient_auth_id,
        "doctor_id": doctor_id,
        "sender_role": "patient",
        "content": f"📎 Bilan téléversé: {file.filename} — {description.strip()}" if description.strip() else f"📎 Bilan téléversé: {file.filename}",
        "created_at": record["created_at"],
        "bilan_id": bid,
    })
    _save_messages(messages)
    return {"bilan": record}


@app.get("/api/bilans")
def list_bilans(patient_auth_id: Optional[str] = None, doctor_id: Optional[str] = None):
    if patient_auth_id and doctor_id:
        auths = _load_patients_auth()
        pa = next((a for a in auths if a.get("id") == patient_auth_id), None)
        if pa and pa.get("doctor_id") != doctor_id:
            raise HTTPException(status_code=403, detail="Accès non autorisé")
    bilans = _load_bilans()
    filtered = bilans
    if patient_auth_id:
        filtered = [b for b in filtered if b.get("patient_auth_id") == patient_auth_id]
    if doctor_id:
        filtered = [b for b in filtered if b.get("doctor_id") == doctor_id]
    filtered.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return {"bilans": filtered}


@app.get("/api/bilans/download/{bilan_id}")
def download_bilan(bilan_id: str):
    bilans = _load_bilans()
    bilan = next((b for b in bilans if b.get("id") == bilan_id), None)
    if not bilan:
        raise HTTPException(status_code=404, detail="Bilan non trouvé")
    path = UPLOADS_DIR / bilan.get("patient_auth_id", "") / bilan.get("stored_filename", "")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Fichier introuvable")
    media = "application/pdf" if bilan.get("suffix") == ".pdf" else "image/jpeg" if bilan.get("suffix") in (".jpg", ".jpeg") else "image/png"
    return StreamingResponse(open(path, "rb"), media_type=media, headers={"Content-Disposition": f'inline; filename="{bilan.get("filename", "bilan")}"'})


@app.post("/api/bilans/{bilan_id}/reply")
def reply_bilan(bilan_id: str, payload: BilanReply):
    bilans = _load_bilans()
    bilan = next((b for b in bilans if b.get("id") == bilan_id), None)
    if not bilan:
        raise HTTPException(status_code=404, detail="Bilan non trouvé")
    if bilan.get("doctor_id") != payload.doctor_id:
        raise HTTPException(status_code=403, detail="Non autorisé")
    bilan["reply"] = payload.reply.strip()
    bilan["status"] = "repondu"
    bilan["replied_at"] = datetime.now().isoformat(timespec="seconds")
    _save_bilans(bilans)
    # notify via message
    messages = _load_messages()
    messages.append({
        "id": f"msg-{uuid.uuid4().hex[:8]}",
        "patient_auth_id": bilan.get("patient_auth_id"),
        "doctor_id": payload.doctor_id,
        "sender_role": "doctor",
        "content": f"🩺 Réponse au bilan {bilan.get('filename')}: {payload.reply.strip()}",
        "created_at": bilan["replied_at"],
        "bilan_id": bilan_id,
    })
    _save_messages(messages)
    return {"bilan": bilan}


# ── Servir le frontend en prod (single URL pour démo jury) ──
FRONTEND_DIST = PROJECT_DIR / "frontend" / "dist"
if FRONTEND_DIST.exists():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
