"""
FastAPI Pocket-Doctor Universal Agent (LLM-assisted MVP)

What this file does (implemented now):
 - Provides a FastAPI service with a single powerful endpoint `/analyze` that accepts a
   patient record (labs, vitals, meds, notes, radiology/pathology report text, symptoms).
 - Produces a patient-facing "pocket-doctor" report that attempts to:
     1) identify likely diseases/conditions (broad differential) using an LLM-stub + heuristics
     2) for each candidate, give a detailed, test-by-test explanation of why it was suggested
     3) compute an explainable severity score and category (Low/Moderate/High/Emergency)
     4) provide immediate safety-first next steps and clinician-facing evidence snippets
 - Includes a lightweight LLM stub so you can later replace the `llm_suggest_differential`
   function with calls to a real LLM (OpenAI/GPT/other) or a hosted model.
 - All outputs include provenance (which labs/lines triggered each conclusion) and confidence.
 - Emergency findings are flagged clearly to require human-in-the-loop confirmation.

Important safety reminder (displayed here in code):
 - THIS IS A PROTOTYPE. Not for production or to replace clinicians.
 - Urgent recommendations must be treated conservatively and confirmed by a clinician.

Run locally:
 - pip install fastapi uvicorn pydantic matplotlib
 - uvicorn fastapi_pocket_doctor_universal:app --reload --port 8000

Example input (JSON):
{
  "patient_id": "demo-1",
  "labs": {"hba1c_percent": 8.2, "fasting_glucose_mg_dL": 160, "creatinine_mg_dL": 1.5, "wbc": 14},
  "vitals": {"systolic_bp": 150, "diastolic_bp": 95, "heart_rate": 110, "temperature_c": 38.5},
  "medications": ["lisinopril"],
  "notes": "Patient with polyuria and polydipsia. CT head shows no acute hemorrhage. Ultrasound: gallstones noted in impression.",
  "reports": ["Impression: multiple gallstones in the gallbladder; no wall thickening."],
  "symptoms": ["polyuria","polydipsia"]
}

Response: JSON with an overall summary and per-disease detailed blocks.

"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import math, io, base64, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = FastAPI(title="Pocket-Doctor Universal Agent (LLM-assisted MVP)")
from fastapi import UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict
import aiofiles
import os
import uuid
from PIL import Image

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# root/health already present
@app.post("/upload", tags=["files"])
async def upload_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    # Basic validation
    if file.content_type.split("/")[0] not in {"image", "application"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # limit size (approx) - you can read stream to check exact size if needed
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(UPLOAD_DIR, filename)

    # Save file asynchronously
    async with aiofiles.open(filepath, "wb") as out_file:
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:  # 10 MB limit
            raise HTTPException(status_code=413, detail="File too large")
        await out_file.write(content)

    # Optional sanity check for images (Pillow)
    try:
        if file.content_type.startswith("image"):
            img = Image.open(filepath)
            img.verify()
    except Exception:
        os.remove(filepath)
        raise HTTPException(status_code=400, detail="Invalid image")

    # Kick off background analysis (non-blocking)
    if background_tasks is not None:
        background_tasks.add_task(analyze_file_task, filepath)

    return {"upload_id": filename, "message": "File uploaded, analysis started (async)"}

# example background task - put AI call here
def analyze_file_task(filepath: str):
    # 1) Send file to AI model or ML pipeline
    # 2) Store results in DB or storage
    # 3) Optionally notify user (webhook/email)
    try:
        # pseudo-code: results = call_your_ai_api(filepath)
        # For now we'll write a local JSON result file
        result_path = filepath + ".result.json"
        with open(result_path, "w") as f:
            f.write('{"status":"analysis-complete","note":"dummy result, replace with AI call"}')
    except Exception as e:
        # log error
        print("analysis error:", e)

from fastapi.responses import JSONResponse

@app.get("/", tags=["meta"])
def root():
    """
    Simple health / root endpoint so / and /docs don't 404.
    """
    return JSONResponse({"message": "Pocket Doctor backend is live!"})

@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allows your frontend (React) to connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- Health + Upload endpoints ---
from fastapi import UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
import aiofiles, os, uuid
from PIL import Image

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", tags=["meta"])
def root():
    """Simple root endpoint so / and /docs don't 404."""
    return JSONResponse({"message": "Pocket Doctor backend is live!"})

@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}

@app.post("/upload", tags=["files"])
async def upload_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    if file.content_type.split("/")[0] not in {"image", "application"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    filename = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(UPLOAD_DIR, filename)

    async with aiofiles.open(filepath, "wb") as out_file:
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:  # 10 MB limit
            raise HTTPException(status_code=413, detail="File too large")
        await out_file.write(content)

    try:
        if file.content_type.startswith("image"):
            img = Image.open(filepath)
            img.verify()
    except Exception:
        os.remove(filepath)
        raise HTTPException(status_code=400, detail="Invalid image")

    if background_tasks is not None:
        background_tasks.add_task(lambda: print(f"Analyzing {filepath}..."))  # placeholder

    return {"upload_id": filename, "message": "File uploaded successfully"}



# -------------------- Data models --------------------
class PatientRecord(BaseModel):
    patient_id: str
    labs: Optional[Dict[str, float]] = Field(default_factory=dict)
    vitals: Optional[Dict[str, float]] = Field(default_factory=dict)
    medications: Optional[List[str]] = Field(default_factory=list)
    notes: Optional[str] = ""
    reports: Optional[List[str]] = Field(default_factory=list)
    symptoms: Optional[List[str]] = Field(default_factory=list)

class DiseaseDetail(BaseModel):
    disease_id: str
    disease_name: str
    icd10: Optional[str] = None
    summary: str
    severity_score: float
    category: str
    recommended_action: str
    confidence: float
    evidence_snippets: Dict[str, Any]
    detailed_explanation: str

class AnalyzeOutput(BaseModel):
    patient_id: str
    overall_score: float
    overall_category: str
    top_differential: List[Dict[str, Any]]
    findings: List[DiseaseDetail]
    pie_chart_base64: Optional[str] = None

# -------------------- Lightweight disease library (starter) --------------------
# This should be expanded by clinicians. Each entry has keywords, test names to inspect, and explanation templates.
DISEASE_LIBRARY = {
    'diabetes_mellitus': {
        'name': 'Diabetes mellitus',
        'icd10': 'E11',
        'keywords': ['diabetes', 'hyperglycemia', 'hba1c', 'insulin', 'polyuria', 'polydipsia'],
        'tests': ['hba1c_percent', 'fasting_glucose_mg_dL', 'random_glucose_mg_dL'],
        'explain_template': (
            'Diabetes is suggested by elevated glucose values. Typical diagnostic cutoffs used by clinicians: HbA1c >= 6.5% or fasting glucose >= 126 mg/dL. ' 
            'Contributions: {contribs}.'
        )
    },
    'hypertension': {
        'name': 'Hypertension',
        'icd10': 'I10',
        'keywords': ['hypertension', 'blood pressure', 'bp', 'hypertensive'],
        'tests': ['systolic_bp', 'diastolic_bp'],
        'explain_template': (
            'Hypertension is suggested by elevated blood pressure readings. Severity often depends on sustained levels; single readings may be noisy. Contributions: {contribs}.'
        )
    },
    'acute_kidney_injury': {
        'name': 'Acute kidney injury (AKI)',
        'icd10': 'N17',
        'keywords': ['acute kidney injury', 'aki', 'creatinine rise', 'oliguria'],
        'tests': ['creatinine_mg_dL', 'bun_mg_dL', 'urine_output_ml'],
        'explain_template': (
            'AKI is suggested by rising creatinine or reduced urine output. Rapid change matters more than absolute value. Contributions: {contribs}.'
        )
    },
    'gallstones': {
        'name': 'Cholelithiasis (gallstones)',
        'icd10': 'K80',
        'keywords': ['gallstones', 'cholelithiasis', 'gallbladder stone', 'gallbladder stones', 'sonography'],
        'tests': [],
        'explain_template': (
            'Gallstones are usually identified on ultrasound. The agent highlights the report impressions that explicitly state stones. Contributions: {contribs}.'
        )
    },
    'ischemic_stroke': {
        'name': 'Ischemic stroke',
        'icd10': 'I63',
        'keywords': ['acute infarct', 'ischemia', 'stroke', 'hypodensity', 'restricted diffusion', 'thrombolysis'],
        'tests': [],
        'explain_template': (
            'Ischemic stroke is suggested by imaging findings or notes describing acute focal deficits. Time of onset is critical for urgent treatments. Contributions: {contribs}.'
        )
    },
    'breast_cancer': {
        'name': 'Breast cancer',
        'icd10': 'C50',
        'keywords': ['breast cancer', 'carcinoma', 'invasive ductal', 'biopsy: invasive', 'malignant'],
        'tests': ['tumor_size_cm', 'er_status', 'pr_status', 'her2_status'],
        'explain_template': (
            'Breast cancer inference is from pathology or imaging; key elements include tumor size and receptor status. Contributions: {contribs}.'
        )
    }
}

# -------------------- Utilities: normalization, keyword finding, pie chart --------------------

def find_keywords(text: str, keywords: List[str]) -> List[str]:
    if not text:
        return []
    txt = text.lower()
    found = []
    for k in keywords:
        if k.lower() in txt:
            found.append(k)
    return found

# Basic numeric normalizers per test name (extend as needed)
def normalize_numeric(test: str, value: Optional[float]) -> float:
    if value is None:
        return 0.0
    try:
        v = float(value)
    except Exception:
        return 0.0
    if test == 'hba1c_percent':
        if v <= 5.6: return 0.0
        if v >= 12.0: return 1.0
        return (v - 5.6) / (12.0 - 5.6)
    if test == 'fasting_glucose_mg_dL' or test == 'random_glucose_mg_dL':
        if v <= 99: return 0.0
        if v >= 400: return 1.0
        return (v - 99) / (400 - 99)
    if test == 'systolic_bp':
        if v < 120: return 0.0
        if v >= 180: return 1.0
        return (v - 120) / (180 - 120)
    if test == 'diastolic_bp':
        if v < 80: return 0.0
        if v >= 120: return 1.0
        return (v - 80) / (120 - 80)
    if test == 'creatinine_mg_dL':
        if v <= 1.2: return 0.0
        if v >= 5.0: return 1.0
        return (v - 1.2) / (5.0 - 1.2)
    if test == 'wbc':
        if v <= 11: return 0.0
        if v >= 25: return 1.0
        return (v - 11) / (25 - 11)
    # default gentle mapping
    return min(1.0, max(0.0, (v - 0.0) / (v + 10.0)))

def score_to_category(score: float) -> str:
    if score >= 0.8: return 'Emergency'
    if score >= 0.6: return 'High'
    if score >= 0.4: return 'Moderate'
    return 'Low'

def make_pie_base64(score: float) -> str:
    labels = ['Emergency','High','Moderate','Low']
    # transform score to slices
    low=mod=high=emerg=0.0
    if score <= 0.4:
        low = 1.0
    else:
        rem = score
        if rem > 0.8:
            emerg = (rem - 0.8) / 0.2; rem = 0.8
        if rem > 0.6:
            high = (rem - 0.6) / 0.2; rem = 0.6
        if rem > 0.4:
            mod = (rem - 0.4) / 0.2; rem = 0.4
        low = max(0.0, 1.0 - (emerg + high + mod))
    slices = [emerg, high, mod, low]
    total = sum(slices)
    if total <= 0: slices = [0,0,0,1]; total=1
    slices = [s/total for s in slices]
    fig, ax = plt.subplots(figsize=(4,4))
    ax.pie(slices, labels=labels, autopct='%1.1f%%')
    ax.set_title('Overall severity (visual)')
    buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# -------------------- LLM stub: generate broad differential with evidence tokens --------------------
# Replace this with a real LLM call if available. The stub uses heuristics + keyword matching.

def llm_suggest_differential(record: PatientRecord, top_k: int = 5) -> List[Dict[str,Any]]:
    """
    Returns a ranked list of candidate diseases with short rationale and evidence cues.
    This is an LLM *stub* that combines:
      - keyword matches in notes/reports
      - strong lab signals (normalized values)
      - abnormal vitals
    The output format is [{'disease_id':..., 'score':..., 'rationale':..., 'evidence':[...]}, ...]
    """
    candidates = []
    text_blob = ' '.join((record.notes or '') + ' ' + ' '.join(record.reports or []) + ' ' + ' '.join(record.symptoms or []))

    # keyword signals
    for did, meta in DISEASE_LIBRARY.items():
        kw_found = find_keywords(text_blob, meta.get('keywords', []))
        kw_score = min(1.0, len(kw_found) / max(1.0, len(meta.get('keywords', []))))

        # lab-driven signals
        lab_signal = 0.0
        for t in meta.get('tests', []):
            v = record.labs.get(t)
            lab_signal = max(lab_signal, normalize_numeric(t, v))

        # vitals signals for hypertension/sepsis
        vit_signal = 0.0
        if did == 'hypertension':
            vit_signal = max(normalize_numeric('systolic_bp', record.vitals.get('systolic_bp') if record.vitals else None), normalize_numeric('diastolic_bp', record.vitals.get('diastolic_bp') if record.vitals else None))
        if did == 'acute_kidney_injury':
            vit_signal = normalize_numeric('creatinine_mg_dL', record.labs.get('creatinine_mg_dL'))

        # simple ensemble score
        score = 0.6 * lab_signal + 0.25 * kw_score + 0.15 * vit_signal
        # bump score if explicit phrase like 'impression: gallstones' appears
        if re.search(r'impression', text_blob.lower()) and any(k in text_blob.lower() for k in meta.get('keywords', [])):
            score = min(1.0, score + 0.15)

        rationale = f"keyword_score={kw_score:.2f}, lab_signal={lab_signal:.2f}, vitals={vit_signal:.2f}"
        evidence = []
        if kw_found:
            evidence.extend([f"keyword:{k}" for k in kw_found])
        for t in meta.get('tests', []):
            if t in record.labs and record.labs.get(t) is not None:
                evidence.append(f"lab:{t}={record.labs.get(t)}")
        candidates.append({'disease_id': did, 'score': round(score,4), 'rationale': rationale, 'evidence': evidence})

    # sort and return top_k
    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates[:top_k]

# -------------------- Detailed per-disease evaluator --------------------

def evaluate_disease(did: str, record: PatientRecord, llm_hint: Optional[Dict[str,Any]] = None) -> DiseaseDetail:
    meta = DISEASE_LIBRARY.get(did)
    if not meta:
        return DiseaseDetail(
            disease_id=did, disease_name=did, icd10=None,
            summary='No rule', severity_score=0.0, category='Unknown',
            recommended_action='No rule available', confidence=0.0, evidence_snippets={}, detailed_explanation=''
        )

    evidence = {}
    contribs = []
    # gather numeric contributions
    for test in meta.get('tests', []):
        val = record.labs.get(test)
        norm = normalize_numeric(test, val)
        if val is not None:
            evidence[test] = val
            contribs.append(f"{test}={val} (norm {norm:.2f})")

    # keyword evidence from notes/reports/symptoms
    text_blob = ' '.join((record.notes or '') + ' ' + ' '.join(record.reports or []) + ' ' + ' '.join(record.symptoms or []))
    kw_found = find_keywords(text_blob, meta.get('keywords', []))
    if kw_found:
        evidence['keywords'] = kw_found

    # combine an explainable score: average of top signals (lab, keyword presence, vitals)
    lab_scores = [normalize_numeric(t, record.labs.get(t)) for t in meta.get('tests', [])]
    lab_score = max(lab_scores) if lab_scores else 0.0
    kw_score = min(1.0, len(kw_found) / max(1.0, len(meta.get('keywords', []))))

    vit_score = 0.0
    if did == 'hypertension':
        vit_score = max(normalize_numeric('systolic_bp', record.vitals.get('systolic_bp') if record.vitals else None), normalize_numeric('diastolic_bp', record.vitals.get('diastolic_bp') if record.vitals else None))
    if did == 'ischemic_stroke':
        # if reports contain 'acute infarct' or 'restricted diffusion' set vit_score higher as proxy for imaging
        vit_score = 1.0 if re.search(r'acute infarct|restricted diffusion|acute ischemic', text_blob.lower()) else 0.0

    # LLM hint (if provided) increases confidence and nudges score
    llm_score = llm_hint['score'] if llm_hint and 'score' in llm_hint else 0.0

    # weight ensemble (tunable)
    final_score = 0.0
    weights = {'lab': 0.5, 'kw': 0.25, 'vit': 0.15, 'llm': 0.1}
    final_score = lab_score * weights['lab'] + kw_score * weights['kw'] + vit_score * weights['vit'] + llm_score * weights['llm']
    final_score = max(0.0, min(1.0, final_score))

    category = score_to_category(final_score)

    # recommendation templates (safety-first)
    if category == 'Emergency':
        action = 'Emergency: seek immediate emergency care or call local emergency services. Do not delay.'
    elif category == 'High':
        action = 'High urgency: contact your treating clinician within 24 hours or visit urgent care.'
    elif category == 'Moderate':
        action = 'Moderate urgency: schedule a clinician follow-up within 72 hours and closely monitor symptoms.'
    else:
        action = 'Low urgency: routine follow-up with primary care and self-monitoring as advised.'

    # detailed explanation built from template and evidence
    contribs_text = '; '.join(contribs) if contribs else 'no strong lab contributors'
    detailed_explanation = meta.get('explain_template', '').format(contribs=contribs_text)

    # confidence heuristic
    confidence = min(1.0, 0.15 + 0.5 * (1.0 if lab_score>0 else 0.0) + 0.3 * (kw_score) + 0.1 * llm_score)

    return DiseaseDetail(
        disease_id=did,
        disease_name=meta.get('name'),
        icd10=meta.get('icd10'),
        summary=meta.get('explain_template','').split('.')[0],
        severity_score=round(final_score,4),
        category=category,
        recommended_action=action,
        confidence=round(confidence,4),
        evidence_snippets=evidence,
        detailed_explanation=detailed_explanation
    )

# -------------------- Main analyze endpoint --------------------
@app.post('/analyze', response_model=AnalyzeOutput)
def analyze(record: PatientRecord):
    try:
        # 1) LLM-assisted differential (stub)
        diff = llm_suggest_differential(record, top_k=8)

        # 2) For each suggested disease evaluate in depth
        findings = []
        for cand in diff:
            did = cand['disease_id']
            detail = evaluate_disease(did, record, llm_hint=cand)
            # attach LLM rationale into evidence if useful
            if cand.get('evidence'):
                # merge evidence list into evidence_snippets for traceability
                for i,e in enumerate(cand['evidence']):
                    detail.evidence_snippets[f'lm_evidence_{i}'] = e
            findings.append(detail)

        # 3) sort findings by severity
        findings.sort(key=lambda x: x.severity_score, reverse=True)

        overall_score = max([f.severity_score for f in findings]) if findings else 0.0
        overall_cat = score_to_category(overall_score)
        pie_b64 = make_pie_base64(overall_score)

        # 4) top differential summary for quick view
        top_diff = [{'disease_id': d['disease_id'], 'score': d['score'], 'rationale': d['rationale'], 'evidence': d['evidence']} for d in diff]

        return AnalyzeOutput(
            patient_id=record.patient_id,
            overall_score=round(overall_score,4),
            overall_category=overall_cat,
            top_differential=top_diff,
            findings=findings,
            pie_chart_base64=pie_b64
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/health')
def health():
    return {'status': 'ok'}

# -------------------- Developer notes --------------------
# - To integrate a real LLM: replace llm_suggest_differential with a function that calls the LLM
#   (OpenAI/GPT or local model) passing in a structured prompt containing extracted labs, vitals, reports,
#   and ask the model to return a JSON array of candidate diagnoses + explanations + evidence pointers.
# - To support more diseases: extend DISEASE_LIBRARY with keywords, tests, and an explain_template.
# - For images (mammogram/CT/US): integrate an imaging model pipeline (DICOM ingestion + pre-trained model)
#   and include the model outputs in the evidence blob fed to the LLM/heuristics.
# - ALWAYS include human-in-the-loop for Emergency/High categories before sending automated patient-facing push notifications.

if __name__ == '__main__':
    import uvicorn
    print('Run: uvicorn fastapi_pocket_doctor_universal:app --reload --port 8000')
    uvicorn.run('fastapi_pocket_doctor_universal:app', host='127.0.0.1', port=8000, log_level='info')
