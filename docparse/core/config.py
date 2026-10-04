"""Central configuration for the parser comparison project."""
from pathlib import Path

# --- LLM endpoint (probed 2026-10-03) ---
BASE_URL = "http://100.117.48.99:8888/v1"
MODEL = "Qwen3.8-27B"
API_KEY = "none"  # server requires a non-empty bearer token; "none" is accepted

TEMPERATURE = 0.0
MAX_TOKENS = 4096        # output budget per call (thinking disabled)
ENABLE_THINKING = False  # Qwen3 reasoning models: chat_template_kwargs
TIMEOUT = 300            # seconds per HTTP request
MAX_RETRIES = 2          # extra attempts when JSON is invalid / fails validation
CONCURRENCY = 3          # parallel documents (single-GPU endpoint)

# --- Paths ---
ROOT = Path(__file__).resolve().parents[2]  # repo root (docparse/core/ -> root)
DATASET_DIR = ROOT / "data" / "synthetic-australian-medical-documents-sample"
PDF_DIR = DATASET_DIR / "pdfs"
GT_CSV = DATASET_DIR / "ground_truth.csv"
REPORTS_DIR = ROOT / "reports"
CORPUS_DIR = REPORTS_DIR / "corpus"
RUNS_DIR = REPORTS_DIR / "runs"

# --- Pilot set: 11 docs, stratified across 8 document families ---
PILOT_DOCS = [
    "0001_prescription_DEPRESSION_GAD_Edwards.pdf",
    "0027_prescription_THYROID_HYPER_Stavros.pdf",
    "0033_discharge_summary_CAP_PNEUMONIA_Khan.pdf",
    "0041_discharge_summary_AORTIC_STENOSIS_Conti.pdf",
    "0007_ed_assessment_CARDIOMYOPATHY_DCM_Stewart.pdf",
    "0017_ed_assessment_GASTRO_VIRAL_Stewart.pdf",
    "0003_pathology_report_GORD_BARRETT_Mehta.pdf",
    "0012_ecg_report_NSTEMI_Jenkins.pdf",
    "0006_referral_letter_COLON_CA_Powell.pdf",
    "0010_progress_note_COVID_PNEUMONIA_Green.pdf",
    "0045_mental_health_assessment_DEPRESSION_GAD_McKenzie.pdf",
]  # 2 rx, 2 discharge, 2 ed, 1 lab, 1 ecg, 1 correspondence, 1 note, 1 assessment
