"""Evaluation: compare parser output against ground_truth.csv + source text.

Field mapping (parser field -> GT column(s)):
  - L1/L2 fields map the same way for every document family.
  - L3 detail fields map per family; a None target means "no GT column exists,
    reported but not scored" (e.g. free-text `findings`).
  - `facility.name` and per-family `provider` use ACCEPT-ANY-OF column lists.
  - Dataset metadata columns (batch ids, scan quality, bboxes, surname/given
    splits) are never scored: they are not extractable from the document.

Verdicts per scored pair (model value m, GT value g):
  abstain  g null, m null          (correctly not fabricated)
  correct  g non-null, values match (normalized exact, or F1 >= 0.99 for text)
  partial  g non-null, 0.6 <= F1 < 0.99
  wrong    g non-null, values differ
  missed   g non-null, m null
  extra    g null, m non-null      (checked vs source text -> fabricated | not_in_gt)

Headline metrics:
  field_acc   (correct + abstain) / all scored pairs
  recall      correct (+0.5*partial) / pairs where GT non-null
  precision   matches / pairs where model non-null
  halluc      (model non-null AND value absent from source text) / model non-null
  clean_docs  docs with zero wrong/fabricated/extra values
"""
import csv
import re
from collections import defaultdict
from datetime import datetime

from ..core import config
from ..core import schema

STOP = {
    "the", "and", "for", "with", "from", "this", "that", "was", "were", "has",
    "had", "not", "per", "on", "in", "of", "to", "a", "an", "by", "at", "as",
}

# ---------------- normalization ----------------

def norm(s) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().casefold()


def _dates(s: str):
    s = str(s).strip()
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%y", "%d %b %Y",
                "%d %B %Y", "%d-%m-%y %H:%M", "%d/%m/%Y %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def norm_val(v) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return " ".join(norm(x) for x in v)
    if isinstance(v, dict):
        return " ".join(f"{k}={x}" for k, x in v.items())
    s = norm(v)
    d = _dates(s)
    return d.strftime("%Y-%m-%d") if d else s


def tokens(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9%.]+", str(s).casefold()) if t not in STOP and len(t) > 2]


def f1(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    ca, cb = defaultdict(int), defaultdict(int)
    for t in ta:
        ca[t] += 1
    for t in tb:
        cb[t] += 1
    inter = sum(min(ca[t], cb[t]) for t in ca.keys() | cb.keys())
    if inter == 0:
        return 0.0
    prec = inter / len(ta)
    rec = inter / len(tb)
    return 2 * prec * rec / (prec + rec)


def source_support(value: str, source_tokens: set[str]) -> bool:
    """Is the value actually present in the source text? (anti-hallucination check)"""
    vt = [t for t in tokens(value) if len(t) > 2]
    if not vt:
        return True
    hit = sum(1 for t in vt if t in source_tokens)
    return hit / len(vt) >= 0.5


# ---------------- field mapping ----------------

PROVIDER_COLS = {
    "rx": ["consultant_name", "gp_name", "registrar_name"],
    "lab": ["pathologist", "requesting_clinician"],
    "imaging": ["radiologist", "requesting_clinician"],
    "ecg": ["radiologist", "requesting_clinician", "consultant_name"],
    "ed": ["ed_clinician"],
    "discharge": ["consultant_name", "registrar_name", "gp_name"],
    "note": ["registrar_name", "consultant_name"],
    "correspondence": ["gp_name", "specialist_name"],
    "assessment": ["examiner_name", "requesting_clinician"],
    "checklist": ["examiner_name"],
    "certificate": ["examiner_name"],
    "consent": ["examiner_name"],
    "other": [],
}

FAMILY_DATE_COL = {
    "rx": None, "lab": "reported_date", "imaging": "exam_date", "ecg": "exam_date",
    "ed": "arrival_datetime", "discharge": None,  # discharge: either date accepted below
    "note": "note_date", "correspondence": "letter_date", "assessment": None,
    "checklist": "exam_date", "certificate": "exam_date", "consent": "exam_date", "other": None,
}

L3_MAP = {
    "rx": {"medications": "medications", "new_medications": "new_medications",
           "chart_period": "chart_period_start", "prescriber": None},
    "lab": {"lab_name": "lab_name", "lab_ref": "lab_ref", "specimen_type": "specimen_type",
            "specimen_date": "specimen_date", "reported_date": "reported_date",
            "pathologist": "pathologist", "tests_requested": "tests_requested",
            "results": None},
    "imaging": {"modality": "modality", "examination": "examination",
                "accession_no": "accession_no", "exam_date": "exam_date",
                "reported_date": "reported_date", "requesting_clinician": "requesting_clinician",
                "radiologist": "radiologist", "report_priority": "report_priority",
                "procedure": "procedure", "findings": None},
    "ecg": {"ecg_rate": "ecg_rate", "ecg_rhythm": "ecg_rhythm", "ecg_qrs_ms": "ecg_qrs_ms",
            "ecg_qtc_ms": "ecg_qtc_ms", "accession_no": "accession_no", "exam_date": "exam_date",
            "reported_date": "reported_date", "requesting_clinician": "requesting_clinician",
            "radiologist": "radiologist", "findings": None},
    "ed": {"triage_category": "triage_category", "arrival_datetime": "arrival_datetime",
           "arrival_mode": "arrival_mode", "departure_datetime": "departure_datetime",
           "disposition": "disposition", "presentation_no": "presentation_no",
           "ed_clinician": "ed_clinician", "ambulance_priority": "ambulance_priority",
           "destination_hospital": "destination_hospital", "findings": None},
    "discharge": {"admission_date": "admission_date", "discharge_date": "discharge_date",
                  "length_of_stay": "length_of_stay_days",
                  "discharge_destination": "discharge_destination",
                  "procedures": "procedure", "medications": "medications",
                  "day_of_admission": "day_of_admission"},
    "note": {"note_date": "note_date", "day_of_admission": "day_of_admission",
             "assessment": None, "plan": None, "anaesthetic_type": "anaesthetic_type",
             "asa_status": "asa_status", "operation_date": "operation_date",
             "duration_minutes": "duration_minutes", "procedure": "procedure"},
    "correspondence": {"letter_date": "letter_date", "recipient": "recipient",
                       "subject": "subject", "priority": "priority", "gp_name": "gp_name",
                       "gp_clinic": "gp_clinic", "specialist_name": "specialist_name",
                       "specialist_role": "specialist_role",
                       "specialist_clinic": "specialist_clinic"},
    "assessment": {"requesting_clinician": "requesting_clinician", "gad7": "gad7",
                   "phq9": "phq9", "risk_level": "risk_level",
                   "hl_classification": "hl_classification",
                   "boston_naming_test": "boston_naming_test", "diet_level": "diet_level",
                   "wab_aphasia_quotient": "wab_aphasia_quotient",
                   "iop_left": "iop_left_mmHg", "iop_right": "iop_right_mmHg",
                   "va_left": "va_left", "va_right": "va_right",
                   "pta_left": "pta_left_db", "pta_right": "pta_right_db",
                   "findings": None},
    "checklist": {"procedure": "procedure", "drug_infused": "drug_infused",
                  "pump_make_model": "pump_make_model", "checklist_items": None},
    "certificate": {"certificate_type": "certificate_type", "unfit_from": "unfit_from",
                    "unfit_to": "unfit_to", "unfitness_days": "unfitness_days"},
    "consent": {"procedure": "procedure", "anaesthetic_type": "anaesthetic_type",
                "consent_status": None},
    "other": {},
}


def load_gt(pdf_names: list[str]) -> dict[str, dict]:
    out = {}
    with open(config.GT_CSV) as f:
        for row in csv.DictReader(f):
            if row["pdf_filename"] in pdf_names:
                out[row["pdf_filename"]] = row
    return out


def _pair(name: str, m, g, source_tokens: set[str], free_text: bool = False) -> dict:
    """Score one (model, gt) value pair."""
    g_null = g in (None, "")
    m_null = m is None or (isinstance(m, str) and not m.strip()) or m == []
    if g_null and m_null:
        return {"field": name, "verdict": "abstain"}
    if g_null:
        v = norm_val(m)
        fab = not source_support(v, source_tokens)
        return {"field": name, "verdict": "extra", "fabricated": fab,
                "m": v, "g": ""}
    if m_null:
        return {"field": name, "verdict": "missed", "m": "", "g": norm_val(g)}
    mv, gv = norm_val(m), norm_val(g)
    if isinstance(m, list):
        # set-style compare for code/diagnosis lists
        sm, sg = set(tokens(mv)), set(tokens(gv))
        score = (len(sm & sg) / len(sm | sg)) if (sm | sg) else 1.0
        verdict = "correct" if score >= 0.9 else ("partial" if score >= 0.5 else "wrong")
    elif free_text or len(tokens(mv)) + len(tokens(gv)) > 6:
        score = f1(mv, gv)
        verdict = "correct" if score >= 0.9 else ("partial" if score >= 0.6 else "wrong")
    else:
        score = 1.0 if mv == gv else 0.0
        verdict = "correct" if mv == gv else "wrong"
    return {"field": name, "verdict": verdict, "score": round(score, 3),
            "m": mv[:120], "g": gv[:120]}


def eval_record(rec: dict, gt: dict, source_text: str) -> dict:
    """Score one parser record. Returns field verdicts + headline numbers."""
    final = rec["final"] or {}
    src_tokens = set(tokens(source_text))
    fam = final.get("document_family") or "other"
    if fam not in schema.FAMILIES:
        fam = "other"
    gt_fam = schema.TYPE_TO_FAMILY.get(gt.get("document_type", ""), "other")

    verdicts = []

    def na(name, m, gval=""):
        """Excluded-from-scoring field: shown in the report, not counted in metrics.

        Exclusion reasons (documented):
          - patient.sex     : GT assigns sex but NO document in the sample prints it
          - icd_codes       : GT populated, but most documents never print a code;
                              scoring would penalize correct abstention
          - specialty       : GT holds the CASE specialty; documents print the ISSUING
                              department, and verbatim-copy rules forbid inference
          - document_number : GT document_id is the synthetic-library BATCH ID from a
                              footer the model is explicitly told to ignore
        """
        verdicts.append({"field": name, "verdict": "n/a",
                         "m": norm_val(m)[:120] if m not in (None, "", []) else "",
                         "g": norm_val(gval)[:120] if gval not in (None, "") else ""})

    def add(m, g, name, free_text=False, accept_any=False, raw=False):
        """Score one field. `g` is a GT column name (or list of acceptable columns),
        or, with raw=True, the GT value itself. g=None -> unscored, skipped."""
        if g is None:
            return
        if accept_any:
            # any of the acceptable columns may hold the value
            if raw:
                vals = [v for v in g if v not in (None, "")]
            else:
                vals = [gt.get(c, "") for c in g if gt.get(c, "")]
            m_null = m is None or (isinstance(m, str) and not m.strip()) or m == []
            if not vals:
                # no GT value at all: abstain if model silent, extra otherwise
                verdicts.append(_pair(name, m, "", src_tokens, free_text))
                return
            if m_null:
                # model said nothing but GT is populated -> missed
                verdicts.append(_pair(name, m, vals[0], src_tokens, free_text))
                return
            for v in vals:
                r = _pair(name, m, v, src_tokens, free_text)
                if r["verdict"] in ("correct", "partial", "abstain"):
                    verdicts.append(r)
                    return
            verdicts.append(r)  # all candidates failed -> last verdict stands
        else:
            if raw:
                gval = g if g not in (None, "") else ""
            else:
                col = g if isinstance(g, str) else (g[0] if g else None)
                gval = gt.get(col, "") if col else ""
            verdicts.append(_pair(name, m, gval, src_tokens, free_text))

    patient = final.get("patient") or {}
    facility = final.get("facility") or {}
    clinical = final.get("clinical") or {}

    add(final.get("document_type"), "document_type", "document_type")
    add(final.get("document_family"), [gt_fam], "document_family", raw=True)
    na("document_number", final.get("document_number"), gt.get("document_id"))
    date_col = FAMILY_DATE_COL.get(fam)
    if date_col:
        add(final.get("document_date"), date_col, "document_date")
    elif fam == "discharge":
        add(final.get("document_date"), ["admission_date", "discharge_date"],
            "document_date", accept_any=True)

    for mkey, gcol in [("full_name", "patient_name"), ("dob", "patient_dob"),
                       ("age", "patient_age"),
                       ("address", "patient_address"), ("phone", "patient_phone"),
                       ("occupation", "patient_occupation"), ("mrn", "mrn"),
                       ("medicare", "medicare"), ("nok_name", "nok_name"),
                       ("nok_relationship", "nok_relationship"),
                       ("allergies", "allergies")]:
        add(patient.get(mkey), gcol, f"patient.{mkey}")
    na("patient.sex", patient.get("sex"), gt.get("patient_sex"))  # never printed on docs

    # NOTE: family-driven lookups below follow the GT family (gt_fam), not the
    # family the model declared — the scored field inventory must be fixed per
    # document so both approaches are measured on the same denominator, even
    # when a model misclassifies the family.
    add(facility.get("name"), ["facility", "hospital_name", "gp_clinic",
                               "specialist_clinic"], "facility.name", accept_any=True)
    add(facility.get("lhd"), "hospital_lhd", "facility.lhd")
    add(facility.get("ward"), "ward", "facility.ward")
    add(facility.get("provider"), PROVIDER_COLS.get(gt_fam, []), "facility.provider",
        free_text=True, accept_any=True)

    na("clinical.specialty", clinical.get("specialty"), gt.get("specialty"))
    # GT = case specialty; docs print issuing department (scoring unfair under verbatim rules)
    dx_cols = ["principal_diagnosis", "principal_diagnosis_or_problem"] \
        if gt_fam == "correspondence" else ["principal_diagnosis"]
    add(clinical.get("principal_diagnosis"), dx_cols, "clinical.principal_diagnosis",
        free_text=True, accept_any=True)
    na("clinical.icd_codes", clinical.get("icd_codes"), gt.get("principal_icd"))
    # GT populated but most documents never print a code (penalizes correct abstention)
    add(clinical.get("additional_diagnoses"), "additional_diagnoses",
        "clinical.additional_diagnoses", free_text=True)

    # L3 detail: values come from the typed block matching the GT family, else
    # the block the model populated. Values parked in the key_values catch-all
    # (what happens after a family misclassification) do NOT count toward
    # typed fields: downstream consumers of the schema would see them as missing.
    detail = final.get(gt_fam)
    if not isinstance(detail, dict):
        detail = final.get(fam) if isinstance(final.get(fam), dict) else {}
    detail = detail or {}
    for fkey, gcol in L3_MAP.get(gt_fam, {}).items():
        if gcol is None:
            continue  # no GT column: reported in raw output, not scored
        v = detail.get(fkey)
        if isinstance(v, list) and v and isinstance(v[0], dict):  # Medication objects
            v = " ".join((x.get("name") or "") + " " + (x.get("dose") or "") for x in v)
        add(v, gcol, f"{gt_fam}.{fkey}", free_text=not isinstance(v, list))

    verdicts.sort(key=lambda r: r["field"])
    return {"pdf": rec["pdf"], "gt": rec.get("approach"), "verdicts": verdicts}


def headline(rows: list[dict]) -> dict:
    # "n/a" = documented exclusions (GT fields the document never prints, etc.)
    allv = [v for r in rows for v in r["verdicts"] if v["verdict"] != "n/a"]
    n = len(allv)
    def count(*vs):
        return sum(1 for v in allv if v["verdict"] in vs)
    gt_nonnull = sum(1 for v in allv if v["verdict"] != "abstain" and v.get("g"))
    model_nonnull = sum(1 for v in allv if v.get("m"))
    fabricated = sum(1 for v in allv if v["verdict"] == "extra" and v.get("fabricated"))
    matches = count("correct", "partial")
    return {
        "scored_fields": n,
        "field_acc": round((count("correct", "abstain")) / n, 3) if n else None,
        "recall": round((count("correct") + 0.5 * count("partial")) / gt_nonnull, 3) if gt_nonnull else None,
        "precision": round(matches / model_nonnull, 3) if model_nonnull else None,
        "hallucination_rate": round(fabricated / model_nonnull, 3) if model_nonnull else None,
        "verdict_counts": {v: count(v) for v in
                           ("abstain", "correct", "partial", "wrong", "missed", "extra")},
        "na_fields": sum(1 for r in rows for v in r["verdicts"] if v["verdict"] == "n/a"),
        "fabricated_count": fabricated,
    }


def full_metrics(records: list[dict], eval_rows: list[dict]) -> dict:
    """Headline field metrics + efficiency/robustness metrics for one approach."""
    m = headline(eval_rows)
    lat = [r["meta"].get("latency") or 0 for r in records if r.get("meta")]
    tok = [r["meta"].get("total_tokens") or 0 for r in records if r.get("meta")]
    m["mean_latency"] = round(sum(lat) / len(lat), 1) if lat else None
    m["mean_tokens"] = round(sum(tok) / len(tok)) if tok else None
    m["docs"] = len(records)
    m["parsed_ok"] = sum(1 for r in records if r.get("ok"))

    # first-try schema validity: monolithic = whole call; chain = the extract step
    if records and "s2" in str((records[0].get("meta") or {})):
        first = [1 for r in records
                 if ((r.get("meta") or {}).get("steps") or {}).get("s2_extract", {}).get("attempts") == 1]
    else:
        first = [1 for r in records if (r.get("meta") or {}).get("attempts") == 1]
    m["first_try"] = f"{len(first)}/{len(records)}" if records else "0/0"

    # clean docs: no wrong values and no values the source text does not support.
    # Source-SUPPORTED extras (GT sparsely labelled; the doc clearly prints the value)
    # are NOT counted against the model.
    dirty = set()
    for row in eval_rows:
        if any(v["verdict"] == "wrong" or (v["verdict"] == "extra" and v.get("fabricated"))
               for v in row["verdicts"]):
            dirty.add(row["pdf"])
    m["clean_docs"] = f"{len(records) - len(dirty)}/{len(records)}" if records else "0/0"
    return m
