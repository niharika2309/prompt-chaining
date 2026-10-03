"""Pydantic output schema, shared by BOTH parser approaches (controlled experiment).

Derivation (documented for reproducibility)
--------------------------------------------
1. Inventory  = the dataset's ground_truth.csv (115 columns) — the dataset
   author's own definition of "a complete extraction of these documents".
2. Layering   = presence-rate analysis across the 50 docs:
                   - L1 identity : present in 50/50 docs  -> every prompt
                   - L2 clinical : present in ~47/50 docs -> every prompt
                   - L3 detail   : the long tail, clustered into 12 document
                                   families; a field becomes *typed* when it is
                                   populated in >=50% of its family's documents,
                                   or by domain judgement when it is the family's
                                   core payload (e.g. `medications` for rx).
   Dataset-metadata columns (batch_seq, pdf_filename, scan_quality, bboxes,
   patient surname/given-name splits) are NOT extractable and are excluded.
3. Long-tail  = rarer family fields live in the `key_values` catch-all:
   never silently dropped, but reported rather than strictly scored.
"""
from pydantic import BaseModel, ConfigDict

# The 29 document types in the sample (from manifest.json / README)
DOC_TYPES = [
    "ed_assessment", "discharge_summary", "prescription", "referral_letter",
    "progress_note", "medication_chart", "pathology_report", "physiotherapy_assessment",
    "imaging_report", "vascular_ultrasound_report", "ecg_report", "fluid_order",
    "mental_health_assessment", "anaesthetic_record", "ambulance_record",
    "audiology_assessment", "ophthalmology_assessment", "speech_pathology_assessment",
    "endoscopy_report", "ecg_12lead", "pathology_request", "medical_certificate",
    "consent_for_treatment", "admission_checklist", "patient_safety_checklist",
    "correct_patient_checklist", "infusion_pump_checklist", "internal_correspondence",
    "external_correspondence",
]

FAMILIES = [
    "rx", "lab", "imaging", "ecg", "ed", "discharge",
    "note", "correspondence", "assessment", "checklist", "certificate", "consent", "other",
]

TYPE_TO_FAMILY = {
    "prescription": "rx", "fluid_order": "rx", "medication_chart": "rx",
    "pathology_request": "lab", "pathology_report": "lab",
    "imaging_report": "imaging", "endoscopy_report": "imaging",
    "vascular_ultrasound_report": "imaging",
    "ecg_report": "ecg", "ecg_12lead": "ecg",
    "ed_assessment": "ed", "ambulance_record": "ed",
    "discharge_summary": "discharge", "admission_checklist": "discharge",
    "progress_note": "note", "anaesthetic_record": "note",
    "referral_letter": "correspondence", "internal_correspondence": "correspondence",
    "external_correspondence": "correspondence",
    "physiotherapy_assessment": "assessment", "mental_health_assessment": "assessment",
    "audiology_assessment": "assessment", "ophthalmology_assessment": "assessment",
    "speech_pathology_assessment": "assessment",
    "patient_safety_checklist": "checklist", "correct_patient_checklist": "checklist",
    "infusion_pump_checklist": "checklist",
    "medical_certificate": "certificate",
    "consent_for_treatment": "consent",
}

FAMILY_DOCS = {
    "rx": "prescription / fluid order / medication chart",
    "lab": "pathology request / pathology report",
    "imaging": "imaging report / endoscopy report / vascular ultrasound report",
    "ecg": "ECG report / 12-lead ECG",
    "ed": "emergency department assessment / ambulance record",
    "discharge": "discharge summary / admission checklist",
    "note": "progress note / anaesthetic record",
    "correspondence": "referral letter / internal or external correspondence",
    "assessment": "specialist assessment (physiotherapy, mental health, audiology, ophthalmology, speech pathology)",
    "checklist": "patient safety / correct patient / infusion pump checklist",
    "certificate": "medical certificate",
    "consent": "consent for treatment form",
    "other": "any other document",
}


class _S(BaseModel):
    # All schema fields default to None: "not stated in the document" -> null.
    model_config = ConfigDict(extra="ignore")


class PatientBlock(_S):
    full_name: str | None = None
    dob: str | None = None            # date of birth, as printed
    age: int | None = None
    sex: str | None = None
    address: str | None = None
    phone: str | None = None
    occupation: str | None = None
    mrn: str | None = None            # medical record number
    medicare: str | None = None
    nok_name: str | None = None       # next of kin
    nok_relationship: str | None = None
    allergies: str | None = None


class FacilityBlock(_S):
    name: str | None = None           # clinic / hospital / facility name
    lhd: str | None = None            # local health district
    ward: str | None = None
    provider: str | None = None       # the primary clinician named for this document type


class ClinicalBlock(_S):
    specialty: str | None = None
    principal_diagnosis: str | None = None
    icd_codes: list[str] = []         # ICD-10 codes printed on the document
    additional_diagnoses: list[str] = []


class Medication(_S):
    name: str | None = None           # drug / form / strength
    dose: str | None = None           # dose / route / frequency (+ quantity, repeats)


# ---------------- L3 family detail blocks ----------------

class RxDetail(_S):
    medications: list[Medication] = []
    new_medications: list[str] = []
    chart_period: str | None = None
    prescriber: str | None = None


class LabDetail(_S):
    lab_name: str | None = None
    lab_ref: str | None = None
    specimen_type: str | None = None
    specimen_date: str | None = None
    reported_date: str | None = None
    pathologist: str | None = None
    tests_requested: list[str] = []
    results: str | None = None


class ImagingDetail(_S):
    modality: str | None = None
    examination: str | None = None
    accession_no: str | None = None
    exam_date: str | None = None
    reported_date: str | None = None
    requesting_clinician: str | None = None
    radiologist: str | None = None
    report_priority: str | None = None
    procedure: str | None = None
    findings: str | None = None


class EcgDetail(_S):
    ecg_rate: str | None = None
    ecg_rhythm: str | None = None
    ecg_qrs_ms: str | None = None
    ecg_qtc_ms: str | None = None
    accession_no: str | None = None
    exam_date: str | None = None
    reported_date: str | None = None
    requesting_clinician: str | None = None
    radiologist: str | None = None
    findings: str | None = None


class EdDetail(_S):
    triage_category: str | None = None
    arrival_datetime: str | None = None
    arrival_mode: str | None = None
    departure_datetime: str | None = None
    disposition: str | None = None
    presentation_no: str | None = None
    ed_clinician: str | None = None
    ambulance_priority: str | None = None
    destination_hospital: str | None = None
    findings: str | None = None


class DischargeDetail(_S):
    admission_date: str | None = None
    discharge_date: str | None = None
    length_of_stay: str | None = None
    discharge_destination: str | None = None
    procedures: list[str] = []
    medications: list[Medication] = []
    day_of_admission: str | None = None


class NoteDetail(_S):
    note_date: str | None = None
    day_of_admission: str | None = None
    assessment: str | None = None
    plan: str | None = None
    anaesthetic_type: str | None = None
    asa_status: str | None = None
    operation_date: str | None = None
    duration_minutes: str | None = None
    procedure: str | None = None


class CorrespondenceDetail(_S):
    letter_date: str | None = None
    recipient: str | None = None
    subject: str | None = None
    priority: str | None = None
    gp_name: str | None = None
    gp_clinic: str | None = None
    specialist_name: str | None = None
    specialist_role: str | None = None
    specialist_clinic: str | None = None


class AssessmentDetail(_S):
    requesting_clinician: str | None = None
    gad7: str | None = None
    phq9: str | None = None
    risk_level: str | None = None
    hl_classification: str | None = None
    boston_naming_test: str | None = None
    diet_level: str | None = None
    wab_aphasia_quotient: str | None = None
    iop_left: str | None = None
    iop_right: str | None = None
    va_left: str | None = None
    va_right: str | None = None
    pta_left: str | None = None
    pta_right: str | None = None
    findings: str | None = None


class ChecklistDetail(_S):
    procedure: str | None = None
    drug_infused: str | None = None
    pump_make_model: str | None = None
    checklist_items: list[str] = []


class CertificateDetail(_S):
    certificate_type: str | None = None
    unfit_from: str | None = None
    unfit_to: str | None = None
    unfitness_days: str | None = None


class ConsentDetail(_S):
    procedure: str | None = None
    anaesthetic_type: str | None = None
    consent_status: str | None = None


class ParsedDocument(_S):
    # L1 — identity (all documents)
    document_type: str = "unknown"
    document_family: str = "other"
    document_number: str | None = None
    document_date: str | None = None
    patient: PatientBlock = PatientBlock()
    facility: FacilityBlock = FacilityBlock()
    # L2 — clinical core (all documents)
    clinical: ClinicalBlock = ClinicalBlock()
    # L3 — exactly one family block is populated (selected by document_family)
    rx: RxDetail | None = None
    lab: LabDetail | None = None
    imaging: ImagingDetail | None = None
    ecg: EcgDetail | None = None
    ed: EdDetail | None = None
    discharge: DischargeDetail | None = None
    note: NoteDetail | None = None
    correspondence: CorrespondenceDetail | None = None
    assessment: AssessmentDetail | None = None
    checklist: ChecklistDetail | None = None
    certificate: CertificateDetail | None = None
    consent: ConsentDetail | None = None
    # Long-tail catch-all: field name -> value, for rarer family fields
    key_values: dict[str, str] = {}

    def detail_block(self) -> _S | None:
        for fam in ("rx", "lab", "imaging", "ecg", "ed", "discharge", "note",
                    "correspondence", "assessment", "checklist", "certificate", "consent"):
            block = getattr(self, fam)
            if block is not None:
                return block
        return None

    def detail_family(self) -> str:
        for fam in ("rx", "lab", "imaging", "ecg", "ed", "discharge", "note",
                    "correspondence", "assessment", "checklist", "certificate", "consent"):
            if getattr(self, fam) is not None:
                return fam
        return "other"
