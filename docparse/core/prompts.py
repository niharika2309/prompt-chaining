"""Prompt building blocks shared by BOTH approaches (control variable: structure only).

Both approaches use the same persona, the same output rules, and the same
field descriptions. The ONLY differences are:
  - monolithic: one call, spec covers L1 + L2 + ALL 12 family detail blocks
  - chain:      classify -> extract (spec covers ONLY the selected family)

The spec renders the NATIVE ParsedDocument shape (flat top level, no wrapper):
{
  "document_type": ..., "document_family": ..., ...,
  "patient": {...}, "facility": {...}, "clinical": {...},
  "rx" | "lab" | ... : {...}, "key_values": {...}
}
Validators tolerate a model-side "document" wrapper regardless.
"""
from .schema import (PatientBlock, FacilityBlock, ClinicalBlock, DOC_TYPES, FAMILIES)

PERSONA = (
    "You are a domain expert in medical documents, an AI assistant that converts "
    "raw medical document text into structured JSON. Your responses must be "
    "informed, precise, and factually accurate. You never fabricate data or "
    "opinions: a field is filled ONLY when the document explicitly states it."
)

RULES = """\
Output rules (strict):
- Reply with a single valid JSON object matching the specification exactly.
  No markdown fences, no commentary, no text before or after.
- Copy values VERBATIM from the document (names, numbers, codes, dates stay exactly
  as printed, including date formats).
- If a field is not stated in the document, set it to null. Never guess, infer, or
  complete a value the document does not contain.
- For list fields, use an empty array [] when the document has no such items.
- Use ONLY the fields shown in the specification; do not add any other fields.
- The "SYNTHETIC TRAINING DOCUMENT" footers and dataset/batch metadata
  (batch IDs, bounding boxes, scan quality notes) are NOT document content: ignore them."""

PATIENT_DESC = {
    "full_name": "the patient's full name as printed",
    "dob": "date of birth, verbatim",
    "age": "age in years as an integer, if printed",
    "sex": "sex/gender as printed",
    "address": "home address, verbatim",
    "phone": "contact phone number, verbatim",
    "occupation": "occupation if printed",
    "mrn": "medical record number, verbatim",
    "medicare": "medicare number, verbatim",
    "nok_name": "next-of-kin name if printed",
    "nok_relationship": "next-of-kin relationship if printed",
    "allergies": "allergies as listed, verbatim",
}

FACILITY_DESC = {
    "name": "name of the clinic/hospital/facility issuing the document",
    "lhd": "local health district (LHD) if printed",
    "ward": "ward or unit if printed",
    "provider": "the primary clinician this document is written by/for (e.g. prescribing doctor for a prescription, pathologist for a pathology report, radiologist for imaging, ED clinician for an ED assessment), verbatim including title",
}

CLINICAL_DESC = {
    "specialty": "the medical specialty named in the document, verbatim",
    "principal_diagnosis": "the principal diagnosis/problem as stated, verbatim",
    "icd_codes": "each ICD-10 code printed on the document (e.g. F32.1), one per array item; [] if none",
    "additional_diagnoses": "each additional/secondary diagnosis listed, verbatim, one per array item; [] if none",
}

FAMILY_DESC = {
    "rx": "medication detail block — populate for prescriptions, fluid orders, medication charts",
    "lab": "pathology detail block — populate for pathology requests/reports",
    "imaging": "imaging detail block — populate for imaging/endoscopy/ultrasound reports",
    "ecg": "ECG detail block — populate for ECG reports",
    "ed": "ED detail block — populate for ED assessments and ambulance records",
    "discharge": "discharge detail block — populate for discharge summaries",
    "note": "clinical note detail block — populate for progress notes and anaesthetic records",
    "correspondence": "letter detail block — populate for referrals and correspondence",
    "assessment": "specialist assessment detail block — populate for specialist assessment documents",
    "checklist": "checklist detail block — populate for checklist documents",
    "certificate": "certificate detail block — populate for medical certificates",
    "consent": "consent detail block — populate for consent forms",
}


def _object(label: str, fields: list[tuple[str, str]], pad: str = "  ") -> str:
    """Render a JSON object block with consistent indentation."""
    lines = [f'{pad}"{label}": {{']
    for i, (name, desc) in enumerate(fields):
        comma = "," if i < len(fields) - 1 else ""
        lines.append(f'{pad}  "{name}": "{desc}"{comma}')
    lines.append(pad + "}")
    return "\n".join(lines)


def _fields_of(model_cls, desc: dict) -> list[tuple[str, str]]:
    out = []
    for name in model_cls.model_fields:
        out.append((name, desc.get(name, "as printed in the document; null if not stated")))
    return out


def _med_field() -> tuple:
    return ("medications",
            'array of {"name": "drug / form / strength", "dose": "dose / route / frequency / quantity / repeats"}, verbatim; [] if none')


def family_spec_block(family: str) -> str:
    """JSON-spec for one L3 family detail block (fields from schema.py)."""
    from .schema import (RxDetail, LabDetail, ImagingDetail, EcgDetail, EdDetail,
                         DischargeDetail, NoteDetail, CorrespondenceDetail,
                         AssessmentDetail, ChecklistDetail, CertificateDetail,
                         ConsentDetail)

    blocks = {
        "rx": RxDetail, "lab": LabDetail, "imaging": ImagingDetail, "ecg": EcgDetail,
        "ed": EdDetail, "discharge": DischargeDetail, "note": NoteDetail,
        "correspondence": CorrespondenceDetail, "assessment": AssessmentDetail,
        "checklist": ChecklistDetail, "certificate": CertificateDetail,
        "consent": ConsentDetail,
    }
    if family == "other":
        return None
    cls = blocks[family]
    fields = []
    for name in cls.model_fields:
        if name == "medications":
            fields.append(_med_field())
        else:
            fields.append((name, "as printed; null if not stated"))
    return _object(family, fields, pad="  ")


def full_spec(families: list[str]) -> str:
    """Full ParsedDocument JSON spec (L1 + L2 + selected L3 blocks + catch-all)."""
    parts = ["{"]
    parts.append('  "document_type": "the specific document type, one of: '
                 + ", ".join(DOC_TYPES) + '",')
    parts.append('  "document_family": "the family, one of: '
                 + ", ".join(FAMILIES) + '",')
    parts.append('  "document_number": "the document reference/serial number printed on it (e.g. RX-2026-236006), verbatim; null if none",')
    parts.append('  "document_date": "the main date of the document as printed, verbatim; null if none",')
    parts.append(_object("patient", _fields_of(PatientBlock, PATIENT_DESC)) + ",")
    parts.append(_object("facility", _fields_of(FacilityBlock, FACILITY_DESC)) + ",")
    parts.append(_object("clinical", _fields_of(ClinicalBlock, CLINICAL_DESC)) + ",")

    fams = [f for f in families if f != "other"]
    for f in fams:
        parts.append(family_spec_block(f) + ",")
    if not fams:
        parts.append('  "_hint": "no family detail block matches this document type - capture any remaining structured fields in key_values",')
    parts.append('  "key_values": {"any other structured field on the document not covered above": "its value, verbatim"}')
    parts.append("}")
    return "\n".join(parts)


def s1_classification_spec() -> str:
    return (
        "{\n"
        f'  "document_type": "one of: {", ".join(DOC_TYPES)}",\n'
        f'  "document_family": "one of: {", ".join(FAMILIES)}",\n'
        '  "confidence": 0.0,\n'
        '  "basis": "one short sentence citing what in the text identifies the document"\n'
        "}"
    )


