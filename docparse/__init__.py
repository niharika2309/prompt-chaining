"""Monolithic long prompt vs prompt chaining for medical document parsing.

Layout:
  core/        shared infrastructure — config, output schema, prompts, LLM client
  pipeline/    per-approach parsing — text extraction, monolithic, 2-step chain
  evaluation/  scoring vs ground truth, paired statistics, canonical table
  reporting/   markdown + HTML side-by-side reports

Entry points (repo root): main.py (CLI), app.py (Streamlit viewer).
"""
