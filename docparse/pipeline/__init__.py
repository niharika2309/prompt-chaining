"""Parsing pipeline: PDF text extraction, then one of the two approaches —
monolithic (1 call) or chain (classify → extract). Each exposes parse(text, pdf_name)."""
