"""
Anonymizer module for KVKK-compliant PII masking of interview transcripts.
Masks Turkish person names, TC Kimlik No, IBAN, and phone numbers using regex.
"""

import re
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Two or three consecutive capitalized Turkish words that look like person names.
# Requirements:
#  - Each word starts with a Turkish uppercase letter, followed by lowercase letters.
#  - Minimum 2 characters per word (avoids single-letter initials).
#  - Not immediately followed by a colon (which indicates a document label/header).
#  - Not an all-caps abbreviation.
_NAME_PATTERN = re.compile(
    r'\b([A-ZÇĞİÖŞÜ][a-zçğışöüa-z]{1,} (?:[A-ZÇĞİÖŞÜ][a-zçğışöüa-z]{1,} )?[A-ZÇĞİÖŞÜ][a-zçğışöüa-z]{1,})\b(?!:)'
)

# TC Kimlik No: exactly 11 digits, not part of a longer number sequence.
_TC_PATTERN = re.compile(r'(?<!\d)(\d{11})(?!\d)')

# IBAN: TR followed by exactly 24 digits.
_IBAN_PATTERN = re.compile(r'\b(TR\d{24})\b')

# Phone: 05XX XXXXXXX or +90 5XX XXXXXXX variants (with optional spaces/dashes).
_PHONE_PATTERN = re.compile(
    r'(?<!\d)((?:\+90[\s\-]?)?0?5\d{2}[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2})(?!\d)'
)

# ---------------------------------------------------------------------------
# Known-safe number prefixes to protect from TC-pattern false-positives.
# Article references like "657" won't match 11 digits, so this is mostly safe.
# ---------------------------------------------------------------------------

def anonymize_text(text: str) -> dict:
    """
    Anonymize PII in a Turkish interview transcript for KVKK compliance.

    Applies regex substitutions in the following order:
      1. IBAN  (TR + 24 digits)
      2. TC Kimlik No  (11 digits)
      3. Phone numbers  (05XX or +90 patterns)
      4. Turkish person names  (2-3 capitalized word sequences)

    Parameters
    ----------
    text : str
        Raw transcript text that may contain PII.

    Returns
    -------
    dict with keys:
        - "anonymized_text": str — PII-redacted version of the input.
        - "mask_log": list[dict] — each substitution recorded as
          {"original": str, "replaced_with": str}.
    """
    mask_log = []
    result = text

    def _replace(pattern: re.Pattern, placeholder: str, current_text: str) -> str:
        """Apply a single pattern substitution and record every match."""
        def _sub(match: re.Match) -> str:
            original = match.group(0)
            mask_log.append({"original": original, "replaced_with": placeholder})
            logger.info("Maskelendi: '%s' → '%s'", original, placeholder)
            return placeholder

        return pattern.sub(_sub, current_text)

    # Order matters:
    # 1. IBAN first (TR + 24 digits — avoids TC false match on long runs)
    # 2. Phone before TC (05XXXXXXXXX is 11 digits and would match TC pattern)
    # 3. TC after phone
    # 4. Names last
    result = _replace(_IBAN_PATTERN, "[IBAN]", result)
    result = _replace(_PHONE_PATTERN, "[TELEFON]", result)
    result = _replace(_TC_PATTERN, "[TC-KİMLİK]", result)
    result = _replace(_NAME_PATTERN, "[İSİM]", result)

    logger.info("Anonimleştirme tamamlandı. Toplam %d kayıt maskelendi.", len(mask_log))

    return {
        "anonymized_text": result,
        "mask_log": mask_log,
    }


def test_anonymizer() -> None:
    """
    Run anonymizer on the raw mock interview transcript and print the mask_log.

    Reads data/tacit_interview_raw.txt relative to this file's location,
    runs anonymize_text(), writes the result to processed/tacit_interview_clean.txt,
    and prints every substitution to the console for manual verification.
    """
    base_dir = Path(__file__).resolve().parent.parent
    raw_path = base_dir / "data" / "tacit_interview_raw.txt"
    out_path = base_dir / "processed" / "tacit_interview_clean.txt"

    if not raw_path.exists():
        logger.error("Ham transkript bulunamadı: %s", raw_path)
        return

    raw_text = raw_path.read_text(encoding="utf-8")
    logger.info("Ham dosya okundu: %s (%d karakter)", raw_path.name, len(raw_text))

    result = anonymize_text(raw_text)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result["anonymized_text"], encoding="utf-8")
    logger.info("Anonimleştirilmiş metin yazıldı: %s", out_path)

    print("\n" + "=" * 60)
    print("ANONİMLEŞTİRME KAYIT LOGU")
    print("=" * 60)
    for i, entry in enumerate(result["mask_log"], start=1):
        print(f"{i:3}. '{entry['original']}' → '{entry['replaced_with']}'")
    print("=" * 60)
    print(f"Toplam maskelenen kayıt: {len(result['mask_log'])}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    test_anonymizer()
