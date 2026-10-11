"""Check that the numbers in an AI-written explanation really come from the optimizer and forecast."""
import re

_NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?")


def numbers_in(text):
    out = []
    for m in _NUM.finditer(text):
        raw = m.group(1).replace(",", "") + (m.group(2) or "")
        out.append(float(raw))
    return out


def check(text, allowed, rel_tol=0.02, abs_tol=0.6):
    """Return {'ok', 'checked', 'unmatched'}.

    Numbers below 10 with no decimal point (list numbers, '3 days') are skipped because they are usually
    wording, not claims. Everything else must be within 2% (or 0.6) of a value in `allowed`.
    """
    vals = [float(v) for v in allowed]
    unmatched, checked = [], 0
    for x in numbers_in(text):
        if x < 10 and x == int(x):
            continue
        checked += 1
        if not any(abs(x - v) <= max(abs_tol, rel_tol * abs(v)) for v in vals):
            unmatched.append(x)
    return {"ok": not unmatched, "checked": checked, "unmatched": sorted(set(unmatched))}
