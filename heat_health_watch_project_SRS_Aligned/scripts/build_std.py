"""Generate docs/STD.md test-case table from a real pytest JUnit XML run.

Usage (from project root):
    python -m pytest -q -p no:cacheprovider --junitxml=docs/test_results.xml
    python scripts/build_std.py
"""
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FEATURE = {
    "test_data": ("Data loading, validation, export", "FR1/FR8/FR9"),
    "test_risk_scoring": ("Risk score, 5 categories, advisories", "FR3/FR6"),
    "test_classifier": ("Random Forest classification", "FR4"),
    "test_forecasting": ("Future forecasting", "NEW-1"),
    "test_synthetic_data": ("Synthetic scenarios", "NEW-3"),
    "test_hourly": ("Hourly simulation", "NEW-2"),
    "test_app": ("Dashboard (headless AppTest)", "FR2/FR5/FR7"),
}
root = ET.parse(ROOT / "docs" / "test_results.xml").getroot()
suite = root if root.tag == "testsuite" else root.find("testsuite")
cases = []
for tc in suite.iter("testcase"):
    mod = tc.get("classname").split(".")[-1]
    status = "Passed"
    if tc.find("failure") is not None or tc.find("error") is not None:
        status = "FAILED"
    elif tc.find("skipped") is not None:
        status = "Skipped"
    cases.append((mod, tc.get("name"), status, float(tc.get("time", 0))))
total = len(cases)
passed = sum(c[2] == "Passed" for c in cases)
failed = sum(c[2] == "FAILED" for c in cases)
skipped = sum(c[2] == "Skipped" for c in cases)
rows = []
for i, (mod, name, st, t) in enumerate(cases, 1):
    feat, req = FEATURE.get(mod, (mod, "-"))
    human = re.sub(r"^test_", "", name).replace("_", " ")
    rows.append(f"| TC-{i:03d} | {feat} | {req} | `{mod}.py::{name}` | {human} | {st} |")
tpl = (ROOT / "scripts" / "std_template.md").read_text(encoding="utf-8")
out = (tpl.replace("{{TOTAL}}", str(total)).replace("{{PASSED}}", str(passed))
          .replace("{{FAILED}}", str(failed)).replace("{{SKIPPED}}", str(skipped))
          .replace("{{DURATION}}", f"{float(suite.get('time', 0)):.1f}")
          .replace("{{ROWS}}", "\n".join(rows)))
(ROOT / "docs" / "STD.md").write_text(out, encoding="utf-8")
print(f"STD.md written: {total} cases, {passed} passed, {failed} failed, {skipped} skipped")
