"""Regression tests for the invariants in harness/data.py and
harness/metrics.py that must survive `python -O` (see the "explicit
raise, not assert" comments in both files). An `assert` statement is
stripped entirely under -O; these checks are correctness invariants
(split leakage, vendor/contract metric drift), not debugging aids, so
they must be plain `if ...: raise ...` — not `assert`.
"""

from pathlib import Path

import pytest

from harness import data, metrics

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_metrics_evaluate_raises_on_vendor_primary_divergence(monkeypatch):
    def fake_vendor_evaluate(user_ids, labels, scores, k=5):
        # Deliberately wrong 'primary' — everything else shaped like a
        # real vendor result.
        return {"GAUC": 0.5, f"nDCG@{k}": 0.5, "primary": 999.0, "users": 1, "rows": len(labels)}

    monkeypatch.setattr(metrics._vendor, "evaluate", fake_vendor_evaluate)

    with pytest.raises(AssertionError):
        metrics.evaluate(user_ids=[1, 1], labels=[0, 1], scores=[0.1, 0.9])


def test_data_load_raises_on_row_count_mismatch(monkeypatch):
    monkeypatch.setitem(data.EXPECTED_ROWS, "train", data.EXPECTED_ROWS["train"] + 1)

    with pytest.raises(ValueError):
        data.load("train")


def _non_comment_code(path: Path) -> str:
    """Strips full-line `#` comments and inline trailing `# ...` comments,
    so a grep for a bare `assert ` doesn't false-positive on prose in a
    comment explaining why there ISN'T one (see both files' "explicit
    raise, not assert" comments).
    """
    kept_lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("#"):
            continue
        kept_lines.append(line.split("#", 1)[0])
    return "\n".join(kept_lines)


@pytest.mark.parametrize("relative_path", ["harness/data.py", "harness/metrics.py"])
def test_no_bare_assert_outside_comments(relative_path):
    code = _non_comment_code(REPO_ROOT / relative_path)
    assert "assert " not in code, (
        f"{relative_path} contains a bare `assert` statement outside comments — "
        "this must be an explicit `if ...: raise ...` so it survives python -O"
    )
