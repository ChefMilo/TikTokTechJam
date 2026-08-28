"""Validation of the SUBMISSION FILE, not of splits — split integrity
(train/val/test leakage) is data.py's responsibility, checked at split
construction time.

Will hold: a wrapper around the organizer's `submit.py --check`, plus
extra assertions for NaN/Inf scores and correct row count before a
submission is produced.
"""


def validate_submission(*args, **kwargs):
    pass
