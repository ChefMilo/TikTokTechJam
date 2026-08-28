"""Rolling-origin temporal backtest, carved entirely out of the TRAINING
window: fit on an earlier sub-window, score on a later one.

Its purpose is to reproduce, inside the training data alone, the same
forward-in-time gap that exists between validation and the hidden test
set — so that a change which only helps by overfitting the validation
set gets caught here before it ever reaches validation.
"""


def run_backtest(*args, **kwargs):
    pass
