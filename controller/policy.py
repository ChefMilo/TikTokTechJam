"""Search policies: which slot the next candidate attacks.

These are REAL COMPONENTS, not test doubles. controller/fakes.py holds
stand-ins for collaborators another workstream will eventually write; this
module holds the Controller's own search policy, which is W2's to build and
nobody else's. A UniformPolicy in fakes.py would be a category error - the
Controller is not pretending to have a policy while waiting for one, it has
one.

WHY UNIFORM IS THE FIRST IMPLEMENTATION
---------------------------------------
The next PR adds a cost-aware bandit that scores slots on realized
delta-per-1k-tokens. A bandit is only a result if you can say what it beat.
UniformPolicy is that comparison: same Controller, same doubles, same seeds,
same budget, one line different in the wiring. Without it in the repo from
the start, the bandit ships as an assertion that it helps, and the honest
version of that claim - "it produced X% more accepted delta per token than
uniform slot selection over N seeded runs" - is not available at all,
because nobody kept the baseline around to run.

So uniform is not a placeholder to be deleted when the real one lands. It
is the ablation arm, and it stays.

DETERMINISM
-----------
Every policy here draws from an instance-owned `random.Random`, never the
global `random` module, and nothing here does I/O, sleeps or reads the
clock. Two runs with the same seed pull the same sequence of arms. That is
what makes an ablation reproducible: if replaying a policy could produce a
different sequence, a difference between two policies would be
indistinguishable from a difference between two runs of either one.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

from contracts import SlotName

__all__ = ["FixedOrderPolicy", "UniformPolicy"]


class UniformPolicy:
    """Picks uniformly at random from the offered slots. The ablation baseline.

    THE INSTRUMENT, NOT THE STOPGAP. See this module's docstring: the
    cost-aware bandit in the next PR is measured against this on
    delta-per-1k-tokens, so it has to exist, be seeded, and stay.

    It is also the right *default* for an early run, which is worth saying
    out loud. With no history to learn from, a policy that concentrates on
    one arm is not exploiting knowledge, it is guessing with extra steps.
    Uniform spends the first few evaluations spreading coverage across
    slots, which is exactly the data a bandit needs before it has anything
    to be greedy about.

    WHAT IT POINTEDLY DOES NOT DO: it ignores `state_card` entirely - no
    peeking at the budget, the history or the convergence streak. That is
    the definition of the baseline arm. A "uniform policy, but it skips
    slots that failed twice" is not a control, it is an untested heuristic
    wearing the control's name, and every later measurement would be
    against a moving target.
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self._rng = random.Random(seed)
        """Instance-owned, never the global `random` module.

        The global module's stream is shared with every other caller in the
        process - a test that seeds it, a library that draws from it, an
        import that happens to run first. A policy drawing from it would
        produce a different sequence of arms depending on what else the
        process did, which would make an ablation comparison meaningless
        and a failing run impossible to reproduce.
        """

    def select_slot(
        self, state_card: Mapping[str, Any], candidate_slots: Sequence[SlotName]
    ) -> SlotName:
        """One slot, drawn uniformly. Never anything outside `candidate_slots`.

        `random.Random.choice` rather than `randrange` + index, so the
        draw is over the offered sequence itself and there is no arithmetic
        between the RNG and the returned value that could put it out of
        range.
        """
        if not candidate_slots:
            # The Controller ends a stage before it gets here, so this is a
            # caller bug rather than a run condition. Named explicitly
            # because `random.choice` on an empty sequence raises
            # IndexError, which says nothing about what went wrong.
            raise ValueError("UniformPolicy was offered no candidate slots")
        return self._rng.choice(tuple(candidate_slots))


class FixedOrderPolicy:
    """Cycles through a fixed slot order. No randomness at all.

    Two jobs, both real.

    AS A TEST INSTRUMENT: a test that wants to assert "the Controller
    passed `weighting` to the generator" cannot do that against a random
    policy without either seeding it and hardcoding the draw (brittle - the
    expected value changes if the candidate set changes) or asserting only
    that the result is a member of the set (weak). Naming the order makes
    the expected value obvious from the fixture.

    AS A DEFAULT: it is a legitimate policy in its own right - round-robin
    over the arms, which spreads coverage as evenly as uniform does but
    without a seed to carry around and with a reproducible order rather
    than merely a reproducible distribution. Where a run wants "try each
    slot in turn" and nothing cleverer, this is that, and it is not a
    lesser thing than a random draw.

    SKIPPING, NOT FAILING: slots in `order` that are not currently on offer
    - blocked, or not part of this stage - are stepped over rather than
    returned or raised on. `order` is written once at construction while
    `candidate_slots` changes stage by stage and shrinks as slots get
    blocked, so an order that names a slot the current stage does not
    permit is the normal case, not an error.

    THE CURSOR ADVANCES ACROSS CALLS, and that is what makes it a cycle
    rather than a constant. It persists on the instance, so a fresh
    instance per run is what keeps two runs comparable - exactly the same
    caveat FakeExecutor documents about its own call-sequence dependence.
    """

    def __init__(self, order: Sequence[SlotName]) -> None:
        if not order:
            raise ValueError("FixedOrderPolicy needs at least one slot in its order")
        self.order: tuple[SlotName, ...] = tuple(order)
        self._cursor = 0

    def select_slot(
        self, state_card: Mapping[str, Any], candidate_slots: Sequence[SlotName]
    ) -> SlotName:
        """The next slot in `order` that is currently on offer.

        Walks at most one full lap from the cursor, so the scan terminates
        whether or not anything matches, and leaves the cursor one past
        whatever it returned - so the following call resumes rather than
        restarting. `state_card` is unread: this policy's whole point is
        that its sequence depends on nothing but its own construction.
        """
        offered = set(candidate_slots)
        for step in range(len(self.order)):
            index = (self._cursor + step) % len(self.order)
            slot = self.order[index]
            if slot in offered:
                self._cursor = (index + 1) % len(self.order)
                return slot
        # Raised rather than falling back to `candidate_slots[0]`. A silent
        # fallback would return a slot this policy was never configured to
        # pick, which is precisely the "policy secretly did something else"
        # failure the Controller's own validation exists to catch - and it
        # would be catching our own fallback rather than a real bug.
        raise ValueError(
            f"FixedOrderPolicy order {self.order} has no slot in common with "
            f"the offered candidates {tuple(candidate_slots)}"
        )
