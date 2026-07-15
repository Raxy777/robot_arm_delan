"""Causal fixed-delay and zero-order-hold primitives for simulation datasets."""
from __future__ import annotations

from collections import deque
from copy import deepcopy


class FixedDelayQueue:
    """Return a sample from exactly ``delay_steps`` in the past.

    Before enough real samples exist, ``initial_sample`` is returned. Samples may
    be tuples containing timestamps and arrays. Values are copied on insertion
    and output so later simulator mutations cannot leak backwards in time.
    """

    def __init__(self, delay_steps: int, initial_sample):
        if int(delay_steps) != delay_steps or delay_steps < 0:
            raise ValueError("delay_steps must be a non-negative integer")
        self.delay_steps = int(delay_steps)
        self._initial = deepcopy(initial_sample)
        self._samples = deque(maxlen=self.delay_steps + 1)

    def reset(self, initial_sample=None):
        if initial_sample is not None:
            self._initial = deepcopy(initial_sample)
        self._samples.clear()

    def push(self, sample):
        """Insert the current sample and return the causally available sample."""
        self._samples.append(deepcopy(sample))
        if len(self._samples) <= self.delay_steps:
            return deepcopy(self._initial)
        return deepcopy(self._samples[0])
