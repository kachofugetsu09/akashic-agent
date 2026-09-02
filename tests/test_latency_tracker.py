"""Tests for LatencyTracker — adaptive P90 latency estimation."""

import statistics

import pytest

from agent.scheduler import LatencyTracker


class TestLatencyTrackerDefault:
    def test_returns_default_when_no_samples(self):
        t = LatencyTracker(default=25.0)
        assert t.lead == 25.0

    def test_returns_default_when_one_sample(self):
        t = LatencyTracker(default=25.0)
        t.record(5.0)
        assert t.lead == 25.0

    def test_returns_default_when_two_samples(self):
        t = LatencyTracker(default=25.0)
        t.record(5.0)
        t.record(10.0)
        assert t.lead == 25.0

    def test_custom_default(self):
        t = LatencyTracker(default=10.0)
        assert t.lead == 10.0


class TestLatencyTrackerP90:
    def test_p90_with_sufficient_samples(self):
        t = LatencyTracker(default=25.0, window=20)
        # 20 samples from 1 to 20 seconds
        for i in range(1, 21):
            t.record(float(i))
        # P90 of [1..20] ≈ 18.1 (quantiles n=10 index 8 = 90th percentile)
        expected = statistics.quantiles(list(range(1, 21)), n=10)[8]
        assert abs(t.lead - expected) < 0.01

    def test_p90_is_not_mean(self):
        t = LatencyTracker(default=25.0, window=20)
        samples = [10.0] * 18 + [100.0, 100.0]  # mostly 10s, two spikes
        for s in samples:
            t.record(s)
        mean = sum(samples) / len(samples)
        assert t.lead > mean  # P90 should be higher than mean in this skewed set

    def test_p90_is_not_max(self):
        t = LatencyTracker(default=25.0, window=20)
        for i in range(1, 21):
            t.record(float(i))
        assert t.lead < 20.0  # must be less than max

    def test_window_slides_old_samples_drop(self):
        t = LatencyTracker(default=25.0, window=5)
        # Fill with high latency
        for _ in range(5):
            t.record(100.0)
        high_lead = t.lead
        # Replace with low latency
        for _ in range(5):
            t.record(1.0)
        low_lead = t.lead
        assert low_lead < high_lead

    def test_spike_raises_then_recovers(self):
        # Use window=5 so behavior is easy to reason about
        t = LatencyTracker(default=25.0, window=5)
        # All spikes: window full with 60s
        for _ in range(5):
            t.record(60.0)
        spiked_lead = t.lead
        # Fully recover: replace all window slots with 10s
        for _ in range(5):
            t.record(10.0)
        recovered_lead = t.lead
        assert recovered_lead < spiked_lead
