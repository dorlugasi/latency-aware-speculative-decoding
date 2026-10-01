import pytest

import specdecode.speculative as speculative


def test_cuda_timed_call_uses_cuda_events(monkeypatch):
    """Unit-test the timing mechanism without requiring a real CUDA GPU."""
    calls = []

    class FakeEvent:
        def __init__(self, enable_timing=False):
            calls.append(("create", enable_timing))

        def record(self):
            calls.append(("record",))

        def synchronize(self):
            calls.append(("synchronize",))

        def elapsed_time(self, other):
            calls.append(("elapsed_time",))
            return 12.5

    monkeypatch.setattr(speculative.torch.cuda, "Event", FakeEvent)

    value, elapsed_s = speculative._timed_call("cuda", lambda: "ok")

    assert value == "ok"
    assert elapsed_s == pytest.approx(0.0125)
    assert calls.count(("record",)) == 2
    assert ("synchronize",) in calls
    assert ("elapsed_time",) in calls
