import pytest

from recsys.hashing import stable_fraction


def test_stable_fraction_is_deterministic_and_in_the_unit_interval():
    values = [stable_fraction(f"salt:{i}") for i in range(1000)]

    assert values == [stable_fraction(f"salt:{i}") for i in range(1000)]
    assert all(0.0 <= v < 1.0 for v in values)


def test_stable_fraction_pins_a_known_value_so_assignments_never_drift():
    # SHA-256 based; changing the hash would silently reshuffle tune/test users and A/B arms.
    assert stable_fraction("kuairec:0") == pytest.approx(0.0033851034, abs=1e-9)


def test_stable_fraction_spreads_keys_roughly_uniformly():
    values = [stable_fraction(f"user:{i}") for i in range(10_000)]

    tenths = [sum(int(v * 10) == k for v in values) / len(values) for k in range(10)]

    assert all(0.09 < share < 0.11 for share in tenths)
