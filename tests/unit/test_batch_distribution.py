from itertools import product

import pytest

from chameleon.planner import batch_distributions, repair_zero_partitions


@pytest.mark.parametrize("before,after", [
    ((2, 0), (1, 1)), ((5, 0, 0, 0), (2, 1, 1, 1)),
    ((4, 4, 0, 0, 0), (2, 3, 1, 1, 1)), ((2, 2, 0), (1, 2, 1)),
    ((0, 2, 2, 0), (1, 1, 1, 1)), ((1, 3, 2), (1, 3, 2)),
    ((1,), (1,)), ((0, 0, 2), None), ((0, 0), None),
])
def test_zero_repair_repeated_donors_and_stable_ties(before, after):
    assert repair_zero_partitions(before) == after
    if after is not None:
        assert sum(after) == sum(before) and min(after) >= 1


@pytest.mark.parametrize("lengths", [(1,), (1, 1), (1, 2), (1, 1, 3), (1, 1, 1, 9), (2, 3, 4)])
@pytest.mark.parametrize("nm", range(1, 10))
def test_batch_candidates_against_independent_remainder_product(lengths, nm):
    expected = set()
    if nm >= len(lengths):
        base = [nm * length // sum(lengths) for length in lengths]
        remainder = nm - sum(base)
        for additions in product(range(remainder + 1), repeat=len(lengths)):
            if sum(additions) != remainder:
                continue
            counts = [count + addition for count, addition in zip(base, additions)]
            while 0 in counts:
                recipient = counts.index(0)
                donor = sorted(range(len(counts)), key=lambda i: (-counts[i], i))[0]
                assert counts[donor] > 1
                counts[donor] -= 1
                counts[recipient] += 1
            expected.add(tuple(counts))
    actual = batch_distributions(nm, lengths)
    assert actual == tuple(sorted(expected))
    assert all(sum(counts) == nm and min(counts) >= 1 for counts in actual)


def test_multiple_remainders_can_go_to_the_same_pipeline():
    assert batch_distributions(5, (1, 1, 1)) == (
        (1, 1, 3), (1, 2, 2), (1, 3, 1), (2, 1, 2), (2, 2, 1), (3, 1, 1))
    assert batch_distributions(4, (1, 1, 1, 9)) == ((1, 1, 1, 1),)


@pytest.mark.parametrize("values", [(), (-1, 2), (True, 2), (1.5, 0)])
def test_invalid_zero_repair_inputs(values):
    with pytest.raises(ValueError):
        repair_zero_partitions(values)


@pytest.mark.parametrize("nm,lengths", [(0, (1,)), (True, (1,)), (2, ()), (2, (0, 1)), (2, (1.5,))])
def test_invalid_batch_inputs(nm, lengths):
    with pytest.raises(ValueError):
        batch_distributions(nm, lengths)
