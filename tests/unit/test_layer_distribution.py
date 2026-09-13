from itertools import product

import pytest

from chameleon.planner import layer_distributions, layer_layouts


@pytest.mark.parametrize("layers", range(1, 8))
@pytest.mark.parametrize("stages", range(1, 6))
def test_layer_counts_against_exhaustive_product(layers, stages):
    base, remainder = divmod(layers, stages)
    expected = tuple(sorted({counts for counts in product((base, base + 1), repeat=stages)
                             if sum(counts) == layers and counts.count(base + 1) == remainder}))
    assert layer_distributions(layers, stages) == expected


def test_remainder_layouts_preserve_all_modules_and_endpoint_placement():
    layers = ("blocks.0", "blocks.1", "blocks.2")
    order = ("embedding", "prefix_extra", "blocks.0", "between", "blocks.1", "blocks.2",
             "final_norm", "lm_head", "suffix_extra")
    assert layer_layouts(order, layers, 2) == (
        (("embedding", "prefix_extra", "blocks.0", "between"),
         ("blocks.1", "blocks.2", "final_norm", "lm_head", "suffix_extra")),
        (("embedding", "prefix_extra", "blocks.0", "between", "blocks.1"),
         ("blocks.2", "final_norm", "lm_head", "suffix_extra")),
    )
    assert layer_layouts(order, layers, 1) == ((order,),)


def test_zero_block_endpoint_stages_are_valid_but_empty_stages_are_excluded():
    order, layers = ("embedding", "blocks.0", "final_norm", "lm_head"), ("blocks.0",)
    assert layer_layouts(order, layers, 3) == ((("embedding",), ("blocks.0",), ("final_norm", "lm_head")),)
    assert layer_layouts(order, layers, 4) == ()
    assert layer_layouts(("blocks.0",), layers, 2) == ()


@pytest.mark.parametrize("layers,stages", [(0, 1), (True, 1), (1, 0), (1, 1.5)])
def test_invalid_layer_counts(layers, stages):
    with pytest.raises(ValueError):
        layer_distributions(layers, stages)


@pytest.mark.parametrize("order,layers", [(("a", "b"), ()), (("a", "b"), ("absent",)),
                                          (("a", "b"), ("b", "a")), (("a", "a"), ("a",)),
                                          (("a", "b"), ("a", "a"))])
def test_layer_inventory_must_follow_unique_model_order(order, layers):
    with pytest.raises(ValueError):
        layer_layouts(order, layers, 1)
