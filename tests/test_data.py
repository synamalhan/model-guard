from modelguard.data import split_reference


def test_reference_is_first_30_percent_in_order(synthetic):
    ref, live = split_reference(synthetic)
    assert len(ref) == 120 and len(live) == 280
    assert ref.index[-1] + 1 == live.index[0]
