from modelguard.train import train


def test_train_is_deterministic_and_learns(synthetic):
    m1, a1, f1 = train(synthetic)
    _, a2, _ = train(synthetic)
    assert a1 == a2
    assert 0 < f1 <= 1
    assert a1 > 0.7  # synthetic target is a noisy function of age
    assert len(m1.predict(synthetic.drop(columns="income").head(3))) == 3
