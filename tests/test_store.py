import pytest

from modelguard.store import connect, log_batch, read_batch, set_labels


def test_log_and_read_back(synthetic):
    con = connect(":memory:")
    X = synthetic.drop(columns="income").head(5)
    log_batch(con, "b1", X, ["<=50K", ">50K", "<=50K", "<=50K", ">50K"], timestamp="2026-01-01T00:00:00+00:00")
    out = read_batch(con, "b1")
    assert len(out) == 5
    assert out["prediction"].tolist()[1] == ">50K"
    assert out["label"].isna().all()
    assert (out["timestamp"] == "2026-01-01T00:00:00+00:00").all()
    assert out["age"].tolist() == pytest.approx(X["age"].tolist())
    assert out["workclass"].tolist() == X["workclass"].tolist()


def test_batches_are_isolated(synthetic):
    con = connect(":memory:")
    X = synthetic.drop(columns="income")
    log_batch(con, "a", X.head(2), ["x", "y"])
    log_batch(con, "b", X.head(3), ["x", "y", "z"])
    assert len(read_batch(con, "a")) == 2 and len(read_batch(con, "b")) == 3
    assert read_batch(con, "missing").empty


def test_labels_filled_later(synthetic):
    con = connect(":memory:")
    X = synthetic.drop(columns="income").head(3)
    log_batch(con, "b", X, ["p", "q", "r"])
    set_labels(con, "b", ["1", "0", "1"])
    assert read_batch(con, "b")["label"].tolist() == ["1", "0", "1"]


def test_length_mismatch_raises(synthetic):
    con = connect(":memory:")
    X = synthetic.drop(columns="income").head(3)
    with pytest.raises(ValueError):
        log_batch(con, "b", X, ["p"])
    log_batch(con, "b", X, ["p", "q", "r"])
    with pytest.raises(ValueError):
        set_labels(con, "b", ["1"])


def test_missing_values_survive(synthetic):
    con = connect(":memory:")
    X = synthetic.drop(columns="income").head(2).copy()
    X.loc[0, "workclass"] = None
    X.loc[1, "age"] = float("nan")
    log_batch(con, "m", X, ["p", "q"])
    out = read_batch(con, "m")
    assert out["workclass"].isna().tolist() == [True, False]
    assert out["age"].isna().tolist() == [False, True]
