import pandas as pd

from stcafnet.data import split_manifest


def test_stratified_split_has_no_overlap():
    rows = []
    for treatment in ("CK", "100Hz", "200Hz", "300Hz"):
        for day in range(8):
            for sample in range(100):
                rows.append(
                    {
                        "sample_id": f"{treatment}_{day}_{sample}",
                        "treatment": treatment,
                        "day": day,
                    }
                )
    frame = pd.DataFrame(rows)
    test, folds = split_manifest(frame, 0.15, 5, 42)
    assert len(test) == 480
    assert len(folds) == 5
    for train, validation in folds:
        assert set(train.sample_id).isdisjoint(validation.sample_id)
        assert set(train.sample_id).isdisjoint(test.sample_id)

