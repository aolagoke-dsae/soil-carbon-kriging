"""N1-N4 on a made-up cross-validation record."""
import numpy as np
import pandas as pd
import pytest

import nested as N


def _cv():
    """One field, one repeat, three folds of two locations; model b is best
    on folds 0 and 1, model c on fold 2, and c is far better on fold 2."""
    rows = []
    crps = {("a", 0): 2.0, ("b", 0): 1.0, ("c", 0): 1.5,
            ("a", 1): 2.0, ("b", 1): 1.0, ("c", 1): 1.5,
            ("a", 2): 2.0, ("b", 2): 3.0, ("c", 2): 0.1}
    for fold in range(3):
        for loc in range(2):
            for m in "abc":
                rows.append({"site": "F", "location_id": f"{fold}{loc}", "repeat": 0, "fold": fold,
                             "method": f"kriging_{m}", "y": 10.0, "mean": 10.0, "var": 1.0,
                             "crps": crps[(m, fold)], "q0.050": 8.0, "q0.950": 12.0, "q0.500": 10.0})
    return pd.DataFrame(rows)


def test_the_choice_ignores_the_held_out_folds_own_scores():
    """Prevents: fold 2 choosing c because of c's score on fold 2 itself."""
    ch = N.choices(_cv()).set_index("fold")["chosen"]
    # fold 2: on folds 0 and 1 alone, b (1.0) beats c (1.5)
    assert ch[2] == "b"
    # fold 0: folds 1 and 2: a 2.0, b 2.0, c 0.8 -> c
    assert ch[0] == "c" and ch[1] == "c"


def test_nested_predictions_take_each_folds_choice():
    cv = _cv()
    nested = N.nested_predictions(cv, N.choices(cv))
    assert (nested["method"] == "kriging_n").all() and len(nested) == 6
    assert nested.set_index("location_id")["crps"].to_dict() == {"00": 1.5, "01": 1.5, "10": 1.5, "11": 1.5,
                                                                  "20": 3.0, "21": 3.0}


def test_outcome_calls_small_differences_ties():
    assert N.outcome(1.000, 1.009) == "tie"
    assert N.outcome(1.0, 1.02) == "win"
    assert N.outcome(1.02, 1.0) == "loss"
