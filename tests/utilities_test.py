import warnings

import pytest

from flames.exceptions import InsertionDeletionError, MoveKeyError
from flames.utilities import check_weights


# -----------------------------
# VALID INPUT TEST
# -----------------------------
def test_check_weights_valid_input():
    move_weights = {
        "insertion": 2.0,
        "deletion": 2.0,
        "translation": 1.0,
        "rotation": 1.0,
    }
    result = check_weights(move_weights)

    # Check normalization sum
    total = sum(result.values())
    assert abs(total - 1.0) < 1e-12

    # Check proportional normalization
    expected = {k: v / 6.0 for k, v in move_weights.items()}
    assert result == expected


# -----------------------------
# TYPE CHECKS
# -----------------------------
def test_check_weights_non_dict_input():
    with pytest.raises(TypeError, match="move_weights must be a dictionary"):
        check_weights(["not", "a", "dict"])  # type: ignore


def test_check_weights_non_numeric_value():
    move_weights = {
        "insertion": 1,
        "deletion": 1,
        "translation": "not a number",
        "rotation": 1,
        "reinsertion": 1,
    }
    with pytest.raises(TypeError, match="move_weights\\['translation'\\] must be a number"):
        check_weights(move_weights)


# -----------------------------
# VALUE CHECKS
# -----------------------------
def test_check_weights_negative_value():
    move_weights = {
        "insertion": 1,
        "deletion": 1,
        "translation": -1,
        "rotation": 1,
        "reinsertion": 1,
    }
    with pytest.raises(ValueError, match="move_weights\\['translation'\\] must be non-negative"):
        check_weights(move_weights)


# -----------------------------
# KEY VALIDATION
# -----------------------------
def test_check_weights_invalid_key():
    move_weights = {
        "insertion": 1,
        "deletion": 1,
        "translation": 1,
        "wrong_key": 1,
        "reinsertion": 1,
    }
    with pytest.raises(MoveKeyError):
        check_weights(move_weights)


# -----------------------------
# MISSING KEY WARNINGS
# -----------------------------
def test_check_weights_missing_one_key_warns():
    move_weights = {
        "insertion": 1,
        "deletion": 1,
        "translation": 1,
        # missing "rotation"
        "reinsertion": 1,
    }
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = check_weights(move_weights)

        # Should issue exactly one warning
        assert len(w) == 1
        assert "missing the key 'rotation'" in str(w[0].message)

    # Missing key should be set to zero
    assert "rotation" in result
    assert result["rotation"] == 0.0 or abs(result["rotation"]) < 1e-12


def test_check_weights_missing_multiple_keys_warns():
    move_weights = {
        "insertion": 1,
        "deletion": 1,
        # missing: translation, rotation, reinsertion
    }
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = check_weights(move_weights)

        # Should issue 3 warnings
        assert len(w) == 3
        msgs = [str(wi.message) for wi in w]
        assert any("missing the key 'translation'" in msg for msg in msgs)
        assert any("missing the key 'rotation'" in msg for msg in msgs)

    # Check that missing keys were added as zero
    for key in ["translation", "rotation"]:
        assert key in result
        assert abs(result[key]) < 1e-12


# -----------------------------
# INSERTION/DELETION EQUALITY
# -----------------------------
def test_check_weights_insertion_deletion_inequal():
    move_weights = {
        "insertion": 1,
        "deletion": 2,
        "translation": 1,
        "rotation": 1,
        "reinsertion": 1,
    }
    with pytest.raises(InsertionDeletionError) as exc_info:
        check_weights(move_weights)
    assert "Insertion weight: 1" in exc_info.value.message
    assert "Deletion weight: 2" in exc_info.value.message


# -----------------------------
# NORMALIZATION EDGE CASE
# -----------------------------
def test_check_weights_all_zero():
    move_weights = {
        "insertion": 0,
        "deletion": 0,
        "translation": 0,
        "rotation": 0,
        "reinsertion": 0,
    }
    with pytest.raises(ZeroDivisionError):
        check_weights(move_weights)
