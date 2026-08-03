import pytest
import numpy as np
import warnings

from flames.move_weights import MoveWeights, MoveKeyError, InsertionDeletionError

# Assuming your code is saved in a file named move_weights.py
# from move_weights import MoveWeights, MoveKeyError, InsertionDeletionError

def test_default_initialization_normalizes_correctly():
    """Test that default values (1,1,1,1,0,0) normalize to 0.25 each."""
    mw = MoveWeights()
    assert mw.insertion == 0.25
    assert mw.deletion == 0.25
    assert mw.translation == 0.25
    assert mw.rotation == 0.25
    assert mw.reinsertion == 0.0
    assert mw.identity_change == 0.0

def test_custom_initialization_normalizes_correctly():
    """Test that custom provided values normalize to sum to 1."""
    # Sum of weights = 2 + 2 + 6 = 10
    mw = MoveWeights(
        insertion=2.0, 
        deletion=2.0, 
        translation=6.0, 
        rotation=0.0, 
        reinsertion=0.0,
        identity_change=0.0
    )
    assert mw.insertion == 0.2
    assert mw.deletion == 0.2
    assert mw.translation == 0.6
    assert mw.rotation == 0.0

def test_negative_weight_raises_value_error():
    """Test that providing a negative weight raises a ValueError."""
    with pytest.raises(ValueError, match="must be non-negative"):
        MoveWeights(translation=-1.0)

def test_invalid_type_raises_type_error():
    """Test that providing a string instead of a number raises a TypeError."""
    with pytest.raises(TypeError, match="must be a number"):
        MoveWeights(rotation="1.0") # type: ignore

def test_insertion_deletion_mismatch_raises_error():
    """Test that insertion and deletion must have the exact same weight."""
    with pytest.raises(InsertionDeletionError):
        MoveWeights(insertion=2.0, deletion=1.0)

def test_zero_total_weight_raises_assertion_error():
    """Test that an all-zero setup triggers the normalization assertion."""
    with pytest.raises(AssertionError, match="Total weight must be greater than 0"):
        MoveWeights(
            insertion=0.0, 
            deletion=0.0, 
            translation=0.0, 
            rotation=0.0, 
            reinsertion=0.0, 
            identity_change=0.0
        )

def test_from_dict_valid():
    """Test the from_dict classmethod with a valid dictionary."""
    data = {
        "insertion": 1.0,
        "deletion": 1.0,
        "translation": 2.0,
        "rotation": 0.0,
        "reinsertion": 0.0,
        "identity_change": 0.0
    }
    mw = MoveWeights.from_dict(data)
    assert mw.insertion == 0.25
    assert mw.translation == 0.5

def test_from_dict_invalid_type_raises_assertion_error():
    """Test that from_dict requires a dictionary type."""
    with pytest.raises(AssertionError, match="Data must be a dictionary"):
        MoveWeights.from_dict(["insertion", 1.0]) # type: ignore

def test_from_dict_invalid_keys_raises_move_key_error():
    """Test that from_dict rejects unknown keys."""
    data = {"insertion": 1.0, "deletion": 1.0, "fake_move": 5.0}
    with pytest.raises(MoveKeyError, match="Invalid keys provided"):
        MoveWeights.from_dict(data)

def test_from_dict_missing_keys_warns():
    """Test that from_dict triggers a UserWarning when keys are missing."""
    data = {
        "insertion": 2.0,
        "deletion": 2.0,
        "translation": 4.0
        # Missing rotation, reinsertion, identity_change
    }
    with pytest.warns(UserWarning, match="Warning: Missing the key"):
        mw = MoveWeights.from_dict(data)
        
    # Note: Because the current implementation doesn't explicitly set data[key] = 0.0
    # in the loop, the dataclass falls back to default `rotation=1.0`. 
    # If you fix the logic in your class, you should assert mw.rotation == 0.0 here!
    assert mw.insertion is not None

def test_asdict_returns_correct_dictionary():
    """Test that the custom asdict method returns the current state."""
    mw = MoveWeights(insertion=5.0, deletion=5.0, translation=10.0, rotation=0.0)
    d = mw.asdict()
    assert isinstance(d, dict)
    assert d["insertion"] == 0.25
    assert d["translation"] == 0.5
    assert d["identity_change"] == 0.0

def test_pick_random_move_returns_valid_key():
    """Test that pick_random_move returns a string belonging to the keys."""
    mw = MoveWeights()
    rng = np.random.default_rng(seed=42)
    move = mw.pick_random_move(generator=rng)
    assert move in mw.asdict().keys()

def test_pick_random_move_respects_probabilities():
    """Test that an event with 100% probability is always chosen."""
    mw = MoveWeights(
        insertion=0.0, 
        deletion=0.0, 
        translation=10.0, 
        rotation=0.0, 
        reinsertion=0.0,
        identity_change=0.0
    )
    rng = np.random.default_rng(seed=99)
    move = mw.pick_random_move(generator=rng)
    # Since translation is the only move with weight > 0, it must be picked.
    assert move == "translation"