import warnings
from dataclasses import dataclass, fields

import numpy as np


# Custom exceptions from your original code
class MoveKeyError(Exception):
    pass


class InsertionDeletionError(Exception):
    pass


@dataclass
class MoveWeights:
    """
    Stores and normalizes movement weights for the simulation.
    All weights default to 1.0 if not explicitly provided.
    """

    insertion: float = 1.0
    deletion: float = 1.0
    translation: float = 1.0
    rotation: float = 1.0
    reinsertion: float = 0.0
    identity_change: float = 0.0

    def __post_init__(self):
        """Validates and normalizes weights immediately after initialization."""
        self._validate_types_and_values()
        self._validate_consistence()
        self._normalize_weights()

    def _validate_types_and_values(self):
        # Iterate through all defined fields in the dataclass
        for f in fields(self):
            value = getattr(self, f.name)
            if not isinstance(value, (int, float)):
                raise TypeError(f"Weight for '{f.name}' must be a number, not {type(value)}")
            if value < 0:
                raise ValueError(f"Weight for '{f.name}' must be non-negative, not {value}")

    def _validate_consistence(self):
        if self.insertion != self.deletion:
            raise InsertionDeletionError(
                f"Insertion ({self.insertion}) and deletion ({self.deletion}) weights must be equal."
            )

    def _normalize_weights(self):
        total_weight = sum(getattr(self, f.name) for f in fields(self))

        assert total_weight > 0, "Total weight must be greater than 0 to normalize."

        # Reassign normalized values
        for f in fields(self):
            normalized_value = getattr(self, f.name) / total_weight
            setattr(self, f.name, normalized_value)

    @classmethod
    def from_dict(cls, data: dict):
        """
        Alternative constructor to ease the transition from your old code.
        Allows you to initialize the class directly from a dictionary.
        """
        assert isinstance(data, dict), f"Data must be a dictionary, not {type(data)}"

        valid_keys = {f.name for f in fields(cls)}

        # Check for invalid keys
        invalid_keys = set(data.keys()) - valid_keys
        if invalid_keys:
            raise MoveKeyError(f"Invalid keys provided: {invalid_keys}")

        # Replicate your original warning for missing keys
        missing_keys = valid_keys - set(data.keys())
        for key in missing_keys:
            warnings.warn(f"Warning: Missing the key '{key}'. Assuming weight 0 for this move.")

        return cls(**data)

    def asdict(self) -> dict:
        """Returns the dataclass fields as a dictionary."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def pick_random_move(self, generator: np.random.Generator) -> str:
        """
        Randomly select a move based on the normalized weights.

        Parameters:
        - generator: A random number generator (e.g., numpy.random.default_rng()).

        Returns:
        - str: The name of the selected move.
        """

        weights_dict = self.asdict()

        move = str(generator.choice(a=list(weights_dict.keys()), p=list(weights_dict.values())))

        return move
