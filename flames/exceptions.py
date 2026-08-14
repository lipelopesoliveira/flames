# Created by Felipe Lopes de Oliveira

"""
This file implements custom exceptions for the flames package.
"""


class MoveKeyError(Exception):
    """Exception raised for errors in the move key.

    Attributes
    ----------
    move : str
        The move keys that caused the error.
    """

    def __init__(self, moves: list):
        valid_keys = {"insertion", "deletion", "translation", "rotation"}
        self.message = (
            "Error: move_weights must contain exactly the keys:" + ", ".join(valid_keys) + ".\n"
        )
        self.message += " Invalid keys: " + ", ".join(
            [move for move in moves if move not in valid_keys]
        )
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class InsertionDeletionError(Exception):
    """Exception raised when insertion and deletion weights are not equal.

    Attributes
    ----------
    insertion_weight : float
        The weight for insertion moves.
    deletion_weight : float
        The weight for deletion moves.
    """

    def __init__(self, insertion_weight: float, deletion_weight: float):
        self.message = "Error: Weights for 'insertion' and 'deletion' must be equal.\n"
        self.message += f" Insertion weight: {insertion_weight}, Deletion weight: {deletion_weight}"
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message
