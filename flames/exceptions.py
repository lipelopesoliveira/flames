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
        valid_keys = {
            "insertion",
            "deletion",
            "translation",
            "rotation",
            "reinsertion",
            "position_swap",
            "nve_md",
            "nvt_md",
            "npt_md",
        }

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


class MaxInsertionTriesError(Exception):
    """Exception raised when the maximum number of insertion tries is exceeded.

    Attributes
    ----------
    max_insertion_tries : int
        The maximum number of insertion tries allowed.
    n_adsorbates : int
        The number of adsorbates to be inserted.
    inserted_adsorbates : int
        The number of adsorbates that were successfully inserted.
    """

    def __init__(self, max_insertion_tries: int, n_adsorbates: int, inserted_adsorbates: int):
        self.message = (
            f"Error: Maximum number of insertion attempts ({max_insertion_tries}) exceeded.\n"
            f"Could not insert all {n_adsorbates} adsorbates without overlap. "
            f"Max inserted adsorbates: {inserted_adsorbates}."
        )
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message
