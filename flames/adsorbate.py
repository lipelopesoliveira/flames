from typing import Any

import ase.atoms
import ase.io
import numpy as np

from flames.eos import BaseEOS, PengRobinsonEOS
from flames.move_weights import MoveWeights


class Adsorbate:
    """
    Represents an independent adsorbate species in a simulation, including its
    atomic structure, movement probabilities, and equation of state.

    Attributes:
        name (str): The identifier for the adsorbate species.
        structure (list[ase.atoms.Atoms] | None): The atomic configuration(s) of the adsorbate.
        weights (MoveWeights): The probabilities/weights for different Monte Carlo moves.
        eos (BaseEOS | None): The equation of state associated with the adsorbate.
    """

    def __init__(
        self,
        name: str,
        structure: ase.atoms.Atoms | str | list[ase.atoms.Atoms] | None = None,
        mol_fraction: float = 1.0,
        weights: MoveWeights | dict[str, float] | None = None,
        eos: BaseEOS | dict[str, float] | None = None,
        index: int | str = ":",
        **kwargs: Any,
    ) -> None:
        """
        Initializes an Adsorbate instance.

        Args:
            name (str): The name of the adsorbate species (e.g., 'CO2', 'N2').
            structure (Atoms | str | list[Atoms] | None): The structural representation.
                If a string is provided, it is treated as a file path and read via ASE.
            mol_fraction (float): The mole fraction of the adsorbate in the gas phase.
            weights (MoveWeights | dict[str, float] | None): Move probabilities. If a dict
                is passed, it initializes a MoveWeights object. If None, uses defaults.
            eos (BaseEOS | dict[str, float] | None): Equation of state object or parameters.
            index (int | str): The index or slice to read if `structure` is a file path.
                Defaults to ':' (reads all configurations).
            **kwargs (Any): Additional keyword arguments passed to `ase.io.read`.
        """
        self._name = name

        self._mol_fraction = mol_fraction

        # Handle file reading here to utilize index and **kwargs
        self._structure = None
        if isinstance(structure, str):
            parsed = ase.io.read(structure, index=index, **kwargs)
            self.structure = [parsed] if isinstance(parsed, ase.atoms.Atoms) else parsed
        else:
            self.structure = structure

        # Initialize defaults, then route through setters. Empty dict triggers default MoveWeights
        self._weights: MoveWeights = MoveWeights()
        self.weights = weights if weights is not None else {}

        # Route through setter to handle EOS initialization
        self._eos = None
        self.eos = eos

    def __repr__(self) -> str:
        """Returns a string representation of the Adsorbate object."""
        return f"Adsorbate(name={self.name}, mol_fraction={self.mol_fraction}, structure={self.structure}, weights={self.weights}, eos={self.eos})"

    def __str__(self) -> str:
        """Returns a human-readable string summarizing the Adsorbate."""
        return f"Adsorbate: {self.name}, Mole Fraction: {self.mol_fraction}, Structure: {self.structure}, Weights: {self.weights}, EOS: {self.eos}"

    def __iter__(self):
        """Allows iteration over the adsorbate's atomic structure(s)."""
        return iter(self.structure) if self.structure is not None else iter([])

    @property
    def name(self) -> str:
        """str: The name of the adsorbate species."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def mol_fraction(self) -> float:
        """float: The mole fraction of the adsorbate in the gas phase."""
        return self._mol_fraction

    @mol_fraction.setter
    def mol_fraction(self, value: float) -> None:
        if not (0 <= value <= 1):
            raise ValueError("Mole fraction must be between 0 and 1.")
        self._mol_fraction = value

    @property
    def structure(self) -> list[ase.atoms.Atoms] | None:
        """list[Atoms] | None: The structure(s) of the adsorbate, always stored as a list."""
        return self._structure

    @structure.setter
    def structure(self, value: ase.atoms.Atoms | list[ase.atoms.Atoms] | None) -> None:
        """
        Sets the structure of the adsorbate, ensuring it is stored as a list of Atoms.

        Args:
            value (str | Atoms | list[Atoms] | None): The structure to set. Strings are
                interpreted as file paths and read using ASE's default read behavior.

        Raises:
            ValueError: If the input is not a recognized structure type or valid file.
        """
        if value is None:
            self._structure = None
        elif isinstance(value, ase.atoms.Atoms):
            self._structure = [value]
        elif isinstance(value, list) and all(isinstance(item, ase.atoms.Atoms) for item in value):
            self._structure = value
        else:
            raise ValueError(
                "Structure must be an ASE Atoms object, a list of Atoms objects, or None."
            )

    @property
    def weights(self) -> MoveWeights:
        """MoveWeights: The object managing Monte Carlo move probabilities."""
        return self._weights

    @weights.setter
    def weights(self, value: MoveWeights | dict[str, float]) -> None:
        """
        Sets the move weights for the adsorbate.

        Args:
            value (MoveWeights | dict[str, float]): A MoveWeights instance, or a dictionary
                of move probabilities to construct one.

        Raises:
            ValueError: If the input is neither a MoveWeights instance nor a dictionary.
        """
        if isinstance(value, MoveWeights):
            self._weights = value
        elif isinstance(value, dict):
            # Assuming MoveWeights is imported and available in scope
            self._weights = MoveWeights(**value)
        else:
            raise ValueError(
                "Weights must be a MoveWeights object or a dictionary of move probabilities."
            )

    @property
    def eos(self) -> BaseEOS | None:
        """BaseEOS | None: The equation of state model for the adsorbate."""
        return self._eos

    @eos.setter
    def eos(self, value: BaseEOS | PengRobinsonEOS | dict[str, float] | None) -> None:
        """
        Sets the Equation of State (EOS) for the adsorbate.

        Args:
            value (BaseEOS | PengRobinsonEOS | dict[str, float] | None): The EOS object,
                or a dictionary of parameters to initialize a PengRobinsonEOS.
                If a dict is used, the structure must be set first to calculate molar mass.

        Raises:
            ValueError: If structure is missing when passing a dict, or if the value type is invalid.
        """
        if value is None:
            self._eos = None
        elif isinstance(value, (BaseEOS, PengRobinsonEOS)):
            self._eos = value
        elif isinstance(value, dict):
            if not self.structure:
                raise ValueError(
                    "Structure must be set before setting EOS parameters via dictionary."
                )
            # Because structure is normalized to a list, we can safely grab index 0
            self._eos = PengRobinsonEOS(**value, molarMass=self.structure[0].get_masses().sum())
        else:
            raise ValueError(
                "EOS must be a BaseEOS or PengRobinsonEOS object, a dictionary of EOS parameters, or None."
            )

    def pick_random_move(self, generator: np.random.Generator | None = None) -> str:
        """
        Selects a random Monte Carlo move based on the adsorbate's move weights.

        Args:
            generator (np.random.Generator | None): An optional NumPy random number generator.
                If None, the default RNG is used.

        Returns:
            str: The string identifier of the selected move.
        """
        if generator is None:
            generator = np.random.default_rng()
        return self.weights.pick_random_move(generator=generator)

    def pick_structure(self, generator: np.random.Generator | None = None) -> ase.atoms.Atoms:
        """
        Randomly selects and returns a copy of one of the adsorbate's structures.

        Args:
            generator (np.random.Generator | None): An optional NumPy random number generator.
                If None, the default RNG is used.

        Returns:
            ase.atoms.Atoms: A deep copy of the selected atomic structure.

        Raises:
            ValueError: If no structure has been assigned to the adsorbate.
        """
        if not self.structure:
            raise ValueError("No structure available to pick from. Please set the structure first.")

        if generator is None:
            generator = np.random.default_rng()

        if len(self.structure) == 1:
            return self.structure[0].copy()

        # Using generator.integers avoids issues np.random.choice has with arrays of complex objects
        idx = generator.integers(len(self.structure))
        return self.structure[idx].copy()
