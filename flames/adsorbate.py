import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional

import ase.io
import numpy as np
from ase import Atoms

from flames.eos import BaseEOS
from flames.move_weights import MoveWeights


@dataclass
class Adsorbate:
    """
    Represents an independent adsorbate species, including its
    atomic structure, movement probabilities, and equation of state.
    """

    name: str
    structure: Atoms
    weights: MoveWeights = field(default_factory=MoveWeights)
    eos: Optional[BaseEOS] = None

    @classmethod
    def from_file(
        cls,
        filepath: str,
        name: Optional[str] = None,
        weights: Optional[MoveWeights] = None,
        eos: Optional[BaseEOS] = None,
        **kwargs: Any,
    ) -> "Adsorbate":

        try:
            structure = ase.io.read(filepath, **kwargs)
        except Exception as e:
            raise IOError(f"Failed to read structure from {filepath}: {e}")

        if not isinstance(structure, Atoms):
            if isinstance(structure, list) and len(structure) > 0:
                warnings.warn(f"Multiple structures found in {filepath}. Using the first one.")
                structure = structure[0]
            else:
                raise ValueError(f"Expected an ASE Atoms object, but got {type(structure)}")

        if name is None:
            import os

            name = os.path.splitext(os.path.basename(filepath))[0]

        if weights is None:
            weights = MoveWeights()

        return cls(name=name, structure=structure, weights=weights, eos=eos)

    def pick_random_move(self, generator: np.random.Generator) -> str:
        return self.weights.pick_random_move(generator=generator)


class Adsorbates:
    """
    A collection of Adsorbate objects representing the gas phase mixture.
    Manages partial pressures, mole fractions, and aggregate thermodynamic properties.
    """

    def __init__(self):
        # We store the adsorbates and their partial pressures in dictionaries
        # keyed by the adsorbate's name for easy lookup.
        self._components: Dict[str, Adsorbate] = {}
        self._partial_pressures: Dict[str, float] = {}

    def add(self, adsorbate: Adsorbate, partial_pressure: float) -> None:
        """Add an adsorbate to the mixture with its partial pressure (in Pascals)."""
        if partial_pressure <= 0:
            raise ValueError(f"Partial pressure for {adsorbate.name} must be > 0.")

        self._components[adsorbate.name] = adsorbate
        self._partial_pressures[adsorbate.name] = partial_pressure

        # If the adsorbate has an EOS, update its pressure to match the partial pressure
        if adsorbate.eos is not None:
            adsorbate.eos.P = partial_pressure

    def remove(self, name: str) -> None:
        """Remove an adsorbate from the mixture."""
        if name in self._components:
            del self._components[name]
            del self._partial_pressures[name]
        else:
            raise KeyError(f"Adsorbate '{name}' not found in the mixture.")

    def get(self, name: str) -> Adsorbate:
        """Retrieve an Adsorbate by name."""
        return self._components[name]

    @property
    def total_pressure(self) -> float:
        """Calculate the total pressure of the mixture."""
        return sum(self._partial_pressures.values())

    @property
    def mole_fractions(self) -> Dict[str, float]:
        """Calculate the mole fraction (y_i) for each component."""
        p_tot = self.total_pressure
        if p_tot == 0:
            return {name: 0.0 for name in self._components}
        return {name: p / p_tot for name, p in self._partial_pressures.items()}

    def pick_random_adsorbate(self, generator: np.random.Generator) -> Adsorbate:
        """
        Randomly select an adsorbate for a Monte Carlo insertion move,
        weighted by their mole fractions in the gas phase.
        """
        if not self._components:
            raise ValueError("No adsorbates in the mixture.")

        names = list(self._components.keys())
        fractions = list(self.mole_fractions.values())

        selected_name = str(generator.choice(a=names, p=fractions))
        return self.get(selected_name)

    def __iter__(self) -> Iterator[Adsorbate]:
        """Allow iterating directly over the Adsorbate objects."""
        return iter(self._components.values())

    def __len__(self) -> int:
        """Return the number of unique adsorbate species in the mixture."""
        return len(self._components)
