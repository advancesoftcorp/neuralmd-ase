"""
Copyright (c) 2023, AdvanceSoft Corp.

This source code is licensed under the GNU General Public License Version 2
found in the LICENSE file in the root directory of this source tree.
"""

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.lammpslib  import LAMMPSlib

class NeuralMDCalculator(Calculator):
    """NeuralMD calculator for ASE, implemented as a wrapper of LAMMPSlib."""

    implemented_properties = ("energy", "free_energy", "forces", "stress", "energies")

    def __init__(
        self,
        ffield_file:  str   = "ffield.sannp",
        with_eatom:   bool  = False,
        with_charge:  bool  = False,
        coul_rcut:    float = 10.0,
        coul_style:   str   = "long",
        kspace_style: str   = "pppm 1.0e-6",
        **kwargs
    ):
        """
        Init NeuralMDCalculator with a force-field file.

        Args:
            ffield_file  (str):   path of force-field file
            with_eatom   (bool):  including atomic energy from bias of last layer
            with_charge  (bool):  including charge of 3G-HDNNP
            coul_rcut    (float): cutoff radius of coulomb, in Angstrom, (only if with_charge == True)
            coul_style   (str):   style of coulomb, which is "long" or "cut" (only if with_charge == True)
            kspace_style (str):   kspace_style of LAMMPS to calculate long coulomb (onlyl if with_charge == True)
            **kwargs: Kwargs pass through to super().__init__().
        """
        super().__init__(**kwargs)

        self.ffield_file  = ffield_file
        self.with_eatom   = with_eatom
        self.with_charge  = with_charge
        self.coul_rcut    = coul_rcut
        self.coul_style   = coul_style
        self.kspace_style = kspace_style

        self.elements  = None
        self.lammpslib = None
        
    def __del__(self):
        if self.lammpslib is not None:
            self.lammpslib = None
            
    def __get_elements(self, atoms):
        symbols  = atoms.get_chemical_symbols()
        _, index = np.unique(symbols, return_index = True)
        elemlist = np.array(symbols)[np.sort(index)].tolist()
        elements = " ".join(elemlist)

        return elements

    def __create_lammpslib(self):
        if self.with_charge:
            pair_style = "nnp/coul/" + self.coul_style + " " + str(self.coul_rcut)
        else:
            pair_style = "nnp"
            
        pair_coeff = "* * " + self.ffield_file \
                   + (" " if self.with_eatom else " eatom zero ") + self.elements

        lmpcmds = [
            "pair_style " + pair_style,
            "pair_coeff " + pair_coeff
        ]
        
        if self.with_charge:
            lmpcmds.append("kspace_style " + self.kspace_style)
            lmpcmds.append("kspace_modify gewald 0.5")

        lammps_header = [
            "units metal",
            "atom_style " + ("charge" if self.with_charge else "atomic"),
            "atom_modify map array sort 0 0",
        ]
        
        lammpslib = LAMMPSlib(
            lmpcmds       = lmpcmds,
            lammps_header = lammps_header,
            lammps_name   = "nmd",
            log_file      = "neuralmd.log",
            keep_alive    = False
        )
        
        return lammpslib

    def calculate(
        self,
        atoms:          Atoms = None,
        properties:     list  = None,
        system_changes: list  = None
    ):
        """
        Calculate energy, forces and stress of the Atoms using NeuralMD.

        Args:
            atoms          (Atoms): Atoms object
            properties     (list):  properties to calculate
            system_changes (list):  system has been changed
        """
        elements = self.__get_elements(atoms)

        if (self.lammpslib is None) or (self.elements is None) or (self.elements != elements):
            self.elements  = elements
            self.lammpslib = self.__create_lammpslib()
            
        self.lammpslib.calculate(atoms, properties, system_changes)

        self.results = self.lammpslib.results

