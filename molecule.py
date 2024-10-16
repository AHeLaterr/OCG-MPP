# molecule.py
from rdkit import Chem
from rdkit.Chem import Lipinski, AllChem, Descriptors, rdFreeSASA, Descriptors3D, Fragments, rdMolDescriptors
from rdkit.Chem.EState import Fingerprinter
from ccdc.io import CrystalReader
import numpy as np
import tempfile
import os
import subprocess
import math
from openbabel import openbabel, pybel
import inspect

Rvdw = {'H': 1.20,'He': 1.4,'Li': 1.82,'Be': 1.7,'B': 1.92,'C': 1.70,'N': 1.55,
        'O': 1.52,'F': 1.47,'Ne': 1.54,'Na': 2.27,'Mg': 1.73,'Al': 2.05,'Si': 2.1,
        'P': 1.80,'S': 1.80,'Cl': 1.75,'Ar': 1.88,'Br': 1.85,'I': 1.98}

class MolecularProperties:
    def __init__(self, cif_path):
        self.cif_path = cif_path
        self.warnings = []  # 存储警告和错误信息
        self.mol1 = self.cif_to_rdkit_mol_with_spatial_info(cif_path)
        self.mol = self.remove_isolated_atoms(self.mol1)

    def convert_cif_to_sdf_using_obabel(self, cif_path, sdf_path):
        try:
            subprocess.run(["obabel", "-icif", cif_path, "-osdf", "-O", sdf_path, "-h"], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error converting CIF to SDF using Open Babel: {e}")
            return False

    def cif_to_rdkit_mol_with_spatial_info(self, cif_path):
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as temp_sdf:
            sdf_path = temp_sdf.name

        if not self.convert_cif_to_sdf_using_obabel(cif_path, sdf_path):
            os.remove(sdf_path)
            return None

        mol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]
        os.remove(sdf_path)

        if mol:
            return mol
        else:
            print("Failed to create RDKit Mol from SDF.")
            return None

    def remove_isolated_atoms(self, mol):
        emol = Chem.RWMol(mol)
        to_remove = []
        for atom in emol.GetAtoms():
            if atom.GetDegree() == 0:
                to_remove.append(atom.GetIdx())
        for idx in sorted(to_remove, reverse=True):
            emol.RemoveAtom(idx)
        mol = emol.GetMol()
        Chem.SanitizeMol(mol)  # Ensure the molecule is in a consistent state
        return mol

    def has_warnings(self):
        return bool(self.warnings)

    def get_warnings(self):
        return "\n".join(self.warnings)

    def Estate_calculator(self):
        '''计算分子的 E-state 指数'''
        if not self.mol:
            return None
        exestate1, exestate2 = Fingerprinter.FingerprintMol(self.mol)
        estate1 = np.squeeze(exestate1)
        estate2 = np.squeeze(exestate2)
        estate_dict = {f'Estate_{i}': value for i, value in enumerate(np.append(estate1, estate2))}

        return estate_dict
    #
    def calculate_and_print_all_descriptors(self):
        if not self.mol:
            return None

        # 计算所有可用描述符
        all_descriptors = {desc[0]: desc[1](self.mol) for desc in Descriptors.descList}
        # for name, value in all_descriptors.items():
        #     print(f"{name}: {value}")
        return all_descriptors


    # def get_3d_descriptors(self, confId=-1, useAtomicMasses=True):
    #     if not self.mol:
    #         return None
    #
    #     descriptors = {}
    #     available_descriptors = {
    #         'Asphericity': Descriptors3D.Asphericity,
    #         'Eccentricity': Descriptors3D.Eccentricity,
    #         'InertialShapeFactor': Descriptors3D.InertialShapeFactor,
    #         'NPR1': Descriptors3D.NPR1,
    #         'NPR2': Descriptors3D.NPR2,
    #         'PMI1': Descriptors3D.PMI1,
    #         'PMI2': Descriptors3D.PMI2,
    #         'PMI3': Descriptors3D.PMI3,
    #         'RadiusOfGyration': Descriptors3D.RadiusOfGyration,
    #         'SpherocityIndex': Descriptors3D.SpherocityIndex
    #     }
    #
    #     for desc_name, desc_func in available_descriptors.items():
    #         try:
    #             descriptors[desc_name] = desc_func(self.mol, confId, useAtomicMasses)
    #         except ValueError as e:
    #             print(f"Error calculating {desc_name}: {e}")
    #
    #     return descriptors

    def calculate_shape_factors(self):
        if self.mol is None:
            return None, None

        if self.mol.GetNumConformers() == 0:
            print("No conformers present in the molecule.")
            return None, None

        conf = self.mol.GetConformer()
        coords = [conf.GetAtomPosition(i) for i in range(self.mol.GetNumAtoms())]
        coords = np.array([(p.x, p.y, p.z) for p in coords])

        centroid = np.mean(coords, axis=0)
        cov_matrix = np.cov((coords - centroid).T)
        eigenvalues = np.linalg.eigvals(cov_matrix)
        eigenvalues.sort()

        I1, I2, I3 = eigenvalues
        shape_index = (I1 * I2 * I3) / ((I1 + I2 + I3) ** 3)
        R = eigenvalues[-1] / eigenvalues[0]

        return shape_index, R

    # def calculate_other_descriptors(self):
    #     if not self.mol:
    #         return None
    #
    #     num_h_acceptors = Lipinski.NumHAcceptors(self.mol)
    #     num_h_donors = Lipinski.NumHDonors(self.mol)
    #     rotatable_bonds = Lipinski.NumRotatableBonds(self.mol)
    #     molecular_weight = Descriptors.MolWt(self.mol)
    #     min_partial_charge = Descriptors.MinPartialCharge(self.mol)
    #     max_partial_charge = Descriptors.MaxPartialCharge(self.mol)
    #     mol_volume = AllChem.ComputeMolVolume(self.mol, confId=-1, gridSpacing=0.2, boxMargin=2.0)
    #
    #     num_aromatic_heterocycles = Lipinski.NumAromaticHeterocycles(self.mol)
    #     num_aromatic_carbocycles = Lipinski.NumAromaticCarbocycles(self.mol)
    #     num_heterocycles = Lipinski.NumHeteroatoms(self.mol)
    #     num_lipinski_hba = Lipinski.NumHAcceptors(self.mol)
    #     num_lipinski_hbd = Lipinski.NumHDonors(self.mol)
    #
    #     return {
    #         "Num H Acceptors": num_h_acceptors,
    #         "Num H Donors": num_h_donors,
    #         "Rotatable Bonds": rotatable_bonds,
    #         "Molecular Weight": molecular_weight,
    #         "Min Partial Charge": min_partial_charge,
    #         "Max Partial Charge": max_partial_charge,
    #         "Molecular Volume": mol_volume,
    #         "Num Aromatic Heterocycles": num_aromatic_heterocycles,
    #         "Num Aromatic Carbocycles": num_aromatic_carbocycles,
    #         "Num Heteroatoms": num_heterocycles,
    #         "Num Lipinski HBA": num_lipinski_hba,
    #         "Num Lipinski HBD": num_lipinski_hbd,
    #     }

    def calculate_axis_lengths_and_ratios(self):

        # 计算主惯性矩
        pmi1 = rdMolDescriptors.CalcPMI1(self.mol)
        pmi2 = rdMolDescriptors.CalcPMI2(self.mol)
        pmi3 = rdMolDescriptors.CalcPMI3(self.mol)

        S, M, L = sorted([pmi1, pmi2, pmi3])

        S_L = S / L
        M_L = M / L
        S_M = S / M

        return S, M, L, S_L, M_L, S_M

    def GlobularityAndFrTPSA(self, cif_path, includeSandP=1):
        crystals = CrystalReader(cif_path)
        if not crystals:
            raise ValueError("No crystal found in CIF file.")
        crystal = crystals[0]
        mol_volume = crystal.molecule.molecular_volume

        r_sphere = (3 * mol_volume / (4 * math.pi)) ** (1 / 3)
        area_sphere = 4 * math.pi * r_sphere ** 2

        if not self.mol:
            return None, None

        radii = rdFreeSASA.classifyAtoms(self.mol)
        sasa = rdFreeSASA.CalcSASA(self.mol, radii)

        globularity = area_sphere / sasa if sasa != 0 else 0
        FrTPSA = Descriptors.TPSA(self.mol, includeSandP=includeSandP) / sasa if sasa != 0 else 0

        return globularity, FrTPSA

    def DipoleMoment(self, charge_model='eem2015bm'):
        mol_block = Chem.MolToMolBlock(self.mol)
        ob_mol = pybel.readstring('mol', mol_block)
        dipole = openbabel.OBChargeModel_FindType(charge_model).GetDipoleMoment(ob_mol.OBMol)
        dipole_moment = math.sqrt(dipole.GetX() ** 2 + dipole.GetY() ** 2 + dipole.GetZ() ** 2)
        return dipole_moment

    def FractionNO(self):
        num_heavy_atoms = self.mol.GetNumHeavyAtoms()
        if num_heavy_atoms != 0:
            return Descriptors.NOCount(self.mol) / float(num_heavy_atoms)
        else:
            return 0.0

    def FractionAromaticAtoms(self):
        num_heavy_atoms = self.mol.GetNumHeavyAtoms()
        if num_heavy_atoms == 0:
            return 0.0
        return len(self.mol.GetAromaticAtoms()) / float(num_heavy_atoms)

    def generate_and_calculate_rmsd(self, num_confs=100):
        if not self.mol:
            return None, None

        self.mol = Chem.AddHs(self.mol)
        _ = AllChem.EmbedMultipleConfs(self.mol, numConfs=num_confs, randomSeed=42)
        AllChem.MMFFOptimizeMoleculeConfs(self.mol)

        num_confs = self.mol.GetNumConformers()
        rmsd_matrix = np.zeros((num_confs, num_confs))
        for i in range(num_confs):
            for j in range(i + 1, num_confs):
                rmsd = AllChem.GetConformerRMS(self.mol, i, j)
                rmsd_matrix[i, j] = rmsd_matrix[j, i] = rmsd

        rmsd_average = np.mean(rmsd_matrix[np.nonzero(rmsd_matrix)])
        return rmsd_average

    # def get_fragment_counts(self):   # 基团
    #     fragment_functions = inspect.getmembers(Fragments, inspect.isfunction)
    #     fragment_counts = {}
    #     for name, func in fragment_functions:
    #         try:
    #             count = func(self.mol)
    #             fragment_counts[name] = count
    #         except Exception as e:
    #             fragment_counts[name] = None
    #     return fragment_counts



    def get_molecular_properties(self):
        shape_index, eccentricity = self.calculate_shape_factors()
        # other_descriptors = self.calculate_other_descriptors()
        S, M, L, S_L, M_L, S_M = self.calculate_axis_lengths_and_ratios()
        globularity, FrTPSA = self.GlobularityAndFrTPSA(self.cif_path, includeSandP=1)
        DipoleMoment = self.DipoleMoment()
        FractionNO = self.FractionNO()
        FractionAromaticAtoms = self.FractionAromaticAtoms()
        generate_and_calculate_rmsd = self.generate_and_calculate_rmsd()
        # fragment_counts = self.get_fragment_counts()
        estate_values = self.Estate_calculator()
        # descriptors_3d = self.get_3d_descriptors()
        all_descriptors = self.calculate_and_print_all_descriptors()

        return {
            "Shape Index": shape_index,
            "Eccentricity": eccentricity,
            'S_L': S_L,
            'M_L': M_L,
            'S_M': S_M,
            'S': S,
            'globularity': globularity,
            'FrTPSA': FrTPSA,
            'DipoleMoment': DipoleMoment,
            'FractionNO': FractionNO,
            'FractionAromaticAtoms': FractionAromaticAtoms,
            'generate_and_calculate_rmsd': generate_and_calculate_rmsd,
            # **descriptors_3d,
            # **fragment_counts,
            **estate_values,  # 将字典直接合并到属性字典中
            # **other_descriptors,
            **all_descriptors
        }
