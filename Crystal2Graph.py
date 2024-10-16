import os
import torch
import numpy as np
import subprocess
import tempfile
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import networkx as nx
from ccdc.io import CrystalReader
import traceback

att_dtype = np.float32
PeriodicTable = Chem.GetPeriodicTable()

try:
    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
except:
    fdefName = os.path.join('/RDKit file path**/RDKit/Data/', 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

possible_atom_type = ['H', 'B', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
possible_hybridization = ['S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'UNSPECIFIED']
possible_bond_type = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def donor_acceptor(rd_mol):
    is_donor = defaultdict(int)
    is_acceptor = defaultdict(int)

    feats = factory.GetFeaturesForMol(rd_mol)
    for feat in feats:
        if feat.GetFamily() == 'Donor': 
            for atom_id in feat.GetAtomIds():
                is_donor[atom_id] = 1  
        elif feat.GetFamily() == 'Acceptor': 
            for atom_id in feat.GetAtomIds():
                is_acceptor[atom_id] = 1 
    return is_donor, is_acceptor

def convert_cif_to_sdf_using_obabel(cif_path, sdf_path):
    try:
        subprocess.run(["obabel", "-icif", cif_path, "-osdf", "-O", sdf_path, "-h"], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting CIF to SDF using Open Babel: {e}")
        return False

def cif_to_rdkit_mol_with_spatial_info(cif_path):
    with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as temp_sdf:
        sdf_path = temp_sdf.name

    if not convert_cif_to_sdf_using_obabel(cif_path, sdf_path):
        os.remove(sdf_path)  
        return None

    mol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]
    os.remove(sdf_path) 

    if mol:
        return mol
    else:
        print("Failed to create RDKit Mol from SDF.")
        return None

def remove_isolated_atoms(mol):
    emol = Chem.RWMol(mol)
    to_remove = []

    for atom in emol.GetAtoms():
        if atom.GetDegree() == 0:
            to_remove.append(atom.GetIdx())

    for idx in sorted(to_remove, reverse=True):
        emol.RemoveAtom(idx) 

    mol = emol.GetMol()
    Chem.SanitizeMol(mol)
    return mol

def get_electronegativity(symbol):
    return electronegativity_table.get(symbol, 0.0)  

def AtomAttributes(rd_atom, is_donor, is_acceptor, extra_attributes=[]):
    rd_idx = rd_atom.GetIdx()  
    attributes = []

    attributes += one_of_k_encoding(rd_atom.GetSymbol(), possible_atom_type)  
    attributes += one_of_k_encoding(len(rd_atom.GetNeighbors()), [0, 1, 2, 3, 4, 5, 6])  
    attributes += one_of_k_encoding(rd_atom.GetTotalNumHs(includeNeighbors=True), [0, 1, 2, 3, 4])  
    attributes += one_of_k_encoding(rd_atom.GetHybridization().__str__(), possible_hybridization)  
    attributes += one_of_k_encoding(int(rd_atom.GetChiralTag()), [0, 1, 2, 3])  
    attributes.append(rd_atom.IsInRing())  
    attributes.append(rd_atom.GetIsAromatic()) 
    attributes.append(is_donor.get(rd_idx, 0)) 
    attributes.append(is_acceptor.get(rd_idx, 0))  

    # attributes.append(rd_atom.GetAtomicNum())  
    # attributes.append(rd_atom.GetFormalCharge())  
    # attributes.append(rd_atom.GetTotalValence()) 
    # attributes.append(rd_atom.GetTotalDegree())  
    # attributes.append(rd_atom.GetNumRadicalElectrons()) 
    # attributes.append(get_electronegativity(rd_atom.GetSymbol()))  

    attributes += extra_attributes
    return np.array(attributes, dtype=att_dtype)

def atom_featurizer(mol):
    is_donor, is_acceptor = donor_acceptor(mol)
    V = []
    for atom in mol.GetAtoms():
        all_atom_attr = AtomAttributes(atom, is_donor, is_acceptor)
        V.append(all_atom_attr)
    return np.array(V, dtype=att_dtype)

def get_bond_features_from_mol(mol):
    original_edge_idx, original_edge_feats = [], []
    for b in mol.GetBonds():
        start = b.GetBeginAtomIdx()  
        end = b.GetEndAtomIdx()  
        start_atom = mol.GetAtomWithIdx(start)  
        end_atom = mol.GetAtomWithIdx(end)  
        start_symbol = start_atom.GetSymbol() 
        end_symbol = end_atom.GetSymbol()  

        b_type = one_of_k_encoding(b.GetBondType().__str__(), possible_bond_type)

        is_conjugated = b.GetIsConjugated()
        is_in_ring = b.IsInRing()
        b_type.append(is_conjugated)
        b_type.append(is_in_ring)

        # print(f"Bond: {start_symbol} ({start}) - {end_symbol} ({end}), Features: {b_type}")

        original_edge_idx.append([start, end])
        original_edge_idx.append([end, start])

        original_edge_feats.append(b_type)
        original_edge_feats.append(b_type)

    e_sorted_idx = sorted(range(len(original_edge_idx)), key=lambda k: original_edge_idx[k])
    original_edge_idx = np.array(original_edge_idx)[e_sorted_idx]
    original_edge_feats = np.array(original_edge_feats, dtype=np.float32)[e_sorted_idx]

    return original_edge_idx.astype(np.int64), original_edge_feats.astype(np.float32)


def fractional_to_cartesian(fractional_coords, cell_lengths, cell_angles):
    a, b, c = cell_lengths  
    alpha, beta, gamma = np.radians(cell_angles) 

    v = np.sqrt(1 - np.cos(alpha) ** 2 - np.cos(beta) ** 2 - np.cos(gamma) ** 2 +
                2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma))

    volume_factor = a * b * c * v

    matrix = np.zeros((3, 3)) 
    matrix[0, 0] = a  
    matrix[0, 1] = b * np.cos(gamma)  
    matrix[0, 2] = c * np.cos(beta)  
    matrix[1, 1] = b * np.sin(gamma)  
    matrix[1, 2] = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)  
    matrix[2, 2] = volume_factor / (a * b * np.sin(gamma))  

    cartesian_coords = np.dot(matrix, fractional_coords)
    return cartesian_coords


def get_symmetric_atoms(crystal):
    supercell_atoms = []

    symmetry_operators = crystal.symmetry_operators
    # print(symmetry_operators)

    for symmop in symmetry_operators:
        symm_molecule = crystal.symmetric_molecule(symmop)

        for atom in symm_molecule.atoms:
            supercell_atoms.append((atom.label, atom.fractional_coordinates))
            # supercell_atoms.append((atom.label, atom.coordinates))

    return supercell_atoms


def expand_supercell(supercell_atoms, cell_lengths, cell_angles, a_times=1, b_times=1, c_times=1):

    expanded_atom_positions = []  

    for a in range(a_times):
        for b in range(b_times):
            for c in range(c_times):
                translation_vector = np.array([a, b, c]) * cell_lengths

                for atom_index, (label, coords) in enumerate(supercell_atoms):
                    fractional_coords = np.array([coords.x, coords.y, coords.z])
                    cartesian_coords = fractional_to_cartesian(fractional_coords, cell_lengths, cell_angles)

                    if cartesian_coords.shape == (3,):
                        new_position = cartesian_coords + translation_vector
                        expanded_atom_positions.append((atom_index, new_position))
                    else:
                        raise ValueError(f"Unexpected shape for cartesian_coords: {cartesian_coords.shape}")

    return expanded_atom_positions

def calculate_supercell_replications(cell_lengths, cell_angles):
    return 3, 3, 3  

def get_number_of_units_per_cell(Z, Z_prime):
    num_asymmetric_units = int(Z / Z_prime)
    return num_asymmetric_units  


def expand_molecular_graph(mol, supercell_atoms, cell_lengths, cell_angles, Z, Z_prime):
    is_donor, is_acceptor = donor_acceptor(mol)
    original_atom_features = atom_featurizer(mol)

    num_asymmetric_units = int(Z / Z_prime)
    num_original_atoms = original_atom_features.shape[0]  

    cell_atom_features = np.tile(original_atom_features, (num_asymmetric_units, 1))

    expanded_atom_features = []

    for i in range(len(supercell_atoms)):
        _, fractional_coords = supercell_atoms[i]
        atom_index = i % num_original_atoms
        atom_features = cell_atom_features[atom_index]

        extended_features = np.concatenate((atom_features, fractional_coords))
        expanded_atom_features.append(extended_features)

    expanded_atom_features = np.array(expanded_atom_features)

    original_edge_idx, original_edge_feats = get_bond_features_from_mol(mol)
    expanded_edge_idx = []
    expanded_edge_feats = []

    a_times, b_times, c_times = 3, 3, 3
    num_units_per_cell = get_number_of_units_per_cell(Z, Z_prime)

    for a in range(a_times):
        for b in range(b_times):
            for c in range(c_times):
                for unit in range(num_units_per_cell):
                    cell_offset = (a * b_times * c_times + b * c_times + c) * num_units_per_cell + unit

                    for (start, end), edge_feat in zip(original_edge_idx, original_edge_feats):
                        start_offset = start + cell_offset * num_original_atoms
                        end_offset = end + cell_offset * num_original_atoms

                        expanded_edge_idx.append([start_offset, end_offset])
                        expanded_edge_feats.append(edge_feat)

    expanded_edge_idx = np.array(expanded_edge_idx)
    expanded_edge_feats = np.array(expanded_edge_feats)

    return expanded_atom_features, expanded_edge_idx, expanded_edge_feats


def get_center_molecule_features(expanded_atom_features, expanded_edge_idx, expanded_edge_feats, num_original_atoms):
    total_atoms = expanded_atom_features.shape[0]
    num_molecules = total_atoms // num_original_atoms
    center_molecule_index = num_molecules // 2
    start_atom_idx = center_molecule_index * num_original_atoms
    end_atom_idx = (center_molecule_index + 1) * num_original_atoms
    center_atom_features = expanded_atom_features[start_atom_idx:end_atom_idx]

    center_edge_idx = []
    center_edge_feats = []
    for idx, (start, end) in enumerate(expanded_edge_idx):
        if start_atom_idx <= start < end_atom_idx and start_atom_idx <= end < end_atom_idx:
            center_edge_idx.append([start - start_atom_idx, end - start_atom_idx])
            center_edge_feats.append(expanded_edge_feats[idx])

    center_edge_idx = np.array(center_edge_idx)
    center_edge_feats = np.array(center_edge_feats)

    return center_atom_features, center_edge_idx, center_edge_feats

def calculate_vdw_radii(atom_label):
    """获取原子的范德华半径，单位是Ångström。"""
    vdw_radii = {
        'H': 1.20, 'He': 1.40, 'Li': 1.82, 'Be': 1.53, 'B': 1.92, 'C': 1.70,
        'N': 1.55, 'O': 1.52, 'F': 1.47, 'Ne': 1.54, 'Na': 2.27, 'Mg': 1.73,
        'Al': 1.84, 'Si': 2.10, 'P': 1.80, 'S': 1.80, 'Cl': 1.75, 'Ar': 1.88,
        'K': 2.75, 'Ca': 2.31, 'Sc': 2.30, 'Ti': 2.15, 'V': 2.05, 'Cr': 2.05,
        'Mn': 2.05, 'Fe': 2.05, 'Co': 2.00, 'Ni': 2.00, 'Cu': 2.00, 'Zn': 2.10,
        'Ga': 1.87, 'Ge': 2.11, 'As': 1.85, 'Se': 1.90, 'Br': 1.85, 'Kr': 2.02,
        'Rb': 3.03, 'Sr': 2.49, 'Y': 2.40, 'Zr': 2.30, 'Nb': 2.15, 'Mo': 2.10,
        'Tc': 2.05, 'Ru': 2.05, 'Rh': 2.00, 'Pd': 2.05, 'Ag': 2.10, 'Cd': 2.20,
        'In': 2.20, 'Sn': 2.25, 'Sb': 2.20, 'Te': 2.20, 'I': 2.15, 'Xe': 2.16,
        'Cs': 3.43, 'Ba': 2.68, 'La': 2.50, 'Ce': 2.48, 'Pr': 2.47, 'Nd': 2.45,
        'Pm': 2.43, 'Sm': 2.42, 'Eu': 2.40, 'Gd': 2.38, 'Tb': 2.37, 'Dy': 2.35,
        'Ho': 2.33, 'Er': 2.32, 'Tm': 2.30, 'Yb': 2.28, 'Lu': 2.27, 'Hf': 2.25,
        'Ta': 2.20, 'W': 2.10, 'Re': 2.05, 'Os': 2.00, 'Ir': 2.00, 'Pt': 2.05,
        'Au': 2.10, 'Hg': 2.05, 'Tl': 2.20, 'Pb': 2.30, 'Bi': 2.30, 'Po': 2.40,
        'At': 2.40, 'Rn': 2.40, 'Fr': 2.40, 'Ra': 2.40, 'Ac': 2.40, 'Th': 2.40,
        'Pa': 2.40, 'U': 2.40, 'Np': 2.40, 'Pu': 2.40, 'Am': 2.40, 'Cm': 2.40,
        'Bk': 2.40, 'Cf': 2.40, 'Es': 2.40, 'Fm': 2.40, 'Md': 2.40, 'No': 2.40,
        'Lr': 2.40, 'Rf': 2.40, 'Db': 2.40, 'Sg': 2.40, 'Bh': 2.40, 'Hs': 2.40,
        'Mt': 2.40, 'Ds': 2.40, 'Rg': 2.40, 'Cn': 2.40, 'Nh': 2.40, 'Fl': 2.40,
        'Mc': 2.40, 'Lv': 2.40, 'Ts': 2.40, 'Og': 2.40
    }
    return vdw_radii.get(atom_label, 2.0)  


def get_other_molecules_features(expanded_atom_features, expanded_edge_idx, expanded_edge_feats, num_original_atoms):
    total_atoms = expanded_atom_features.shape[0]  
    num_molecules = total_atoms // num_original_atoms  

    all_molecule_atom_features = []
    all_molecule_edge_idx = []
    all_molecule_edge_feats = []

    for mol_idx in range(num_molecules):
        if mol_idx == num_molecules // 2:
            continue  

        start_atom_idx = mol_idx * num_original_atoms
        end_atom_idx = (mol_idx + 1) * num_original_atoms

        molecule_atom_features = expanded_atom_features[start_atom_idx:end_atom_idx]
        all_molecule_atom_features.append(molecule_atom_features)

        molecule_edge_idx = []
        molecule_edge_feats = []

        for idx, (start, end) in enumerate(expanded_edge_idx):
            if start_atom_idx <= start < end_atom_idx and start_atom_idx <= end < end_atom_idx:
                molecule_edge_idx.append([start - start_atom_idx, end - start_atom_idx])
                molecule_edge_feats.append(expanded_edge_feats[idx])
                
        all_molecule_edge_idx.append(np.array(molecule_edge_idx))
        all_molecule_edge_feats.append(np.array(molecule_edge_feats))

    all_molecule_atom_features = np.concatenate(all_molecule_atom_features, axis=0)
    all_molecule_edge_feats = np.concatenate(all_molecule_edge_feats, axis=0)
    all_molecule_edge_idx = np.concatenate(all_molecule_edge_idx, axis=0)

    return all_molecule_atom_features, all_molecule_edge_idx, all_molecule_edge_feats


def extract_atom_label(attributes):
    possible_atom_type = ['H', 'B', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
    for i, is_type in enumerate(attributes[:len(possible_atom_type)]):
        if np.any(is_type):  
            return possible_atom_type[i]
    return 'Unknown'  


def check_vdw_contact(center_atom_features, other_atom_features):
    contacts = []  

    for center_feat in center_atom_features:
        center_label = extract_atom_label(center_feat)
        center_coords = center_feat[-3:]
        center_vdw_radius = calculate_vdw_radii(center_label)

        for other_feat in other_atom_features:
            # for feat in other_feat:
            other_label = extract_atom_label(other_feat)
            other_coords = np.array(other_feat[-3:], dtype=float)
            other_vdw_radius = calculate_vdw_radii(other_label)
            try:
                distance = np.linalg.norm(center_coords - other_coords)
            except ValueError as e:
                print(f"Error calculating distance: {e}")
                continue  # 跳过这个配对，继续下一个
            contact_distance = center_vdw_radius + other_vdw_radius
            if distance < contact_distance:
                if (center_label == 'C' and other_label == 'C') or (center_label == 'H' and other_label == 'H') or (center_label == 'C' and other_label == 'H') or (center_label == 'H' and other_label == 'C'):
                # if (center_label == 'C' and other_label == 'C') or (center_label == 'H' and other_label == 'H'):
                    continue
                contacts.append((center_label, other_label, distance, contact_distance))
    return contacts 


def add_contacts_to_graph(center_atom_features, center_edge_idx, center_edge_feats, other_atom_features):
    possible_atom_type = ['H', 'B', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
    contacts = check_vdw_contact(center_atom_features, other_atom_features)
    new_edge_idx = []
    new_edge_feats = []

    new_center_edge_feats = []
    for edge_feats in center_edge_feats:
        expanded_feat = list(edge_feats) + [0] 
        new_center_edge_feats.append(expanded_feat)

    center_atom_features = np.array(center_atom_features)
    other_atom_features = np.array(other_atom_features)

    new_other_atom_features = np.array([feat for other_feat in other_atom_features for feat in other_feat])
    new_other_atom_features = new_other_atom_features.reshape(-1, center_atom_features.shape[1])

    # print(f"New other atom features shape: {new_other_atom_features.shape}")
    if new_other_atom_features.ndim != 2:
        raise ValueError("new_other_atom_features should be a 2-dimensional array")

    center_labels = center_atom_features[:, :len(possible_atom_type)].argmax(axis=1)  
    other_labels = new_other_atom_features[:, :len(possible_atom_type)].argmax(axis=1)

    # print(f"Center labels: {center_labels}")
    # print(f"Other labels: {other_labels}")

    # print(f"Contacts: {contacts}")

    for center_label, other_label, distance, contact_distance in contacts:
        center_indices = np.where(center_labels == possible_atom_type.index(center_label))[0]
        other_indices = np.where(other_labels == possible_atom_type.index(other_label))[0]

        if center_indices.size == 0 or other_indices.size == 0:
            print(f"No matching atom found for labels {center_label} or {other_label}")
            continue  

        center_idx = center_indices[0]
        other_idx = other_indices[0] + len(center_atom_features)

        other_idx = other_idx % len(center_atom_features)
        new_edge_idx.append([center_idx, other_idx])
        new_edge_feats.append([0] * (len(new_center_edge_feats[0]) - 1) + [1.0])

        new_edge_idx.append([other_idx, center_idx])
        new_edge_feats.append([0] * (len(new_center_edge_feats[0]) - 1) + [1.0])

    extended_edge_feats = np.array(new_center_edge_feats)
    new_edge_idx = np.array(new_edge_idx, dtype=np.int64)

    if new_edge_idx.ndim != 2:
        raise ValueError("new_edge_idx should be a 2-dimensional array")

    updated_edge_idx = np.concatenate((np.array(center_edge_idx), new_edge_idx), axis=0)
    updated_edge_feats = np.concatenate((extended_edge_feats, np.array(new_edge_feats)), axis=0)

    return updated_edge_idx, updated_edge_feats


class Crystal2Graph:
    def __init__(self, cif_path, refcode, a_times=1, b_times=1, c_times=1):
        self.cif_path = cif_path
        self.refcode = refcode
        self.a_times = a_times
        self.b_times = b_times
        self.c_times = c_times

        try:
            with CrystalReader(cif_path) as reader:
                crystal = reader.crystal(refcode)

                self.Z = crystal.z_value
                self.Z_prime = crystal.z_prime

            self.mol = self.cif_to_rdkit_mol_with_spatial_info()
            self.mol = remove_isolated_atoms(self.mol)  # Remove isolated atoms here

            self.is_donor, self.is_acceptor = donor_acceptor(self.mol)
            self.original_atom_features = atom_featurizer(self.mol)
            self.edge_idx, self.edge_feats = get_bond_features_from_mol(self.mol)

            self.crystal = self.load_crystal()
            self.cell_atoms = self.get_symmetric_atoms()
            self.supercell_atoms = self.expand_supercell()
            self.expanded_atom_features, self.expanded_edge_idx, self.expanded_edge_feats = self.expand_molecular_graph()
            self.center_atom_features, self.center_edge_idx, self.center_edge_feats = self.get_center_molecule_features()
            self.other_molecule_atom_features, self.other_molecule_edge_idx, self.other_molecule_edge_feats = self.get_other_molecules_features()

            # print(f"Other molecule atom features shape: {self.other_molecule_atom_features.shape}")
            # print(f"Other molecule edge idx shape: {self.other_molecule_edge_idx.shape}")
            # print(f"Other molecule edge feats shape: {self.other_molecule_edge_feats.shape}")

            self.center_atom_features_without_coords = self.center_atom_features[:, :-3]
            self.self_connection_center_edge_idx, self.self_connection_center_edge_feats = self.center_molecule_self_connection_graph()
            
        except Exception as e:
            print(f"Warning: Failed to process file {cif_path}. Error: {e}")
            traceback.print_exc()

    def cif_to_rdkit_mol_with_spatial_info(self):
        mol = cif_to_rdkit_mol_with_spatial_info(self.cif_path)
        if not mol:
            raise ValueError("Failed to create RDKit Mol from SDF.")
        return mol

    def load_crystal(self):
        crystal_reader = CrystalReader(self.cif_path)
        return crystal_reader[0]  # Assuming the CIF file contains only one crystal structure

    def get_symmetric_atoms(self):
        return get_symmetric_atoms(self.crystal)

    def expand_supercell(self):
        cell_lengths = self.crystal.cell_lengths
        cell_angles = self.crystal.cell_angles
        return expand_supercell(self.cell_atoms, cell_lengths, cell_angles, self.a_times, self.b_times, self.c_times)

    def expand_molecular_graph(self):
        cell_lengths = self.crystal.cell_lengths
        cell_angles = self.crystal.cell_angles
        return expand_molecular_graph(self.mol, self.supercell_atoms, cell_lengths, cell_angles, self.Z, self.Z_prime)

    def get_expanded_features(self):
        return self.expanded_atom_features, self.expanded_edge_idx, self.expanded_edge_feats

    def get_center_molecule_features(self):
        num_original_atoms = self.original_atom_features.shape[0]
        return get_center_molecule_features(self.expanded_atom_features, self.expanded_edge_idx, self.expanded_edge_feats, num_original_atoms)

    def calculate_vdw_radii(self, atom_label):
        return calculate_vdw_radii(atom_label)

    def get_other_molecules_features(self):
        num_original_atoms = self.original_atom_features.shape[0]
        return get_other_molecules_features(self.expanded_atom_features, self.expanded_edge_idx, self.expanded_edge_feats, num_original_atoms)

    def center_molecule_self_connection_graph(self):
        return add_contacts_to_graph(self.center_atom_features, self.center_edge_idx, self.center_edge_feats, self.other_molecule_atom_features)

def visualize_molecule_rdkit(mol):
    img = Draw.MolToImage(mol, size=(300, 300))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def visualize_graph(nodes, edges):
    G = nx.Graph()

    for i, node in enumerate(nodes):
        G.add_node(i, features=node)

    for edge in edges:
        G.add_edge(edge[0], edge[1])

    if nx.is_connected(G):
        print("The graph is connected.")
    else:
        print("The graph is not connected.")
        connected_components = list(nx.connected_components(G))
        print(f"The graph has {len(connected_components)} connected components.")

    pos = nx.spring_layout(G, k=0.1, iterations=100, seed=42)
    plt.figure(figsize=(12, 12))
    nx.draw(G, pos, with_labels=False, node_color='lightblue', edge_color='gray', node_size=300, width=1.5)
    plt.show()

def visualize_atoms(expanded_atoms):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract atomic labels and positions
    labels, positions = zip(*expanded_atoms)
    positions = np.array(positions)

    # Extract x, y, z coordinates
    xs, ys, zs = positions[:, 0], positions[:, 1], positions[:, 2]

    # Plot scatter plot
    scatter = ax.scatter(xs, ys, zs, c='b', marker='o')

    # Annotate atoms
    for label, x, y, z in zip(labels, xs, ys, zs):
        ax.text(x, y, z, label, size=10, zorder=1, color='k')

    # Set labels and title
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title('3D Visualization of Expanded Supercell Atoms')

    plt.show()


# Main program
if __name__ == '__main__':
    cif_path = 'AABHTZ.cif'
    refcode = "AABHTZ"
    a_times, b_times, c_times = 3, 3, 3

    c2g = Crystal2Graph(cif_path, refcode, a_times, b_times, c_times)
    expanded_atom_features, expanded_edge_idx, expanded_edge_feats = c2g.get_expanded_features()

    np.set_printoptions(threshold=np.inf, linewidth=200)
    print(c2g.original_atom_features)
    print(c2g.edge_idx)
    print(c2g.edge_feats)

    expanded_atoms = c2g.supercell_atoms

    if torch.cuda.is_available():
        print("CUDA (GPU support) is available and enabled!")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA (GPU support) is not available. Using CPU.")

    num_original_atoms = c2g.original_atom_features.shape[0]
    center_atom_features, center_edge_idx, center_edge_feats = c2g.get_center_molecule_features()

    print('center_atom_features\n', center_atom_features)
    print('center_edge_idx\n', center_edge_idx)
    print('center_edge_feats\n', center_edge_feats)

    self_connection_center_edge_idx, self_connection_center_edge_feats = c2g.self_connection_center_edge_idx, c2g.self_connection_center_edge_feats
    center_atom_features_without_coords = c2g.center_atom_features_without_coords
    print('center_atom_features_without_coords\n', center_atom_features_without_coords)
    print('self_connection_center_edge_idx\n',self_connection_center_edge_idx)
    print('self_connection_center_edge_feats\n',self_connection_center_edge_feats)
