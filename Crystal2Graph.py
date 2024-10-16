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
    # 使用列表推导式对输入进行独热编码
    return list(map(lambda s: x == s, allowable_set))


# 提取供体和受体特征的函数
def donor_acceptor(rd_mol):
    # 使用 defaultdict 初始化供体和受体字典，默认值为0
    is_donor = defaultdict(int)
    is_acceptor = defaultdict(int)

    # 从全局变量 factory 获取分子的特征
    feats = factory.GetFeaturesForMol(rd_mol)
    for feat in feats:
        if feat.GetFamily() == 'Donor':  # 如果特征是供体
            for atom_id in feat.GetAtomIds():
                is_donor[atom_id] = 1  # 将相应原子的供体标志设为1
        elif feat.GetFamily() == 'Acceptor':  # 如果特征是受体
            for atom_id in feat.GetAtomIds():
                is_acceptor[atom_id] = 1  # 将相应原子的受体标志设为1
    return is_donor, is_acceptor


# 使用 Open Babel 将 CIF 文件转换为 SDF 文件的函数
def convert_cif_to_sdf_using_obabel(cif_path, sdf_path):
    try:
        # 调用 Open Babel 命令行工具进行文件转换
        subprocess.run(["obabel", "-icif", cif_path, "-osdf", "-O", sdf_path, "-h"], check=True)
        return True
    except subprocess.CalledProcessError as e:
        # 捕获转换过程中出现的异常并打印错误信息
        print(f"Error converting CIF to SDF using Open Babel: {e}")
        return False


# 将 CIF 文件转换为 RDKit 分子对象并保留空间信息的函数
def cif_to_rdkit_mol_with_spatial_info(cif_path):
    # 创建一个临时 SDF 文件
    with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as temp_sdf:
        sdf_path = temp_sdf.name

    # 转换 CIF 文件为 SDF 文件
    if not convert_cif_to_sdf_using_obabel(cif_path, sdf_path):
        os.remove(sdf_path)  # 转换失败时删除临时文件
        return None

    # 使用 RDKit 读取 SDF 文件，保留氢原子信息
    mol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]
    os.remove(sdf_path)  # 删除临时 SDF 文件

    # 检查是否成功创建 RDKit 分子对象
    if mol:
        return mol
    else:
        print("Failed to create RDKit Mol from SDF.")
        return None


# 移除孤立原子的函数
def remove_isolated_atoms(mol):
    # 将传入的 RDKit 分子对象转换为可编辑的 RWMol 对象
    emol = Chem.RWMol(mol)

    # 用于存储需要移除的孤立原子的索引列表
    to_remove = []

    # 遍历分子中的所有原子
    for atom in emol.GetAtoms():
        # 如果原子的度数为0（即该原子没有与其他原子形成键），则将其索引添加到 to_remove 列表中
        if atom.GetDegree() == 0:
            to_remove.append(atom.GetIdx())

    # 反向排序要移除的原子索引列表，以便从分子中删除原子时不会影响其他原子的索引
    for idx in sorted(to_remove, reverse=True):
        emol.RemoveAtom(idx)  # 从分子中移除指定索引的原子

    # 生成更新后的分子对象
    mol = emol.GetMol()

    # 通过 RDKit 的 SanitizeMol 函数对分子进行消毒，确保分子处于一致的状态
    Chem.SanitizeMol(mol)
    return mol

def get_electronegativity(symbol):
    return electronegativity_table.get(symbol, 0.0)  # 如果表中没有该元素，返回默认值0.0


# 修改后的 AtomAttributes 函数
def AtomAttributes(rd_atom, is_donor, is_acceptor, extra_attributes=[]):
    rd_idx = rd_atom.GetIdx()  # 获取原子索引
    attributes = []

    attributes += one_of_k_encoding(rd_atom.GetSymbol(), possible_atom_type)  # 独热编码原子符号，并添加到属性列表中
    attributes += one_of_k_encoding(len(rd_atom.GetNeighbors()), [0, 1, 2, 3, 4, 5, 6])  # 独热编码原子的邻居数量，并添加到属性列表中
    attributes += one_of_k_encoding(rd_atom.GetTotalNumHs(includeNeighbors=True), [0, 1, 2, 3, 4])  # 独热编码原子的总氢原子数量（包括邻居的氢原子），并添加到属性列表中
    attributes += one_of_k_encoding(rd_atom.GetHybridization().__str__(), possible_hybridization)  # 独热编码原子的杂化状态，并添加到属性列表中
    attributes += one_of_k_encoding(int(rd_atom.GetChiralTag()), [0, 1, 2, 3])  # 独热编码原子的手性标签，并添加到属性列表中
    # 0：CHI_UNSPECIFIED，表示未指定手性中心。
    # 1：CHI_TETRAHEDRAL_CW，表示顺时针手性中心。
    # 2：CHI_TETRAHEDRAL_CCW，表示逆时针手性中心。
    # 3：其他手性状态（在某些情况下可能有）。
    attributes.append(rd_atom.IsInRing())  # 添加原子是否在环中
    attributes.append(rd_atom.GetIsAromatic())  # 添加原子是否是芳香性的
    attributes.append(is_donor.get(rd_idx, 0))  # 添加原子是否是供体
    attributes.append(is_acceptor.get(rd_idx, 0))  # 添加原子是否是受体

    # 你可以启用以下附加属性以增加更多特征
    # attributes.append(rd_atom.GetAtomicNum())  # 原子序数
    # attributes.append(rd_atom.GetFormalCharge())  # 形式电荷
    # attributes.append(rd_atom.GetTotalValence())  # 显式价态
    # attributes.append(rd_atom.GetTotalDegree())  # 原子的度数
    # attributes.append(rd_atom.GetNumRadicalElectrons())  # 自由电子对数量
    # attributes.append(get_electronegativity(rd_atom.GetSymbol()))  # 电负性

    attributes += extra_attributes
    return np.array(attributes, dtype=att_dtype)


# 分子原子特征提取函数
def atom_featurizer(mol):
    # 获取供体和受体信息
    is_donor, is_acceptor = donor_acceptor(mol)
    V = []
    for atom in mol.GetAtoms():
        # 提取每个原子的所有属性
        all_atom_attr = AtomAttributes(atom, is_donor, is_acceptor)
        V.append(all_atom_attr)
    return np.array(V, dtype=att_dtype)


# 从分子中获取键特征的函数
def get_bond_features_from_mol(mol):
    original_edge_idx, original_edge_feats = [], []
    for b in mol.GetBonds():
        start = b.GetBeginAtomIdx()  # 获取键的起始原子索引
        end = b.GetEndAtomIdx()  # 获取键的结束原子索引
        start_atom = mol.GetAtomWithIdx(start)  # 获取起始原子
        end_atom = mol.GetAtomWithIdx(end)  # 获取结束原子
        start_symbol = start_atom.GetSymbol()  # 获取起始原子的符号
        end_symbol = end_atom.GetSymbol()  # 获取结束原子的符号

        # 独热编码键类型
        b_type = one_of_k_encoding(b.GetBondType().__str__(), possible_bond_type)

        # 将共轭性和芳香性特征单独处理并添加到键特征中
        is_conjugated = b.GetIsConjugated()
        is_in_ring = b.IsInRing()
        b_type.append(is_conjugated)
        b_type.append(is_in_ring)

        # 打印键信息和特征（调试用）
        # print(f"Bond: {start_symbol} ({start}) - {end_symbol} ({end}), Features: {b_type}")

        # 将键的起始和结束索引添加到边索引列表中（双向）
        original_edge_idx.append([start, end])
        original_edge_idx.append([end, start])

        # 将键特征添加到边特征列表中（双向）
        original_edge_feats.append(b_type)
        original_edge_feats.append(b_type)

    # 根据边索引排序
    e_sorted_idx = sorted(range(len(original_edge_idx)), key=lambda k: original_edge_idx[k])
    original_edge_idx = np.array(original_edge_idx)[e_sorted_idx]
    original_edge_feats = np.array(original_edge_feats, dtype=np.float32)[e_sorted_idx]

    # 返回边索引和边特征
    return original_edge_idx.astype(np.int64), original_edge_feats.astype(np.float32)


# def fractional_to_cartesian(fractional_coords, cell_lengths, cell_angles):
#     # 提取晶胞的长度和角度参数
#     a, b, c = cell_lengths  # 晶胞的三条边长
#     alpha, beta, gamma = np.radians(cell_angles)  # 将角度从度转换为弧度
#
#     # 计算转换矩阵中的体积因子
#     v = np.sqrt(1 - np.cos(alpha) ** 2 - np.cos(beta) ** 2 - np.cos(gamma) ** 2 +
#                 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma))
#
#     # 计算体积因子，用于确定矩阵的c列
#     volume_factor = a * b * c * v
#
#     # 初始化转换矩阵，将晶胞参数转换为笛卡尔坐标
#     matrix = np.zeros((3, 3))  # 初始化3x3的矩阵
#     matrix[0, 0] = a  # 矩阵的第一个元素是晶胞的a边
#     matrix[0, 1] = b * np.cos(gamma)  # b边投影在x轴上的分量
#     matrix[0, 2] = c * np.cos(beta)  # c边投影在x轴上的分量
#     matrix[1, 1] = b * np.sin(gamma)  # b边在y轴方向的分量
#     matrix[1, 2] = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)  # c边在y轴方向的分量
#     matrix[2, 2] = volume_factor / (a * b * np.sin(gamma))  # c边在z轴方向的分量
#
#     # 使用转换矩阵将分数坐标转换为笛卡尔坐标
#     cartesian_coords = np.dot(matrix, fractional_coords)
#     return cartesian_coords

def fractional_to_cartesian(fractional_coords, cell_lengths, cell_angles):
    # 提取晶胞的长度和角度参数
    a, b, c = cell_lengths  # 晶胞的三条边长
    alpha, beta, gamma = np.radians(cell_angles)  # 将角度从度转换为弧度

    # 计算转换矩阵中的体积因子
    v = np.sqrt(1 - np.cos(alpha) ** 2 - np.cos(beta) ** 2 - np.cos(gamma) ** 2 +
                2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma))

    # 计算体积因子，用于确定矩阵的c列
    volume_factor = a * b * c * v

    # 初始化转换矩阵，将晶胞参数转换为笛卡尔坐标
    matrix = np.zeros((3, 3))  # 初始化3x3的矩阵
    matrix[0, 0] = a  # 矩阵的第一个元素是晶胞的a边
    matrix[0, 1] = b * np.cos(gamma)  # b边投影在x轴上的分量
    matrix[0, 2] = c * np.cos(beta)  # c边投影在x轴上的分量
    matrix[1, 1] = b * np.sin(gamma)  # b边在y轴方向的分量
    matrix[1, 2] = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)  # c边在y轴方向的分量
    matrix[2, 2] = volume_factor / (a * b * np.sin(gamma))  # c边在z轴方向的分量

    # 使用转换矩阵将分数坐标转换为笛卡尔坐标
    cartesian_coords = np.dot(matrix, fractional_coords)
    return cartesian_coords


def get_symmetric_atoms(crystal):
    supercell_atoms = []

    # 获取晶体结构的所有对称操作
    symmetry_operators = crystal.symmetry_operators
    # print(symmetry_operators)

    # 遍历每个对称操作并生成相应的分子副本
    for symmop in symmetry_operators:
        symm_molecule = crystal.symmetric_molecule(symmop)

        # 获取副本分子中的所有原子
        for atom in symm_molecule.atoms:
            supercell_atoms.append((atom.label, atom.fractional_coordinates))
            # supercell_atoms.append((atom.label, atom.coordinates))

    return supercell_atoms


def expand_supercell(supercell_atoms, cell_lengths, cell_angles, a_times=1, b_times=1, c_times=1):

    expanded_atom_positions = []  # 存储扩展后所有原子的位置信息

    # 遍历每个方向的复制次数
    for a in range(a_times):
        for b in range(b_times):
            for c in range(c_times):
                # 计算当前平移向量，根据晶胞边长和复制次数确定
                translation_vector = np.array([a, b, c]) * cell_lengths

                # 对于超胞中的每个原子，计算其新位置
                for atom_index, (label, coords) in enumerate(supercell_atoms):
                    # 提取原子的分数坐标
                    fractional_coords = np.array([coords.x, coords.y, coords.z])

                    # 将分数坐标转换为笛卡尔坐标
                    cartesian_coords = fractional_to_cartesian(fractional_coords, cell_lengths, cell_angles)

                    # 检查转换后的笛卡尔坐标是否为3维向量
                    if cartesian_coords.shape == (3,):
                        # 计算每个原子在超胞中的新位置（原位置加上平移向量）
                        new_position = cartesian_coords + translation_vector
                        # 存储原子的索引和新位置
                        expanded_atom_positions.append((atom_index, new_position))
                    else:
                        # 如果笛卡尔坐标的形状不符合预期，抛出异常
                        raise ValueError(f"Unexpected shape for cartesian_coords: {cartesian_coords.shape}")

    return expanded_atom_positions


def calculate_supercell_replications(cell_lengths, cell_angles):
    # 计算超胞在每个方向上的复制次数
    return 3, 3, 3  # 假设在每个方向上复制3次


def get_number_of_units_per_cell(Z, Z_prime):

    num_asymmetric_units = int(Z / Z_prime)
    return num_asymmetric_units  # 假设一个晶胞中有4个不对称单元


def expand_molecular_graph(mol, supercell_atoms, cell_lengths, cell_angles, Z, Z_prime):
    is_donor, is_acceptor = donor_acceptor(mol)
    original_atom_features = atom_featurizer(mol)

    num_asymmetric_units = int(Z / Z_prime)
    num_original_atoms = original_atom_features.shape[0]  # 原始分子的原子数量

    cell_atom_features = np.tile(original_atom_features, (num_asymmetric_units, 1))

    expanded_atom_features = []

    for i in range(len(supercell_atoms)):
        _, fractional_coords = supercell_atoms[i]
        atom_index = i % num_original_atoms
        atom_features = cell_atom_features[atom_index]

        extended_features = np.concatenate((atom_features, fractional_coords))
        expanded_atom_features.append(extended_features)

    expanded_atom_features = np.array(expanded_atom_features)

    # 获取原始分子的边和边特征
    original_edge_idx, original_edge_feats = get_bond_features_from_mol(mol)
    expanded_edge_idx = []
    expanded_edge_feats = []

    # 计算超胞中每个方向的复制次数
    a_times, b_times, c_times = 3, 3, 3

    # 原始单胞中不对称单元的数量
    num_units_per_cell = get_number_of_units_per_cell(Z, Z_prime)

    # 遍历超胞中所有的单元和边进行扩展
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
    # 计算总原子数量和总分子数量
    total_atoms = expanded_atom_features.shape[0]
    num_molecules = total_atoms // num_original_atoms

    # 选择最中心的分子，假设它是索引在中间的那个分子
    center_molecule_index = num_molecules // 2

    # 计算原子和边的索引范围
    start_atom_idx = center_molecule_index * num_original_atoms
    end_atom_idx = (center_molecule_index + 1) * num_original_atoms

    # 获取中心分子的原子特征
    center_atom_features = expanded_atom_features[start_atom_idx:end_atom_idx]

    # 获取中心分子的边特征
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
    return vdw_radii.get(atom_label, 2.0)  # 默认2.0Å


def get_other_molecules_features(expanded_atom_features, expanded_edge_idx, expanded_edge_feats, num_original_atoms):
    total_atoms = expanded_atom_features.shape[0]  # 超胞中的原子个数
    num_molecules = total_atoms // num_original_atoms  # 超胞中的原子个数\不对称单元的原子个数 = 超胞中不对称单元个数

    all_molecule_atom_features = []
    all_molecule_edge_idx = []
    all_molecule_edge_feats = []

    # 遍历每个不对称单元，提取其原子和边特征，排除中心分子
    for mol_idx in range(num_molecules):
        if mol_idx == num_molecules // 2:
            continue  # 跳过中心分子的一半

        start_atom_idx = mol_idx * num_original_atoms
        end_atom_idx = (mol_idx + 1) * num_original_atoms

        # 获取该分子的原子特征
        molecule_atom_features = expanded_atom_features[start_atom_idx:end_atom_idx]
        all_molecule_atom_features.append(molecule_atom_features)

        # 获取该分子的边特征
        molecule_edge_idx = []
        molecule_edge_feats = []

        # 遍历 expanded_edge_idx 中的每一条边及其索引
        for idx, (start, end) in enumerate(expanded_edge_idx):
            # 检查当前边的起点和终点索引是否在特定范围内
            # 条件是边的起点和终点索引都要在 start_atom_idx 和 end_atom_idx 之间
            if start_atom_idx <= start < end_atom_idx and start_atom_idx <= end < end_atom_idx:
                # 如果条件满足，将这条边的调整后的起点和终点索引添加到 molecule_edge_idx
                # 调整是为了将索引转换为相对于当前分子起点的相对索引
                molecule_edge_idx.append([start - start_atom_idx, end - start_atom_idx])
                # 同时，将这条边的特征信息添加到 molecule_edge_feats 列表中
                molecule_edge_feats.append(expanded_edge_feats[idx])

        all_molecule_edge_idx.append(np.array(molecule_edge_idx))
        all_molecule_edge_feats.append(np.array(molecule_edge_feats))

    # 将所有原子特征合并为一个二维数组
    all_molecule_atom_features = np.concatenate(all_molecule_atom_features, axis=0)

    # 将所有边特征合并为一个二维数组，并调整边的索引
    all_molecule_edge_feats = np.concatenate(all_molecule_edge_feats, axis=0)
    all_molecule_edge_idx = np.concatenate(all_molecule_edge_idx, axis=0)

    return all_molecule_atom_features, all_molecule_edge_idx, all_molecule_edge_feats


def extract_atom_label(attributes):

    possible_atom_type = ['H', 'B', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']

    for i, is_type in enumerate(attributes[:len(possible_atom_type)]):
        if np.any(is_type):  # 检查数组中是否存在 True 值
            return possible_atom_type[i]
    return 'Unknown'  # 如果没有找到匹配的原子类型，返回 'Unknown'


def check_vdw_contact(center_atom_features, other_atom_features):
    contacts = []  # 用于存储检测到的接触信息

    for center_feat in center_atom_features:
        # 提取中心原子的元素标签
        center_label = extract_atom_label(center_feat)

        # 获取中心原子的空间坐标（假设特征向量的最后三位为坐标）
        center_coords = center_feat[-3:]

        # 获取中心原子的范德华半径
        center_vdw_radius = calculate_vdw_radii(center_label)

        for other_feat in other_atom_features:
            # for feat in other_feat:
            other_label = extract_atom_label(other_feat)
            other_coords = np.array(other_feat[-3:], dtype=float)
            # 打印坐标和形状以调试
            # print(f"Center coords: {center_coords}, shape: {center_coords.shape}")
            # print(f"Other coords: {other_coords}, shape: {other_coords.shape}")
            # 获取其他原子的范德华半径
            other_vdw_radius = calculate_vdw_radii(other_label)
            # 计算中心原子和其他原子之间的实际距离
            try:
                distance = np.linalg.norm(center_coords - other_coords)
            except ValueError as e:
                print(f"Error calculating distance: {e}")
                continue  # 跳过这个配对，继续下一个
            # 计算范德华接触距离阈值（中心原子和其他原子的范德华半径之和）
            contact_distance = center_vdw_radius + other_vdw_radius
            # 如果实际距离小于接触距离阈值，说明两个原子之间存在范德华接触
            if distance < contact_distance:
                # 跳过CC或HH的情况
                if (center_label == 'C' and other_label == 'C') or (center_label == 'H' and other_label == 'H') or (center_label == 'C' and other_label == 'H') or (center_label == 'H' and other_label == 'C'):
                # if (center_label == 'C' and other_label == 'C') or (center_label == 'H' and other_label == 'H'):
                    continue
                contacts.append((center_label, other_label, distance, contact_distance))
    return contacts  # 返回检测到的所有接触信息


def add_contacts_to_graph(center_atom_features, center_edge_idx, center_edge_feats, other_atom_features):
    possible_atom_type = ['H', 'B', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
    contacts = check_vdw_contact(center_atom_features, other_atom_features)
    new_edge_idx = []
    new_edge_feats = []

    # 扩展中心分子原始边特征，增加一位表示接触特征
    new_center_edge_feats = []
    for edge_feats in center_edge_feats:
        expanded_feat = list(edge_feats) + [0]  # 追加0，表示没有接触
        new_center_edge_feats.append(expanded_feat)

    # 确保中心原子特征和其他原子特征是NumPy数组
    center_atom_features = np.array(center_atom_features)
    other_atom_features = np.array(other_atom_features)

    # 展平 other_atom_features 并重塑为二维数组
    new_other_atom_features = np.array([feat for other_feat in other_atom_features for feat in other_feat])
    new_other_atom_features = new_other_atom_features.reshape(-1, center_atom_features.shape[1])

    # 打印调试信息，检查 new_other_atom_features 的维度
    # print(f"New other atom features shape: {new_other_atom_features.shape}")
    if new_other_atom_features.ndim != 2:
        raise ValueError("new_other_atom_features should be a 2-dimensional array")

    # 提取原子标签
    center_labels = center_atom_features[:, :len(possible_atom_type)].argmax(axis=1)  # argmax函数返回的是最大值所在的索引位置
    other_labels = new_other_atom_features[:, :len(possible_atom_type)].argmax(axis=1)

    # print(f"Center labels: {center_labels}")
    # print(f"Other labels: {other_labels}")

    # print(f"Contacts: {contacts}")

    for center_label, other_label, distance, contact_distance in contacts:
        # 查找匹配的中心原子索引
        center_indices = np.where(center_labels == possible_atom_type.index(center_label))[0]
        # 查找匹配的其他原子索引
        other_indices = np.where(other_labels == possible_atom_type.index(other_label))[0]

        if center_indices.size == 0 or other_indices.size == 0:
            print(f"No matching atom found for labels {center_label} or {other_label}")
            continue  # 跳过没有找到匹配的情况

        center_idx = center_indices[0]
        other_idx = other_indices[0] + len(center_atom_features)

        other_idx = other_idx % len(center_atom_features)
        # 将新的边和特征添加到列表中（正向）
        new_edge_idx.append([center_idx, other_idx])
        # 新的接触特征为1.0，其余部分用0填充
        new_edge_feats.append([0] * (len(new_center_edge_feats[0]) - 1) + [1.0])

        # 将新的边和特征添加到列表中（反向）
        new_edge_idx.append([other_idx, center_idx])
        new_edge_feats.append([0] * (len(new_center_edge_feats[0]) - 1) + [1.0])

    # 将新的特征维度添加到原始边特征
    extended_edge_feats = np.array(new_center_edge_feats)

    # 确保 new_edge_idx 是一个二维数组
    new_edge_idx = np.array(new_edge_idx, dtype=np.int64)

    # 打印调试信息，检查 new_edge_idx 的维度
    # print(f"New edge idx shape: {new_edge_idx.shape}")
    if new_edge_idx.ndim != 2:
        raise ValueError("new_edge_idx should be a 2-dimensional array")

    # 合并原始边和新的接触边
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
            # 使用 CrystalReader 读取 CIF 文件
            with CrystalReader(cif_path) as reader:
                crystal = reader.crystal(refcode)

                # 获取不对称单元的数量 Z 和 Z'
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

            # print(f"Expanded atom features shape: {self.expanded_atom_features.shape}")
            # print(f"Expanded edge idx shape: {self.expanded_edge_idx.shape}")
            # print(f"Expanded edge feats shape: {self.expanded_edge_feats.shape}")

            self.center_atom_features, self.center_edge_idx, self.center_edge_feats = self.get_center_molecule_features()

            # print(f"Center atom features shape: {self.center_atom_features.shape}")
            # print(f"Center edge idx shape: {self.center_edge_idx.shape}")
            # print(f"Center edge feats shape: {self.center_edge_feats.shape}")

            self.other_molecule_atom_features, self.other_molecule_edge_idx, self.other_molecule_edge_feats = self.get_other_molecules_features()

            # 打印检查合并后的数组形状
            # print(f"Other molecule atom features shape: {self.other_molecule_atom_features.shape}")
            # print(f"Other molecule edge idx shape: {self.other_molecule_edge_idx.shape}")
            # print(f"Other molecule edge feats shape: {self.other_molecule_edge_feats.shape}")

            self.center_atom_features_without_coords = self.center_atom_features[:, :-3]
            self.self_connection_center_edge_idx, self.self_connection_center_edge_feats = self.center_molecule_self_connection_graph()

            # print(f"Self connection center edge idx shape: {self.self_connection_center_edge_idx.shape}")
            # print(f"Self connection center edge feats shape: {self.self_connection_center_edge_feats.shape}")

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






# Visualization function using RDKit
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
