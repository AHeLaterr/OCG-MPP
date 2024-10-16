# crystal.py
from ccdc.io import CrystalReader

class CrystalData:
    def __init__(self, cif_path):
        self.crystal = self.load_cif_file(cif_path)

    def load_cif_file(self, cif_path):
        crystals = CrystalReader(cif_path)
        if not crystals:
            raise ValueError("No crystals found in the CIF file.")
        return crystals[0]

    def get_space_group(self):
        return self.crystal.spacegroup_symbol

    def get_space_group_number(self):
        return self.crystal.spacegroup_number_and_setting[0]

    def get_lattice_parameters(self):
        return self.crystal.cell_lengths + self.crystal.cell_angles

    def get_Z_values(self):
        return self.crystal.z_value, self.crystal.z_prime

    def get_volume(self):
        return self.crystal.cell_volume

    def get_density(self):
        return self.crystal.calculated_density

    def get_space_group_one_hot(self):
        space_group_number = self.get_space_group_number()
        one_hot = [0] * 230
        one_hot[space_group_number - 1] = 1

        space_group_dict = {f'space_group_{i}': value for i, value in enumerate(one_hot)}
        return space_group_dict

    def get_crystal_data(self):
        return {
            # "Space Group": self.get_space_group(),
            # "Space Group Number": self.get_space_group_number(),
            # "Lattice Parameters α": self.get_lattice_parameters()[0],
            # "Lattice Parameters β": self.get_lattice_parameters()[1],
            # "Lattice Parameters γ": self.get_lattice_parameters()[2],
            # "Lattice Parameters a": self.get_lattice_parameters()[3],
            # "Lattice Parameters b": self.get_lattice_parameters()[4],
            # "Lattice Parameters c": self.get_lattice_parameters()[5],
            "Z value": self.get_Z_values()[0],
            "Z' value": self.get_Z_values()[1],
            "Volume": self.get_volume(),
            "Density": self.get_density(),
            **self.get_space_group_one_hot()
        }
