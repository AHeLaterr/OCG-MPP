import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data as GeoData
from Crystal2Graph import Crystal2Graph
import os
import pandas as pd

class CIFDataset(Dataset):
    def __init__(self, cif_directory=None, melting_point_txt=None, preprocessed_data_list=None,
                 melting_point_dict=None, global_features_csv=None):
        self.cif_directory = cif_directory
        self.data_list = preprocessed_data_list if preprocessed_data_list is not None else []
        if self.data_list:  
            return
        if melting_point_dict is not None:
            self.melting_point_dict = melting_point_dict
        elif melting_point_txt:
            
            df = pd.read_csv(melting_point_txt, sep='\s+', header=None, names=['refcode', 'melting_point'])
            
            if df.shape[1] == 1:
                self.melting_point_dict = {refcode: None for refcode in df['refcode']}
            else:
                self.melting_point_dict = pd.Series(df.melting_point.values, index=df.refcode).to_dict()
        else:
            self.melting_point_dict = {}

        self.refcodes = list(self.melting_point_dict.keys())

        if global_features_csv:
            self.global_features_df = pd.read_csv(global_features_csv, index_col='Filename')
        else:
            self.global_features_df = None

    def __len__(self):
        if self.data_list:
            return len(self.data_list)
        return len(self.refcodes)

    def __getitem__(self, idx):
        if self.data_list:
            return self.data_list[idx]

        refcode = self.refcodes[idx]
        cif_path = f"{self.cif_directory}/{refcode}.cif"

        if not os.path.exists(cif_path):
            print(f"Warning: File {cif_path} not found.")
            return None

        try:
            graph_data = Crystal2Graph(cif_path, refcode, 3, 3, 3)
            x = torch.tensor(graph_data.center_atom_features_without_coords, dtype=torch.float)
            edge_index = torch.tensor(graph_data.self_connection_center_edge_idx, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(graph_data.self_connection_center_edge_feats, dtype=torch.float)

            print(f"Initial self_connection_center_edge_idx shape: {edge_index.shape}")
            print(f"Initial self_connection_center_edge_feats shape: {edge_attr.shape}")

            melting_point = self.melting_point_dict.get(refcode)
            y = torch.tensor([melting_point], dtype=torch.float) if melting_point is not None else None

            if self.global_features_df is not None:
                global_features = self.global_features_df.loc[refcode].astype(float).values
                global_feature_tensor = torch.tensor(global_features, dtype=torch.float)
                data = GeoData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, global_feature=global_feature_tensor)
            else:
                data = GeoData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

            data.refcode = refcode
            return data
        except Exception as e:
            print(f"Warning: Failed to process file {cif_path}. Error: {e}")
            return None

    @staticmethod
    def create_cif_dataset_from_preprocessed(preprocessed_data_list):
        return CIFDataset(preprocessed_data_list=preprocessed_data_list)
