import os
import numpy as np
import random
from tqdm import tqdm
from typing import List, Any, Tuple, Optional
import json

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

import torch
from torch.utils.data import Subset
from torch_geometric.data import DataLoader, Batch

from train import cross_validate_and_train, ensemble_predictions, initialize_weights
from CCPGraph import CCPGraph
from CIFDataset import CIFDataset



os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def check_gpu_memory():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")

# 固定随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def custom_collate_fn(batch):
    filtered_batch = [item for item in batch if item is not None]

    for item in filtered_batch:
        if item.edge_index.dim() < 2 or item.edge_index.size(1) == 0:
            return None
        if item.edge_attr is not None and item.edge_attr.size(1) != 6:
            return None

    for item in filtered_batch:
        if item.edge_attr is None:
            item.edge_attr = torch.zeros((item.edge_index.size(1), 6), dtype=torch.float)

    if len(filtered_batch) == 0:
        return None

    return Batch.from_data_list(filtered_batch)

def load_and_clean_data(path):
    if os.path.exists(path):
        print("Loading pre-processed data from file...")
        data_list = torch.load(path)
        cleaned_data_list = []
        for data in data_list:
            if not torch.isnan(data.y).any() and not torch.isinf(data.y).any():
                cleaned_data_list.append(data)
            else:
                print(f"Removed data with NaN or inf in targets: {data}")
        print(f"Loaded {len(cleaned_data_list)} clean items from {len(data_list)} original items.")
        return cleaned_data_list
    else:
        raise FileNotFoundError("The specified path does not exist.")

def load_and_preprocess_data(saved_data_path, cif_directory, melting_point_txt, global_features_csv=None):
    if os.path.exists(saved_data_path):
        print("Loading pre-processed data from file...")
        cleaned_dataset = load_and_clean_data(saved_data_path)
        dataset = CIFDataset.create_cif_dataset_from_preprocessed(cleaned_dataset)
    else:
        print("Converting CIF files to graphs and saving to file...")
        dataset = CIFDataset(cif_directory, melting_point_txt, global_features_csv=global_features_csv)

        data_list = []
        with tqdm(total=len(dataset), desc="Converting CIFs to Graphs") as pbar:
            for idx in range(len(dataset)):
                data = dataset[idx]
                if data is not None:
                    data_list.append(data)
                else:
                    print(f"Warning: Failed to process file {dataset.cif_directory}/{dataset.refcodes[idx]}.cif.")
                pbar.update(1)

        if data_list:
            torch.save(data_list, saved_data_path)
            print(f"Data saved to {saved_data_path}")
        else:
            print("No data processed or saved due to errors or empty dataset.")

    return dataset

def remove_entries_by_refcode(data_list: List[Any], refcodes_to_remove: List[str]) -> List[Any]:
    subset_data = [data for data in data_list if data.refcode not in refcodes_to_remove]
    return subset_data

def read_refcodes_from_txt(file_path: str) -> List[str]:
    refcodes = []
    with open(file_path, 'r') as file:
        for line in file:
            refcode = line.strip(' ').split()[0]
            refcodes.append(refcode)
    return refcodes

def select_subset_by_refcode(data_list: List[Any], refcode_list: List[str]) -> List[Any]:
    subset_data = [data for data in data_list if data.refcode in refcode_list]
    return subset_data

def standardize_labels(data, mean, std):
    if torch.isnan(data.y).any() or torch.isinf(data.y).any():
        print(f"Skipping normalization due to NaN or Inf in labels: {data.refcode}")
        return data
    data.y = (data.y - mean) / std
    return data

def compute_global_feature_statistics(dataset):
    global_features = [data.global_feature for data in dataset if hasattr(data, 'global_feature') and data.global_feature is not None]
    if not global_features:
        return None, None
    global_features_tensor = torch.stack(global_features)
    mean = global_features_tensor.mean(dim=0)
    std = global_features_tensor.std(dim=0)
    return mean, std

def normalize_global_features(dataset, mean, std):
    if mean is None or std is None:
        return dataset
    for data in dataset:
        if hasattr(data, 'global_feature') and data.global_feature is not None:
            data.global_feature = (data.global_feature - mean) / std
            if torch.isnan(data.global_feature).any() or torch.isinf(data.global_feature).any():
                print(f"NaN or Inf detected in global_feature after normalization: {data.global_feature}")
                data.global_feature = torch.where(torch.isnan(data.global_feature) | torch.isinf(data.global_feature),
                                                  torch.tensor(0.0, device=data.global_feature.device),
                                                  data.global_feature)
    return dataset

def filter_and_standardize(dataset: List[Any]) -> Tuple[List[Any], float, float, np.ndarray, Optional[torch.Tensor], Optional[torch.Tensor]]:
    filtered_dataset = [data for data in dataset if data is not None]
    all_targets = [data.y for data in filtered_dataset]
    all_targets_np = torch.stack(all_targets).numpy()

    label_mean = np.mean(all_targets_np)
    label_std = np.std(all_targets_np)

    standardized_dataset = [standardize_labels(data, label_mean, label_std) for data in filtered_dataset]
    global_mean, global_std = compute_global_feature_statistics(filtered_dataset)
    standardized_dataset = normalize_global_features(standardized_dataset, global_mean, global_std)

    return standardized_dataset, label_mean, label_std, all_targets_np, global_mean, global_std

def select_and_standardize_data(
        refcodes_file_path: str,
        dataset: List[Any],
        refcodes_to_remove: Optional[List[str]] = None
) -> Tuple[List[Any], float, float, np.ndarray, Optional[torch.Tensor], Optional[torch.Tensor]]:

    selected_refcodes = read_refcodes_from_txt(refcodes_file_path)
    subset_data = select_subset_by_refcode(dataset, selected_refcodes)
    print(f"Number of selected data points: {len(subset_data)}")

    if refcodes_to_remove is not None:
        subset_data = remove_entries_by_refcode(subset_data, refcodes_to_remove)
        print(f"Number of data points after removal: {len(subset_data)}")

    standardized_dataset, label_mean, label_std, all_targets_np, global_mean, global_std = filter_and_standardize(
        subset_data)
    print(label_mean, label_std)

    return standardized_dataset, label_mean, label_std, all_targets_np, global_mean, global_std

def create_data_loaders(train_dataset, val_dataset, batch_size=8, collate_fn=None):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True)
    return train_loader, val_loader

def calculate_statistics(targets_np):
    mean = np.mean(targets_np)
    std = np.std(targets_np)
    return mean, std

def split_and_prepare_data_loaders(standardized_dataset, all_targets_np, batch_size=32):
    dataset_indices = list(range(len(standardized_dataset)))
    train_indices, test_indices = train_test_split(dataset_indices, test_size=0.2, random_state=42)

    train_dataset = Subset(standardized_dataset, train_indices)
    test_dataset = Subset(standardized_dataset, test_indices)

    train_targets = all_targets_np[train_indices]
    test_targets = all_targets_np[test_indices]

    train_mean, train_std = calculate_statistics(train_targets)
    test_mean, test_std = calculate_statistics(test_targets)
    print(train_mean, train_std, '\n', test_mean, test_std)

    train_loader, test_loader = create_data_loaders(train_dataset, test_dataset, batch_size=batch_size, collate_fn=custom_collate_fn)

    return train_loader, test_loader, train_mean, train_std, test_mean, test_std

def main():
    saved_data_path = 'D:\\solo-mt\\core\\molgraph_6.7w_with_global_feature.pt'
    cif_directory = 'D:\\solo-mt\\core\\14.cif'
    melting_point_txt = 'D:\\solo-mt\\core\\21.MT.txt'
    refcodes_file_path = 'D:\\solo-mt\\core\\21.MT.txt'
    # refcodes_file_path = 'D:\\solo-mt\\core\\SELECT.txt'
    global_features_csv = 'D:\\solo-mt\\core\\first_global_feature_6.7w.csv'

    dataset = load_and_preprocess_data(saved_data_path, cif_directory, melting_point_txt,global_features_csv)

    standardized_dataset, label_mean, label_std, all_targets_np, global_mean, global_std = (
        select_and_standardize_data(refcodes_file_path, dataset))

    train_loader, val_loader, train_mean, train_std, val_mean, val_std = split_and_prepare_data_loaders(
        standardized_dataset, all_targets_np, batch_size=64)

    standardization_params = {
        'label_mean': float(label_mean),
        'label_std': float(label_std),
        'global_mean': global_mean.tolist() if global_mean is not None else None,
        'global_std': global_std.tolist() if global_std is not None else None
    }

    with open('standardization_params.json', 'w') as f:
        json.dump(standardization_params, f)

    models, all_fold_metrics = cross_validate_and_train(
        lambda: CCPGraph(use_global_features=False),
        Subset(standardized_dataset, list(range(len(train_loader.dataset)))),
        k_folds=5,
        n_epoch=200,
        snapshot_path='./snapshot_Bayes_Opt/test/',
        save_att=True,
        train_mean=train_mean,
        train_std=train_std,
        val_mean=val_mean,
        val_std=val_std,
    )

    check_gpu_memory()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_targets_tensor, ensemble_preds_val, val_metrics = ensemble_predictions(models, val_loader, device,
                                                                               './snapshot_Bayes_Opt/test/val_predictions.csv',
                                                                               train_mean, train_std)
    ensemble_preds_val = ensemble_preds_val.to(device)

    val_targets_tensor = val_targets_tensor.to(device)

    mae_val = torch.mean(torch.abs(ensemble_preds_val - val_targets_tensor))
    rmse_val = torch.sqrt(torch.mean((ensemble_preds_val - val_targets_tensor) ** 2))
    r2_val = r2_score(val_targets_tensor.cpu(), ensemble_preds_val.cpu())
    pccs_val = pearsonr(val_targets_tensor.cpu(), ensemble_preds_val.cpu())[0]
    r2_pccs_val = pccs_val ** 2

    print(
        f'Validation Ensemble MAE: {mae_val.item()}, Validation Ensemble RMSE: {rmse_val.item()}, Validation Ensemble R²: {r2_val}, Validation Ensemble R² (PCCS): {r2_pccs_val}')

    check_gpu_memory()

    val_results = []
    val_refcodes = [data.refcode for data in list(val_loader.dataset)[:ensemble_preds_val.size(0)]]
    val_targets_original = val_targets_tensor
    ensemble_preds_original = ensemble_preds_val

    for refcode, true_val, pred_val in zip(val_refcodes, val_targets_original.tolist(),
                                           ensemble_preds_original.tolist()):
        val_results.append((refcode, true_val, pred_val))


if __name__ == "__main__":
    main()
