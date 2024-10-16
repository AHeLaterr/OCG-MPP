import os
import torch
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from train import cross_validate_and_train, ensemble_predictions, custom_collate_fn
from CCPGraph import CCPGraph
from CIFDataset import CIFDataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from crystal import CrystalData
from molecule import MolecularProperties
from data_saver import DataSaver
import json
from main_ import load_and_clean_data


def load_models(model_directory, device):
    model_paths = [os.path.join(model_directory, f) for f in os.listdir(model_directory) if f.endswith('.pth')]
    models = []
    for model_path in model_paths:
        model = CCPGraph()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)
    return models


def prepare_data(saved_data_path, cif_directory, global_features_csv, melting_point_txt=None):
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
                if data is not None and data.y is not None:  # 检查 data 是否有效
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

def standardize_data(data_list, label_mean, label_std, global_mean, global_std):
    def standardize_labels(data, mean, std):
        if data.y is not None:
            data.y = (data.y - mean) / std
        return data

    def normalize_global_features(data, mean, std):
        if data is not None and hasattr(data, 'global_feature'):
            data.global_feature = (data.global_feature - mean) / std
            if torch.isnan(data.global_feature).any() or torch.isinf(data.global_feature).any():
                print(f"NaN or Inf detected in global_feature after normalization: {data.global_feature}")
                data.global_feature = torch.where(
                    torch.isnan(data.global_feature) | torch.isinf(data.global_feature),
                    torch.tensor(0.0, device=data.global_feature.device),
                    data.global_feature)
        return data

    standardized_dataset = [standardize_labels(data, label_mean, label_std) for data in data_list if data is not None and data.y is not None]
    standardized_dataset = [normalize_global_features(data, global_mean, global_std) for data in standardized_dataset]
    return standardized_dataset


def predict_ensemble(models, data_loader, device, csv_file, train_mean, train_std):
    true_values, ensemble_preds, metrics = ensemble_predictions(models, data_loader, device, csv_file, train_mean, train_std)
    return true_values, ensemble_preds, metrics


def process_file(cif_path, data_saver):
    filename_without_ext = os.path.splitext(os.path.basename(cif_path))[0]
    try:
        crystal_data = CrystalData(cif_path)
        molecular_properties = MolecularProperties(cif_path)
        crystal_info = crystal_data.get_crystal_data()
        molecular_info = molecular_properties.get_molecular_properties()
        combined_data = {"Filename": filename_without_ext, **crystal_info, **molecular_info}

        if molecular_properties.has_warnings():
            warnings = molecular_properties.get_warnings()
            data_saver.log_warning(filename_without_ext, warnings)

        data_saver.add_data(combined_data)
    except Exception as e:
        error_message = str(e)
        data_saver.log_failure(filename_without_ext, error_message)
        print(f"Error in processing {cif_path}: {error_message}")


def process_cif_files(directory, output_csv, failed_txt):
    data_saver = DataSaver(output_csv, failed_txt)
    processed_files = set(data_saver.df['Filename'])
    all_files = [f for f in os.listdir(directory) if f.endswith(".cif")]
    files_to_process = [f for f in all_files if os.path.splitext(f)[0] not in processed_files]

    print(f"Already processed: {len(processed_files)} files")
    print(f"Files to process: {len(files_to_process)} files")
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_cif = {executor.submit(process_file, os.path.join(directory, cif_path), data_saver): cif_path for cif_path in files_to_process}
        progress = tqdm(as_completed(future_to_cif), total=len(files_to_process), desc="Processing CIF files")
        for future in progress:
            try:
                future.result()
            except Exception as e:
                print(f"Error processing file: {e}")


def load_standardization_params(meta_file):
    with open(meta_file, 'r') as f:
        params = json.load(f)
    label_mean = params['label_mean']
    label_std = params['label_std']
    global_mean = torch.tensor(params['global_mean'])
    global_std = torch.tensor(params['global_std'])
    return label_mean, label_std, global_mean, global_std


def main():
    # 路径配置
    saved_data_path = 'D:\\solo-mt\\core\\绝对正确.pt'
    model_directory = 'D:\\solo-mt\\3.分子图+全局特征+交叉验证\\model'  # 模型存储目录
    cif_directory = 'D:\\solo-mt\\core\\正确txt'
    melting_point_txt = 'D:\\solo-mt\\core\\正确txt_MT.txt'
    global_features_csv = 'D:\\solo-mt\\core\\global_正确txt_features.csv'
    failed_txt = 'D:\\solo-mt\\core\\global_features_failed.txt'
    meta_file = 'D:\\solo-mt\\core\\standardization_params.json'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    process_cif_files(cif_directory, global_features_csv, failed_txt)
    models = load_models(model_directory, device)
    data_list = prepare_data(saved_data_path, cif_directory, global_features_csv, melting_point_txt)
    label_mean, label_std, global_mean, global_std = load_standardization_params(meta_file)
    standardized_data = standardize_data(data_list, label_mean, label_std, global_mean, global_std)
    data_loader = DataLoader(standardized_data, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)


    predict_ensemble(models, data_loader, device, './snapshot_Bayes_Opt/test/test_predictions.csv', label_mean, label_std)

    # for i, data in enumerate(standardized_data):
    #     if data.y is not None:
    #         true_value = filtered_true_values[i].item() if i < len(filtered_true_values) else "N/A"
    #     else:
    #         true_value = "N/A"
    #     pred_value = filtered_predictions[i].item() if i < len(filtered_predictions) else "N/A"
    #     print(f"Refcode: {data.refcode}, True Melting Point: {true_value}, Predicted Melting Point: {pred_value}")

if __name__ == "__main__":
    main()
