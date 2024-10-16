import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data, DataLoader, Batch
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr
from torch.utils.data import Subset
import os
import time
from sklearn.model_selection import KFold
import csv
from typing import Union, Dict, List, Tuple
from torch.optim import Optimizer


def verify_dir_exists(dirname: str) -> None:
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def create_file(file_path: str, content: str) -> None:
    verify_dir_exists(os.path.dirname(file_path))
    with open(file_path, 'w') as file:
        file.write(content)
    print(f"File created and content written to {file_path}")


def check_data_integrity(data: Data) -> None:
    if data.edge_index.max() >= data.x.size(0) or data.edge_index.min() < 0:
        raise ValueError("Invalid edge index detected")
    if torch.isnan(data.x).any() or torch.isinf(data.x).any():
        raise ValueError("NaN or Inf detected in node features")
    if torch.isnan(data.edge_attr).any() or torch.isinf(data.edge_attr).any():
        raise ValueError("NaN or Inf detected in edge attributes")


'''------------------------------------------------------------------------------------------------------------------'''


def train(model: torch.nn.Module,
          train_loader: DataLoader,
          device: Union[torch.device, str],
          optimizer: Optimizer,
          accumulation_steps: int = 4) -> float:

    model.train()
    loss_all = 0
    optimizer.zero_grad()  # 仅在每次累积开始时清零梯度

    for i, data in enumerate(train_loader):
        try:
            check_data_integrity(data)  # 一个batch

            data = data.to(device)
            output, att = model(data)

            if output.shape != data.y.shape:
                print(f"Skipping batch {i} due to mismatched sizes: output shape {output.shape}, target shape {data.y.shape}")
                continue

            loss = F.mse_loss(output, data.y) / accumulation_steps
            loss_all += loss.item() * data.num_graphs
            loss.backward()

            if (i + 1) % accumulation_steps == 0:  # 梯度累积和梯度裁剪操作，以便控制梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()  # 在每次参数更新后清零梯度
                torch.cuda.empty_cache()

        except (ValueError, AttributeError, RuntimeError) as e:
            print(f"Skipping batch {i} due to an error: {e}")
            continue

    return loss_all / len(train_loader.dataset)


def test(model: torch.nn.Module,
         loader: DataLoader,
         device: Union[torch.device, str],
         mean: torch.Tensor,
         std: torch.Tensor) -> Tuple[float, float, List[float], List[float], List[float], List[Tuple[str, float, float]]]:
    '''返回了模型在测试集上的  1.平均损失、2.平均绝对误差、3.预测值、4.真实值、5.注意力权重列表 6.详细结果'''
    model.eval()
    error = 0
    loss_all = 0

    model_output = []
    y = []
    att_list = []
    results = []   # 标签 真实值 预测值 详细结果

    with torch.no_grad():
        for i, data in enumerate(loader):
            try:
                check_data_integrity(data)
                data = data.to(device)
                output, att = model(data)

                output_original = output * std + mean  # 预测值
                data_y_original = data.y * std + mean  # 原始值

                if output_original.shape[0] != data_y_original.shape[0]:
                    print(f"Skipping batch {i} due to size mismatch: output shape {output_original.shape}, target shape {data_y_original.shape}")
                    continue

                error += (output_original - data_y_original).abs().sum().item()  # 平均绝对误差
                loss = F.mse_loss(output_original, data_y_original)  # 计算损失
                loss_all += loss.item() * data.num_graphs   # 总损失

                model_output.extend(output_original.tolist())  # 预测数据
                y.extend(data_y_original.tolist())  # 真实数据
                att_list.extend(att.tolist())  # 注意力得分

                # 标签 真实值 预测值
                refcodes = data.refcode
                true_values = data_y_original.tolist()
                predicted_values = output_original.tolist()
                for refcode, true_val, pred_val in zip(refcodes, true_values, predicted_values):
                    results.append((refcode, true_val, pred_val))

                del data, output, loss  # 清除不再需要的变量，释放内存
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"An error occurred during testing: {e}")
                continue

    return loss_all / len(loader.dataset), error / len(loader.dataset), model_output, y, att_list, results


'''------------------------------------------------------------------------------------------------------------------'''


def save_results(results: List[Tuple[str, float, float]], foldername: str, epoch: int) -> None:
    verify_dir_exists(foldername)
    filename = os.path.join(foldername, f'results_epoch_{epoch}.txt')
    with open(filename, 'w') as f:
        f.write("Refcode\tTrue Value\tPredicted Value\n")
        for refcode, true_val, pred_val in results:
            f.write(f"{refcode}\t{true_val}\t{pred_val}\n")


def compute_metrics(y_true: List[float], y_pred: List[float]) -> Tuple[float, float, float, float, float]:
    if len(y_true) < 2 or len(y_pred) < 2:
        return 0, 0, 0, 0, 0
    pccs = pearsonr(y_true, y_pred)[0]
    r2_pccs = pccs ** 2
    r2 = r2_score(y_true, y_pred)
    mse = np.mean((np.array(y_true) - np.array(y_pred)) ** 2)
    rmse = np.sqrt(mse)
    mae = sum(abs(y - y_pred) for y, y_pred in zip(y_true, y_pred)) / len(y_true)
    return mae, pccs, r2_pccs, r2, rmse


def update_history(history, train_metrics, valid_metrics, train_output, y_train, valid_output, y_valid):
    '''更新history指标记录字典'''
    train_loss, train_mae, train_pccs, train_r2_pccs, train_r2, train_rmse = train_metrics
    valid_loss, valid_mae, valid_pccs, valid_r2_pccs, valid_r2, valid_rmse = valid_metrics

    history['Train Loss'].append(train_loss)
    history['Train Mae'].append(train_mae)
    history['Train Rpcc'].append(train_pccs)
    history['Train Rpcc2'].append(train_r2_pccs)
    history['Train R2'].append(train_r2)
    history['Train RMSE'].append(train_rmse)
    # history['Train Data'].append({'y_train': y_train, 'train_output': train_output})

    history['Valid Loss'].append(valid_loss)
    history['Valid Mae'].append(valid_mae)
    history['Valid Rpcc'].append(valid_pccs)
    history['Valid Rpcc2'].append(valid_r2_pccs)
    history['Valid R2'].append(valid_r2)
    history['Valid RMSE'].append(valid_rmse)
    # history['Valid Data'].append({'y_valid': y_valid, 'valid_output': valid_output})


def save_metrics_to_csv(
    history: Dict[str, List[float]],
    foldername: str,
    filename: str = 'training_metrics.csv'
) -> None:

    verify_dir_exists(foldername)
    file_path = os.path.join(foldername, filename)
    fieldnames = [
        'Epoch', 'Train Loss', 'Train Mae', 'Train Rpcc', 'Train Rpcc2', 'Train R2', 'Train RMSE',
        'Valid Loss', 'Valid Mae', 'Valid Rpcc', 'Valid Rpcc2', 'Valid R2', 'Valid RMSE'
    ]

    with open(file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)  # 字典格式的数据写入 CSV 文件
        writer.writeheader()

        for epoch in range(len(history['Train Loss'])):
            writer.writerow({
                'Epoch': epoch + 1,
                'Train Loss': history['Train Loss'][epoch],
                'Train Mae': history['Train Mae'][epoch],
                'Train Rpcc': history['Train Rpcc'][epoch],
                'Train Rpcc2': history['Train Rpcc2'][epoch],
                'Train R2': history['Train R2'][epoch],
                'Train RMSE': history['Train RMSE'][epoch],
                'Valid Loss': history['Valid Loss'][epoch],
                'Valid Mae': history['Valid Mae'][epoch],
                'Valid Rpcc': history['Valid Rpcc'][epoch],
                'Valid Rpcc2': history['Valid Rpcc2'][epoch],
                'Valid R2': history['Valid R2'][epoch],
                'Valid RMSE': history['Valid RMSE'][epoch]
            })

    print(f"Metrics saved to {file_path}")


def save_model_parameters(snapshot_path: str, model: torch.nn.Module) -> None:
    file_path = os.path.join(snapshot_path, 'model.pth')
    verify_dir_exists(snapshot_path)
    print(f"Attempting to save model to {file_path}")
    print(f"Directory exists: {os.path.exists(snapshot_path)}")
    print(f"Has write permission: {os.access(snapshot_path, os.W_OK)}")
    try:
        torch.save(model.state_dict(), file_path)
        print(f"Model parameters saved to {file_path}")
    except Exception as e:
        print(f"Failed to save model parameters to {file_path}: {e}")


def update_reports(reports: Dict[str, Union[float, int]], valid_metrics, model: torch.nn.Module, snapshot_path: str) -> Dict[str, Union[float, int]]:  # 更新报告的最好指标，并保存

    valid_loss, valid_mae, valid_pccs, valid_r2_pccs, valid_r2, valid_rmse = valid_metrics
    if valid_mae < reports['valid mae']:
        reports.update({
            'valid mae': valid_mae,
            'valid loss': valid_loss,
            'valid PCCS': valid_pccs,
            'valid R2 PCCS': valid_r2_pccs,
            'valid R2': valid_r2,
            'valid RMSE': valid_rmse
        })
        # Save the model parameters
        save_model_parameters(snapshot_path, model)

    return reports


def print_epoch_results(epoch: int, lr: float, train_metrics, valid_metrics, elapsed_time: float) -> None:

    train_loss, train_mae, train_pccs, train_r2_pccs, train_r2, train_rmse = train_metrics
    valid_loss, valid_mae, valid_pccs, valid_r2_pccs, valid_r2, valid_rmse = valid_metrics

    print(
        f'Epoch: {epoch:03d}, ==> LR: {lr:.7f}, Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, '
        f'Train R² (PCCS): {train_r2_pccs:.4f}, Train R²: {train_r2:.4f}, Train RMSE: {train_rmse:.4f}, '
        f'Valid Loss: {valid_loss:.4f}, Valid MAE: {valid_mae:.4f}, Valid R² (PCCS): {valid_r2_pccs:.4f}, '
        f'Valid R²: {valid_r2:.4f}, Valid RMSE: {valid_rmse:.4f}, Elapsed Time: {elapsed_time:.2f} s'
    )


def debug_data_loader(data_loader):   # 跳过节点形状异常的批次
    for batch_idx, data in enumerate(data_loader):
        try:
            print(f"Batch {batch_idx}: Data shape: {data.x.shape}, Edges shape: {data.edge_index.shape}")
            if data.edge_index.max() >= data.x.shape[0] or data.edge_index.min() < 0:
                raise ValueError(
                    f"Invalid edge index in batch {batch_idx}: max index {data.edge_index.max()}, data shape {data.x.shape}")
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue


def training(Model, data_loaders, n_epoch=100, snapshot_path='./snapshot/', save_att=False, optimizer=None,
             train_mean=None, train_std=None, val_mean=None, val_std=None, reports=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model.to(device)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00002, weight_decay=1e-5)  # L2 正则
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=30, min_lr=0.0000004)

    metrics = ['Loss', 'Mae', 'Rpcc', 'Rpcc2', 'R2', 'RMSE'] # 创建空的history字典
    history = {f'Train {metric}': [] for metric in metrics}
    history.update({f'Valid {metric}': [] for metric in metrics})

    if reports is None:
        reports = {
            'valid mae': float('inf'),
            'valid loss': float('inf'),
            'valid PCCS': 0.0,
            'valid R2 PCCS': 0.0,
            'valid R2': 0.0,
            'valid RMSE': float('inf')
        }

    debug_data_loader(data_loaders['train'])
    debug_data_loader(data_loaders['valid'])

    for epoch in range(1, n_epoch + 1):
        start_time_1 = time.time()
        lr = scheduler.optimizer.param_groups[0]['lr']

        train_loss = train(model, data_loaders['train'], device, optimizer)
        train_loss, train_mae, train_output, y_train, valid_att, train_results = test(model, data_loaders['train'], device, mean=train_mean, std=train_std)
        save_results(train_results, os.path.join(snapshot_path, 'train'), epoch)  # 训练集的一个批次的真实值与预测值
        valid_loss, valid_mae, valid_output, y_valid, valid_att, valid_results = test(model, data_loaders['valid'], device, mean=val_mean, std=val_std)
        save_results(valid_results, os.path.join(snapshot_path, 'valid'), epoch)  # 验证集集的一个批次的真实值与预测值

        train_metrics = (train_loss, *compute_metrics(y_train, train_output))
        valid_metrics = (valid_loss, *compute_metrics(y_valid, valid_output))

        scheduler.step(valid_loss)  # 为了更新lr

        update_history(history, train_metrics, valid_metrics, train_output, y_train, valid_output, y_valid)  # 更新指标
        reports = update_reports(reports, valid_metrics, model, snapshot_path)  # 判断是否为最好的模型，更新最好的指标

        end_time_1 = time.time()
        elapsed_time = end_time_1 - start_time_1
        print_epoch_results(epoch, lr, train_metrics, valid_metrics, elapsed_time)  # 打印当前epoch各项指标

    save_metrics_to_csv(history, snapshot_path)  # 所有指标传csv

    print(f'\nLoss: {reports["valid loss"]}\tMAE: {reports["valid mae"]}\tvalid_r2: {reports["valid R2"]}')
    return reports


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


def initialize_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def cross_validate_and_train(Model, train_dataset, k_folds=5, n_epoch=100, snapshot_path='./snapshot/',
                             save_att=False, train_mean=None, train_std=None, val_mean=None, val_std=None):

    os.makedirs(snapshot_path, exist_ok=True)
    kfold = KFold(n_splits=k_folds, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_fold_metrics = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
        print(f'Fold {fold + 1}/{k_folds}')

        fold_snapshot_path = os.path.join(snapshot_path, f'fold_{fold + 1}')
        os.makedirs(fold_snapshot_path, exist_ok=True)

        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn, drop_last=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn, drop_last=True)
        data_loaders = {'train': train_loader, 'valid': val_loader}

        model = Model().to(device)  # Instantiate the model
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00002, weight_decay=1e-5)

        debug_data_loader(data_loaders['train'])
        debug_data_loader(data_loaders['valid'])

        reports = {
            'valid mae': float('inf'),
            'valid loss': float('inf'),
            'valid PCCS': 0.0,
            'valid R2 PCCS': 0.0,
            'valid R2': 0.0,
            'valid RMSE': float('inf')
        }

        reports = training(
            model,
            data_loaders,
            n_epoch=n_epoch,
            save_att=save_att,
            snapshot_path=fold_snapshot_path,
            train_mean=train_mean,
            train_std=train_std,
            val_mean=val_mean,
            val_std=val_std,
            reports=reports,
            optimizer=optimizer  # Pass the optimizer to the training function
        )

        all_fold_metrics.append(reports)
        models.append(model)

    return models, all_fold_metrics

def ensemble_predictions(models, data_loader, device, csv_file, mean, std):
    verify_dir_exists(os.path.dirname(csv_file))
    all_predictions = []
    models = [model.to(device) for model in models]

    for model in models:
        model.eval()
        predictions = []

        with torch.no_grad():
            for batch_idx, data in enumerate(data_loader):
                try:
                    check_data_integrity(data)  # 确保数据完整性
                    data = data.to(device)
                    output, _ = model(data)
                    predictions.append(output.to(device))  # 确保输出在正确的设备上
                except Exception as e:
                    print(f"Skipping batch {batch_idx} due to error: {e}")
                    continue

        if predictions:
            all_predictions.append(torch.cat(predictions))  # 确保predictions非空再进行拼接
        else:
            print(f"No valid predictions for model {model}")

    if not all_predictions:
        raise RuntimeError("No valid predictions found for any model.")

    all_predictions = torch.stack(all_predictions)  # 堆叠每个模型的预测值

    ensemble_preds = torch.mean(all_predictions, dim=0)
    ensemble_preds = ensemble_preds * std + mean  # 恢复到原始数据尺度

    true_values = []
    refcodes = []
    for data in data_loader:
        if data.y is not None:
            true_values.append(data.y.to(device))  # 确保真实值在正确的设备上
        else:
            true_values.append(torch.tensor([float('nan')], device=device))  # 占位符
        refcodes.extend(data.refcode)

    true_values = torch.cat(true_values)
    true_values = true_values * std + mean

    ensemble_preds = ensemble_preds.cpu()
    true_values = true_values.cpu()

    # 过滤true_values中的NaN值，然后计算度量
    valid_indices = ~torch.isnan(true_values)
    valid_true_values = true_values[valid_indices]
    valid_ensemble_preds = ensemble_preds[valid_indices]

    if len(valid_true_values) > 0:
        pccs, r2_pccs, r2, rmse, mae = compute_metrics(valid_true_values.numpy(), valid_ensemble_preds.numpy())
    else:
        pccs, r2_pccs, r2, rmse, mae = float('nan'), float('nan'), float('nan'), float('nan'), float('nan')

    metrics = {
        "pccs": pccs,
        "r2_pccs": r2_pccs,
        "r2": r2,
        "rmse": rmse,
        "mae": mae
    }

    # 写入CSV文件
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['Refcode', 'True Value'] + [f'Model {i + 1} Prediction' for i in range(len(models))] + [
            'Ensemble Prediction']
        writer.writerow(header)

        for i in range(len(true_values)):
            row = [refcodes[i], true_values[i].item()] + [all_predictions[j][i].item() for j in range(len(models))] + [
                ensemble_preds[i].item()]
            writer.writerow(row)

    return true_values, ensemble_preds, metrics
