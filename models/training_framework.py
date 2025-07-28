# AI模型训练框架
# 提供统一的训练接口、监控和管理功能

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import json
import os
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime, timedelta
import traceback

# TensorBoard支持
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# Weights & Biases支持
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from utils.logger import get_logger
from utils.database import DatabaseManager
from models.data_processor import HealthDataPipeline
from config.settings import Config

logger = get_logger(__name__)

class HealthDataset(Dataset):
    """健康数据PyTorch数据集类"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray = None, task_type: str = 'classification'):
        """
        初始化数据集
        
        Args:
            features: 特征数据
            targets: 目标数据（可选，用于预测时为None）
            task_type: 任务类型，'classification' 或 'regression'
        """
        self.features = torch.FloatTensor(features)
        if targets is not None:
            if task_type == 'classification':
                self.targets = torch.LongTensor(targets)
            else:
                self.targets = torch.FloatTensor(targets)
        else:
            self.targets = None
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        return self.features[idx]

class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

class ModelTrainer:
    """AI模型训练器"""
    
    def __init__(self, model_name: str = None, config: Dict = None):
        """
        初始化训练器
        
        Args:
            model_name: 模型名称
            config: 训练配置
        """
        self.model_name = model_name or f"health_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.config = config or self._get_default_config()
        
        # 组件初始化
        self.logger = logger
        self.db_manager = DatabaseManager()
        self.data_pipeline = HealthDataPipeline()
        
        # 训练状态
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.training_history = []
        self.best_metrics = {}
        
        # 实验跟踪
        self.experiment_tracking = self.config.get('experiment_tracking', {})
        self.tensorboard_writer = None
        self.wandb_run = None
        
        # 设备设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"使用设备: {self.device}")
        
        # 初始化实验跟踪
        self._init_experiment_tracking()
    
    def _get_default_config(self) -> Dict:
        """获取默认训练配置"""
        return {
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 100,
            'validation_split': 0.2,
            'early_stopping': {
                'patience': 15,
                'min_delta': 0.001,
                'restore_best_weights': True
            },
            'optimizer': 'adam',
            'scheduler': {
                'type': 'reduce_on_plateau',
                'factor': 0.5,
                'patience': 10,
                'min_lr': 1e-6
            },
            'experiment_tracking': {
                'use_tensorboard': True,
                'use_wandb': False,
                'log_dir': 'logs/experiments'
            },
            'model_checkpoints': {
                'save_dir': 'models/checkpoints',
                'save_best_only': True,
                'save_frequency': 10
            }
        }
    
    def _init_experiment_tracking(self):
        """初始化实验跟踪工具"""
        try:
            # TensorBoard
            if self.experiment_tracking.get('use_tensorboard', False) and TENSORBOARD_AVAILABLE:
                log_dir = os.path.join(
                    self.experiment_tracking.get('log_dir', 'logs/experiments'),
                    self.model_name
                )
                os.makedirs(log_dir, exist_ok=True)
                self.tensorboard_writer = SummaryWriter(log_dir)
                self.logger.info(f"TensorBoard日志目录: {log_dir}")
            
            # Weights & Biases
            if self.experiment_tracking.get('use_wandb', False) and WANDB_AVAILABLE:
                self.wandb_run = wandb.init(
                    project="shec-ai-health-prediction",
                    name=self.model_name,
                    config=self.config
                )
                self.logger.info("已初始化Weights & Biases跟踪")
                
        except Exception as e:
            self.logger.warning(f"实验跟踪初始化失败: {str(e)}")
    
    def prepare_data(self, query: str = None, patient_ids: List[int] = None,
                    features: List[str] = None, target_column: str = None) -> Tuple[DataLoader, DataLoader, Dict]:
        """
        准备训练数据
        
        Args:
            query: 自定义SQL查询
            patient_ids: 指定患者ID列表
            features: 特征列名列表
            target_column: 目标列名
            
        Returns:
            训练数据加载器, 验证数据加载器, 数据信息字典
        """
        try:
            # 获取原始数据
            if query:
                raw_data = self.db_manager.execute_query(query)
            else:
                # 默认查询：获取健康指标数据
                default_query = """
                SELECT 
                    hm.*
                    -- 计算目标变量：基于多个指标的健康风险评分
                    CASE 
                        WHEN (hm.systolic_pressure > 140 OR hm.diastolic_pressure > 90) THEN 2
                        WHEN (hm.systolic_pressure > 130 OR hm.diastolic_pressure > 80) THEN 1
                        ELSE 0
                    END as health_risk_level
                FROM patient_health_metrics hm
                WHERE hm.systolic_pressure IS NOT NULL 
                  AND hm.diastolic_pressure IS NOT NULL
                """
                
                if patient_ids:
                    placeholder = ', '.join(['%s'] * len(patient_ids))
                    default_query += f" AND hm.patient_id IN ({placeholder})"
                    raw_data = self.db_manager.execute_query(default_query, patient_ids)
                else:
                    raw_data += " ORDER BY hm.measurement_time DESC LIMIT 5000"
                    raw_data = self.db_manager.execute_query(default_query)
            
            if not raw_data:
                raise ValueError("未获取到训练数据")
            
            # 转换为DataFrame
            df = pd.DataFrame(raw_data)
            self.logger.info(f"获取到 {len(df)} 条原始数据")
            
            # 数据预处理
            processed_data = self.data_pipeline.prepare_training_data(raw_data)
            
            # 准备特征和目标
            if features:
                feature_data = processed_data['features'][features]
            else:
                # 默认特征选择
                feature_columns = [
                    'age', 'systolic_pressure', 'diastolic_pressure', 
                    'blood_sugar', 'bmi'
                ]
                available_features = [col for col in feature_columns if col in processed_data['features'].columns]
                feature_data = processed_data['features'][available_features]
            
            # 目标变量
            if target_column and target_column in processed_data['features'].columns:
                target_data = processed_data['features'][target_column].values
            else:
                # 默认使用健康风险等级
                target_data = df['health_risk_level'].values if 'health_risk_level' in df.columns else None
            
            if target_data is None:
                raise ValueError("未找到目标变量")
            
            # 数据标准化
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(feature_data)
            
            # 标签编码（如果需要）
            if target_data.dtype == 'object':
                self.label_encoder = LabelEncoder()
                y_encoded = self.label_encoder.fit_transform(target_data)
            else:
                y_encoded = target_data.astype(np.float32)
            
            # 数据集划分
            dataset = HealthDataset(X_scaled, y_encoded)
            
            val_size = int(len(dataset) * self.config['validation_split'])
            train_size = len(dataset) - val_size
            
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            # 数据加载器
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=0  # Windows兼容性
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=0
            )
            
            # 数据信息
            data_info = {
                'total_samples': len(dataset),
                'train_samples': train_size,
                'val_samples': val_size,
                'feature_dim': X_scaled.shape[1],
                'num_classes': len(np.unique(y_encoded)) if len(np.unique(y_encoded)) > 2 else 1,
                'feature_names': list(feature_data.columns),
                'target_distribution': dict(zip(*np.unique(y_encoded, return_counts=True)))
            }
            
            self.logger.info(f"数据准备完成: {data_info}")
            
            # 记录到实验跟踪
            if self.tensorboard_writer:
                self.tensorboard_writer.add_text('data_info', json.dumps(data_info, indent=2))
            
            if self.wandb_run:
                wandb.log({"data_info": data_info})
            
            return train_loader, val_loader, data_info
            
        except Exception as e:
            self.logger.error(f"数据准备失败: {str(e)}\n{traceback.format_exc()}")
            raise
    
    def create_model(self, model_type: str, data_info: Dict, **kwargs) -> nn.Module:
        """
        创建模型
        
        Args:
            model_type: 模型类型 ('mlp', 'lstm', 'cnn')
            data_info: 数据信息
            **kwargs: 模型特定参数
            
        Returns:
            PyTorch模型
        """
        try:
            input_dim = data_info['feature_dim']
            num_classes = data_info['num_classes']
            
            if model_type == 'mlp':
                # 多层感知机
                hidden_dims = kwargs.get('hidden_dims', [128, 64, 32])
                dropout_rate = kwargs.get('dropout_rate', 0.2)
                
                layers = []
                prev_dim = input_dim
                
                for hidden_dim in hidden_dims:
                    layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate)
                    ])
                    prev_dim = hidden_dim
                
                # 输出层
                if num_classes == 1:
                    layers.append(nn.Linear(prev_dim, 1))  # 回归或二分类
                else:
                    layers.append(nn.Linear(prev_dim, num_classes))  # 多分类
                
                model = nn.Sequential(*layers)
                
            elif model_type == 'lstm':
                # LSTM模型（用于时序数据）
                hidden_size = kwargs.get('hidden_size', 64)
                num_layers = kwargs.get('num_layers', 2)
                dropout_rate = kwargs.get('dropout_rate', 0.2)
                
                class LSTMModel(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.lstm = nn.LSTM(
                            input_dim, hidden_size, num_layers,
                            batch_first=True, dropout=dropout_rate
                        )
                        self.fc = nn.Linear(hidden_size, num_classes if num_classes > 1 else 1)
                        
                    def forward(self, x):
                        # 假设输入已经是序列格式
                        if x.dim() == 2:
                            x = x.unsqueeze(1)  # 添加序列维度
                        lstm_out, _ = self.lstm(x)
                        output = self.fc(lstm_out[:, -1, :])  # 使用最后一个时间步
                        return output
                
                model = LSTMModel()
                
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            # 移动到设备
            model = model.to(self.device)
            
            # 记录模型信息
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            model_info = {
                'type': model_type,
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'input_dim': input_dim,
                'output_dim': num_classes if num_classes > 1 else 1
            }
            
            self.logger.info(f"模型创建完成: {model_info}")
            
            # 记录到实验跟踪
            if self.tensorboard_writer:
                self.tensorboard_writer.add_text('model_info', json.dumps(model_info, indent=2))
            
            if self.wandb_run:
                wandb.log({"model_info": model_info})
                wandb.watch(model)
            
            self.model = model
            return model
            
        except Exception as e:
            self.logger.error(f"模型创建失败: {str(e)}\n{traceback.format_exc()}")
            raise
    
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader,
                   model: nn.Module = None) -> Dict:
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            model: 模型（可选，使用已创建的模型）
            
        Returns:
            训练结果字典
        """
        try:
            if model is None:
                model = self.model
            
            if model is None:
                raise ValueError("未提供模型")
            
            # 设置优化器
            optimizer_type = self.config.get('optimizer', 'adam').lower()
            lr = self.config.get('learning_rate', 0.001)
            
            if optimizer_type == 'adam':
                optimizer = optim.Adam(model.parameters(), lr=lr)
            elif optimizer_type == 'sgd':
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            else:
                optimizer = optim.Adam(model.parameters(), lr=lr)
            
            # 设置学习率调度器
            scheduler_config = self.config.get('scheduler', {})
            if scheduler_config.get('type') == 'reduce_on_plateau':
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    factor=scheduler_config.get('factor', 0.5),
                    patience=scheduler_config.get('patience', 10),
                    min_lr=scheduler_config.get('min_lr', 1e-6)
                )
            else:
                scheduler = None
            
            # 设置损失函数
            # 检查是否为分类任务
            sample_batch = next(iter(val_loader))
            if len(sample_batch) == 2:
                _, sample_targets = sample_batch
                unique_targets = torch.unique(sample_targets)
                
                if len(unique_targets) == 2 and torch.all((unique_targets == 0) | (unique_targets == 1)):
                    # 二分类
                    criterion = nn.BCEWithLogitsLoss()
                    task_type = 'binary_classification'
                elif len(unique_targets) > 2:
                    # 多分类
                    criterion = nn.CrossEntropyLoss()
                    task_type = 'multiclass_classification'
                else:
                    # 回归
                    criterion = nn.MSELoss()
                    task_type = 'regression'
            else:
                criterion = nn.MSELoss()
                task_type = 'regression'
            
            self.logger.info(f"任务类型: {task_type}")
            
            # 早停
            early_stopping_config = self.config.get('early_stopping', {})
            early_stopping = EarlyStopping(
                patience=early_stopping_config.get('patience', 15),
                min_delta=early_stopping_config.get('min_delta', 0.001),
                restore_best_weights=early_stopping_config.get('restore_best_weights', True)
            )
            
            # 训练循环
            epochs = self.config.get('epochs', 100)
            training_history = []
            
            for epoch in range(epochs):
                # 训练阶段
                model.train()
                train_loss = 0.0
                train_samples = 0
                
                for batch_idx, batch_data in enumerate(train_loader):
                    if len(batch_data) == 2:
                        features, targets = batch_data
                        features = features.to(self.device)
                        targets = targets.to(self.device)
                    else:
                        features = batch_data.to(self.device)
                        continue  # 跳过没有目标的批次
                    
                    optimizer.zero_grad()
                    
                    # 前向传播
                    outputs = model(features)
                    
                    # 调整输出和目标的形状
                    if task_type == 'binary_classification':
                        outputs = outputs.squeeze()
                        targets = targets.float()
                    elif task_type == 'multiclass_classification':
                        targets = targets.long()
                    elif task_type == 'regression':
                        outputs = outputs.squeeze()
                        targets = targets.float()
                    
                    loss = criterion(outputs, targets)
                    
                    # 反向传播
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item() * features.size(0)
                    train_samples += features.size(0)
                
                avg_train_loss = train_loss / train_samples
                
                # 验证阶段
                model.eval()
                val_loss = 0.0
                val_samples = 0
                all_predictions = []
                all_targets = []
                
                with torch.no_grad():
                    for batch_data in val_loader:
                        if len(batch_data) == 2:
                            features, targets = batch_data
                            features = features.to(self.device)
                            targets = targets.to(self.device)
                        else:
                            continue
                        
                        outputs = model(features)
                        
                        # 调整输出和目标的形状
                        if task_type == 'binary_classification':
                            outputs = outputs.squeeze()
                            targets = targets.float()
                            predictions = torch.sigmoid(outputs)
                        elif task_type == 'multiclass_classification':
                            targets = targets.long()
                            predictions = torch.softmax(outputs, dim=1)
                        elif task_type == 'regression':
                            outputs = outputs.squeeze()
                            targets = targets.float()
                            predictions = outputs
                        
                        loss = criterion(outputs, targets)
                        val_loss += loss.item() * features.size(0)
                        val_samples += features.size(0)
                        
                        # 收集预测和目标用于计算指标
                        all_predictions.extend(predictions.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                
                avg_val_loss = val_loss / val_samples
                
                # 计算指标
                metrics = self._calculate_metrics(
                    np.array(all_predictions), 
                    np.array(all_targets), 
                    task_type
                )
                
                # 记录历史
                epoch_info = {
                    'epoch': epoch + 1,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    **metrics
                }
                training_history.append(epoch_info)
                
                # 日志记录
                self.logger.info(f"Epoch {epoch+1}/{epochs} - "
                               f"train_loss: {avg_train_loss:.4f} - "
                               f"val_loss: {avg_val_loss:.4f}")
                
                # 实验跟踪记录
                if self.tensorboard_writer:
                    for key, value in epoch_info.items():
                        if isinstance(value, (int, float)):
                            self.tensorboard_writer.add_scalar(f'training/{key}', value, epoch)
                
                if self.wandb_run:
                    wandb.log(epoch_info)
                
                # 学习率调度
                if scheduler:
                    scheduler.step(avg_val_loss)
                
                # 早停检查
                if early_stopping(avg_val_loss, model):
                    self.logger.info(f"早停触发，停止训练于第 {epoch+1} 轮")
                    break
                
                # 模型检查点
                checkpoint_config = self.config.get('model_checkpoints', {})
                if (checkpoint_config.get('save_frequency', 10) > 0 and 
                    (epoch + 1) % checkpoint_config.get('save_frequency', 10) == 0):
                    self._save_checkpoint(model, epoch + 1, avg_val_loss)
            
            # 训练完成
            self.training_history = training_history
            self.model = model
            
            # 保存最终模型
            self._save_final_model(model, training_history)
            
            # 返回训练结果
            best_epoch = min(training_history, key=lambda x: x['val_loss'])
            
            training_result = {
                'model_name': self.model_name,
                'total_epochs': len(training_history),
                'best_epoch': best_epoch,
                'final_metrics': training_history[-1] if training_history else {},
                'training_time': None,  # TODO: 添加训练时间统计
                'model_size': sum(p.numel() for p in model.parameters())
            }
            
            self.logger.info(f"训练完成: {training_result}")
            
            return training_result
            
        except Exception as e:
            self.logger.error(f"模型训练失败: {str(e)}\n{traceback.format_exc()}")
            raise
    
    def _calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray, 
                          task_type: str) -> Dict[str, float]:
        """计算评估指标"""
        try:
            metrics = {}
            
            if task_type == 'binary_classification':
                # 二分类指标
                pred_binary = (predictions > 0.5).astype(int)
                
                metrics.update({
                    'accuracy': accuracy_score(targets, pred_binary),
                    'precision': precision_score(targets, pred_binary, average='binary', zero_division=0),
                    'recall': recall_score(targets, pred_binary, average='binary', zero_division=0),
                    'f1_score': f1_score(targets, pred_binary, average='binary', zero_division=0)
                })
                
                # AUC（如果可能）
                try:
                    metrics['auc'] = roc_auc_score(targets, predictions)
                except:
                    pass
                    
            elif task_type == 'multiclass_classification':
                # 多分类指标
                pred_classes = np.argmax(predictions, axis=1)
                
                metrics.update({
                    'accuracy': accuracy_score(targets, pred_classes),
                    'precision': precision_score(targets, pred_classes, average='weighted', zero_division=0),
                    'recall': recall_score(targets, pred_classes, average='weighted', zero_division=0),
                    'f1_score': f1_score(targets, pred_classes, average='weighted', zero_division=0)
                })
                
            elif task_type == 'regression':
                # 回归指标
                mse = np.mean((predictions - targets) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(predictions - targets))
                
                # R²分数
                ss_res = np.sum((targets - predictions) ** 2)
                ss_tot = np.sum((targets - np.mean(targets)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                metrics.update({
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2_score': r2
                })
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"指标计算失败: {str(e)}")
            return {}
    
    def _save_checkpoint(self, model: nn.Module, epoch: int, val_loss: float):
        """保存模型检查点"""
        try:
            checkpoint_dir = self.config.get('model_checkpoints', {}).get('save_dir', 'models/checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"{self.model_name}_epoch_{epoch}_loss_{val_loss:.4f}.pth"
            )
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'model_name': self.model_name,
                'config': self.config,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder
            }
            
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"检查点已保存: {checkpoint_path}")
            
        except Exception as e:
            self.logger.warning(f"检查点保存失败: {str(e)}")
    
    def _save_final_model(self, model: nn.Module, training_history: List[Dict]):
        """保存最终模型"""
        try:
            models_dir = 'models/trained'
            os.makedirs(models_dir, exist_ok=True)
            
            # 保存PyTorch模型
            model_path = os.path.join(models_dir, f"{self.model_name}.pth")
            
            model_data = {
                'model_state_dict': model.state_dict(),
                'model_name': self.model_name,
                'config': self.config,
                'training_history': training_history,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'created_at': datetime.utcnow().isoformat()
            }
            
            torch.save(model_data, model_path)
            self.logger.info(f"最终模型已保存: {model_path}")
            
            # 保存预处理器
            if self.scaler:
                scaler_path = os.path.join(models_dir, f"{self.model_name}_scaler.joblib")
                joblib.dump(self.scaler, scaler_path)
            
            if self.label_encoder:
                encoder_path = os.path.join(models_dir, f"{self.model_name}_label_encoder.joblib")
                joblib.dump(self.label_encoder, encoder_path)
            
            # 保存训练信息到数据库
            self._save_model_info_to_db(model_path, training_history)
            
        except Exception as e:
            self.logger.error(f"模型保存失败: {str(e)}")
    
    def _save_model_info_to_db(self, model_path: str, training_history: List[Dict]):
        """保存模型信息到数据库"""
        try:
            if not training_history:
                return
            
            best_epoch = min(training_history, key=lambda x: x['val_loss'])
            final_metrics = training_history[-1]
            
            model_info = {
                'model_name': self.model_name,
                'model_type': 'pytorch',
                'version': '1.0.0',
                'description': f"健康风险预测模型，训练于{datetime.now().strftime('%Y-%m-%d')}",
                'model_path': model_path,
                'config_data': json.dumps(self.config),
                'performance_metrics': json.dumps({
                    'best_val_loss': best_epoch['val_loss'],
                    'final_metrics': final_metrics,
                    'total_epochs': len(training_history)
                }),
                'training_data_info': json.dumps({
                    'training_samples': self.config.get('train_samples', 0),
                    'validation_samples': self.config.get('val_samples', 0),
                    'feature_names': getattr(self, 'feature_names', [])
                }),
                'is_active': True,
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow()
            }
            
            # 插入数据库
            columns = list(model_info.keys())
            placeholders = ', '.join(['%s'] * len(columns))
            query = f"INSERT INTO ai_models ({', '.join(columns)}) VALUES ({placeholders})"
            
            result = self.db_manager.execute_query(
                query,
                list(model_info.values()),
                return_lastrowid=True
            )
            
            if result:
                self.logger.info(f"模型信息已保存到数据库: {result}")
            
        except Exception as e:
            self.logger.warning(f"模型信息保存到数据库失败: {str(e)}")
    
    def close(self):
        """清理资源"""
        try:
            if self.tensorboard_writer:
                self.tensorboard_writer.close()
            
            if self.wandb_run:
                wandb.finish()
                
        except Exception as e:
            self.logger.warning(f"资源清理失败: {str(e)}")

# 使用示例
def example_training():
    """训练示例"""
    
    # 配置
    config = {
        'batch_size': 64,
        'learning_rate': 0.001,
        'epochs': 50,
        'validation_split': 0.2,
        'early_stopping': {
            'patience': 10,
            'min_delta': 0.001
        },
        'experiment_tracking': {
            'use_tensorboard': True,
            'use_wandb': False
        }
    }
    
    # 初始化训练器
    trainer = ModelTrainer("health_risk_predictor_v1", config)
    
    try:
        # 准备数据
        train_loader, val_loader, data_info = trainer.prepare_data()
        
        # 创建模型
        model = trainer.create_model('mlp', data_info, hidden_dims=[128, 64, 32])
        
        # 训练模型
        training_result = trainer.train_model(train_loader, val_loader, model)
        
        print(f"训练完成: {training_result}")
        
    except Exception as e:
        print(f"训练失败: {str(e)}")
    finally:
        trainer.close()

if __name__ == "__main__":
    example_training()
