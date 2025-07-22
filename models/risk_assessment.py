# 健康风险评估神经网络模型
# 用于评估用户患病风险和健康状态分类

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
import math

from .base_model import BaseHealthModel
from utils.logger import get_logger

logger = get_logger(__name__)

class RiskAssessmentNet(BaseHealthModel):
    """
    健康风险评估网络
    用于评估糖尿病、高血压、心血管疾病等风险
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int] = [256, 128, 64],
                 num_classes: int = 3,  # 低风险、中风险、高风险
                 dropout: float = 0.3,
                 use_batch_norm: bool = True,
                 activation: str = 'relu',
                 device: Optional[str] = None):
        """
        初始化风险评估网络
        
        Args:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
            num_classes: 分类数量（风险等级）
            dropout: dropout率
            use_batch_norm: 是否使用batch normalization
            activation: 激活函数类型
            device: 计算设备
        """
        super(RiskAssessmentNet, self).__init__(input_dim, device, "RiskAssessmentNet")
        
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.activation = activation
        
        # 更新配置
        self.config.update({
            'hidden_dims': hidden_dims,
            'num_classes': num_classes,
            'dropout': dropout,
            'use_batch_norm': use_batch_norm,
            'activation': activation,
        })
        
        # 构建模型
        self._build_model()
        
        # 移动到指定设备
        self.to(self.device)
        
        logger.info(f"RiskAssessmentNet模型初始化完成: 输入维度={input_dim}, "
                   f"隐藏层={hidden_dims}, 分类数={num_classes}")
    
    def _build_model(self):
        """构建风险评估网络"""
        
        # 获取激活函数
        if self.activation.lower() == 'relu':
            activation_fn = nn.ReLU
        elif self.activation.lower() == 'gelu':
            activation_fn = nn.GELU
        elif self.activation.lower() == 'swish':
            activation_fn = nn.SiLU
        else:
            activation_fn = nn.ReLU
        
        # 输入层
        layers = []
        
        # 输入标准化
        layers.append(nn.LayerNorm(self.input_dim))
        
        # 构建隐藏层
        prev_dim = self.input_dim
        
        for i, hidden_dim in enumerate(self.hidden_dims):
            # 线性层
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # 批标准化
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # 激活函数
            layers.append(activation_fn())
            
            # Dropout
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            
            prev_dim = hidden_dim
        
        # 分类头
        layers.append(nn.Linear(prev_dim, self.num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
        # 风险分数回归头（0-1之间的连续风险分数）
        self.risk_regressor = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            activation_fn(),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
            nn.Linear(prev_dim // 2, 1),
            nn.Sigmoid()  # 确保输出在0-1之间
        )
        
        # 特征重要性分析层
        self.feature_attention = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, input_dim]
            
        Returns:
            Dict[str, torch.Tensor]: 包含分类和回归结果的字典
        """
        # 特征重要性权重
        attention_weights = self.feature_attention(x)
        weighted_features = x * attention_weights
        
        # 通过主网络
        features = self.classifier[:-1](weighted_features)  # 除了最后一层
        
        # 分类预测
        class_logits = self.classifier[-1](features)
        class_probs = F.softmax(class_logits, dim=1)
        
        # 风险分数预测
        risk_score = self.risk_regressor(features)
        
        return {
            'class_logits': class_logits,
            'class_probs': class_probs,
            'risk_score': risk_score,
            'attention_weights': attention_weights,
            'features': features
        }
    
    def predict_risk(self, x: Union[np.ndarray, torch.Tensor]) -> Dict[str, np.ndarray]:
        """
        预测健康风险
        
        Args:
            x: 输入数据
            
        Returns:
            Dict[str, np.ndarray]: 预测结果
        """
        self.eval()
        
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            
            x = x.to(self.device)
            
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            
            outputs = self.forward(x)
            
            # 转换为numpy
            results = {}
            for key, value in outputs.items():
                results[key] = value.cpu().numpy()
            
            # 添加风险等级预测
            class_pred = np.argmax(results['class_probs'], axis=1)
            results['predicted_class'] = class_pred
            
            # 风险等级标签
            risk_labels = ['低风险', '中风险', '高风险']
            results['risk_label'] = [risk_labels[i] for i in class_pred]
            
            return results
    
    def get_feature_importance(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        获取特征重要性
        
        Args:
            x: 输入数据
            
        Returns:
            np.ndarray: 特征重要性权重
        """
        self.eval()
        
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            
            x = x.to(self.device)
            
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            
            attention_weights = self.feature_attention(x)
            
            return attention_weights.cpu().numpy()


class MultiDiseaseRiskNet(BaseHealthModel):
    """
    多疾病风险评估网络
    同时评估多种疾病的患病风险
    """
    
    def __init__(self, 
                 input_dim: int,
                 disease_configs: Dict[str, Dict[str, Any]],
                 shared_hidden_dims: List[int] = [256, 128],
                 dropout: float = 0.3,
                 device: Optional[str] = None):
        """
        初始化多疾病风险评估网络
        
        Args:
            input_dim: 输入特征维度
            disease_configs: 疾病配置 {'disease_name': {'num_classes': int, 'weight': float}}
            shared_hidden_dims: 共享层隐藏维度
            dropout: dropout率
            device: 计算设备
        """
        super(MultiDiseaseRiskNet, self).__init__(input_dim, device, "MultiDiseaseRiskNet")
        
        self.disease_configs = disease_configs
        self.shared_hidden_dims = shared_hidden_dims
        self.dropout = dropout
        
        # 更新配置
        self.config.update({
            'disease_configs': disease_configs,
            'shared_hidden_dims': shared_hidden_dims,
            'dropout': dropout,
        })
        
        # 构建模型
        self._build_model()
        
        # 移动到指定设备
        self.to(self.device)
        
        logger.info(f"MultiDiseaseRiskNet模型初始化完成，疾病数: {len(disease_configs)}")
    
    def _build_model(self):
        """构建多疾病风险评估网络"""
        
        # 共享特征提取器
        shared_layers = [nn.LayerNorm(self.input_dim)]
        
        prev_dim = self.input_dim
        for hidden_dim in self.shared_hidden_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ])
            prev_dim = hidden_dim
        
        self.shared_encoder = nn.Sequential(*shared_layers)
        
        # 疾病特定的分类头
        self.disease_heads = nn.ModuleDict()
        
        for disease_name, config in self.disease_configs.items():
            num_classes = config['num_classes']
            
            # 疾病特定的头部
            disease_head = nn.Sequential(
                nn.Linear(prev_dim, prev_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(prev_dim // 2, num_classes)
            )
            
            self.disease_heads[disease_name] = disease_head
        
        # 全局风险评估头
        self.global_risk_head = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(prev_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            Dict[str, torch.Tensor]: 各疾病的预测结果
        """
        # 共享特征提取
        shared_features = self.shared_encoder(x)
        
        # 各疾病预测
        predictions = {}
        
        for disease_name, disease_head in self.disease_heads.items():
            logits = disease_head(shared_features)
            probs = F.softmax(logits, dim=1)
            
            predictions[f"{disease_name}_logits"] = logits
            predictions[f"{disease_name}_probs"] = probs
        
        # 全局风险分数
        predictions['global_risk'] = self.global_risk_head(shared_features)
        predictions['shared_features'] = shared_features
        
        return predictions
    
    def predict_multi_disease_risk(self, x: Union[np.ndarray, torch.Tensor]) -> Dict[str, Any]:
        """
        多疾病风险预测
        
        Args:
            x: 输入数据
            
        Returns:
            Dict[str, Any]: 预测结果
        """
        self.eval()
        
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            
            x = x.to(self.device)
            
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            
            outputs = self.forward(x)
            
            results = {}
            
            # 处理各疾病预测
            for disease_name in self.disease_configs.keys():
                probs = outputs[f"{disease_name}_probs"].cpu().numpy()
                pred_class = np.argmax(probs, axis=1)
                
                results[disease_name] = {
                    'probabilities': probs,
                    'predicted_class': pred_class,
                    'risk_probability': probs[:, -1] if probs.shape[1] > 1 else probs[:, 0]
                }
            
            # 全局风险分数
            results['global_risk_score'] = outputs['global_risk'].cpu().numpy()
            
            return results


class EnsembleRiskAssessment(BaseHealthModel):
    """
    集成风险评估模型
    结合多个模型的预测结果
    """
    
    def __init__(self, 
                 models: List[BaseHealthModel],
                 model_weights: Optional[List[float]] = None,
                 device: Optional[str] = None):
        """
        初始化集成模型
        
        Args:
            models: 基础模型列表
            model_weights: 模型权重
            device: 计算设备
        """
        # 使用第一个模型的输入维度
        input_dim = models[0].input_dim if models else 0
        super(EnsembleRiskAssessment, self).__init__(input_dim, device, "EnsembleRiskAssessment")
        
        self.models = nn.ModuleList(models)
        
        if model_weights is None:
            model_weights = [1.0 / len(models)] * len(models)
        
        self.model_weights = torch.FloatTensor(model_weights).to(self.device)
        
        # 移动所有模型到指定设备
        for model in self.models:
            model.to(self.device)
        
        logger.info(f"集成模型初始化完成，包含 {len(models)} 个基础模型")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播（集成预测）
        
        Args:
            x: 输入张量
            
        Returns:
            torch.Tensor: 集成预测结果
        """
        predictions = []
        
        for model in self.models:
            pred = model(x)
            # 如果返回字典，取主要预测结果
            if isinstance(pred, dict):
                pred = pred.get('class_probs', pred.get('risk_score', list(pred.values())[0]))
            predictions.append(pred)
        
        # 加权平均
        stacked_preds = torch.stack(predictions, dim=0)
        weighted_pred = torch.sum(stacked_preds * self.model_weights.view(-1, 1, 1), dim=0)
        
        return weighted_pred