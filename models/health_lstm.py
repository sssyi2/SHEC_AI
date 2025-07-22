# 健康时序预测LSTM模型
# 用于预测用户未来的健康指标趋势

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
import math

from .base_model import BaseHealthModel
from utils.logger import get_logger

logger = get_logger(__name__)

class HealthLSTM(BaseHealthModel):
    """
    健康时序预测LSTM模型
    用于预测血压、血糖、心率等健康指标的时序变化
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 output_dim: int = 1,
                 sequence_length: int = 7,  # 默认7天的历史数据
                 dropout: float = 0.2,
                 bidirectional: bool = False,
                 device: Optional[str] = None):
        """
        初始化HealthLSTM模型
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
            output_dim: 输出维度（预测的健康指标数量）
            sequence_length: 输入序列长度
            dropout: dropout率
            bidirectional: 是否使用双向LSTM
            device: 计算设备
        """
        super(HealthLSTM, self).__init__(input_dim, device, "HealthLSTM")
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # 更新配置
        self.config.update({
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'output_dim': output_dim,
            'sequence_length': sequence_length,
            'dropout': dropout,
            'bidirectional': bidirectional,
        })
        
        # 构建模型层
        self._build_model()
        
        # 移动到指定设备
        self.to(self.device)
        
        logger.info(f"HealthLSTM模型初始化完成: 输入维度={input_dim}, 隐藏维度={hidden_dim}, "
                   f"输出维度={output_dim}, 序列长度={sequence_length}")
    
    def _build_model(self):
        """构建模型层"""
        
        # 输入层标准化
        self.input_norm = nn.LayerNorm(self.input_dim)
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
            batch_first=True
        )
        
        # 计算LSTM输出维度
        lstm_output_dim = self.hidden_dim * (2 if self.bidirectional else 1)
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_dim,
            num_heads=8,
            dropout=self.dropout,
            batch_first=True
        )
        
        # 全连接层
        self.fc_layers = nn.ModuleList([
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(lstm_output_dim // 2, lstm_output_dim // 4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(lstm_output_dim // 4, self.output_dim)
        ])
        
        # 输出层标准化
        if self.output_dim > 1:
            self.output_norm = nn.LayerNorm(self.output_dim)
    
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, sequence_length, input_dim]
            hidden: 隐藏状态（可选）
            
        Returns:
            torch.Tensor: 预测结果 [batch_size, output_dim]
        """
        batch_size = x.size(0)
        
        # 确保输入形状正确
        if len(x.shape) == 2:
            # 如果是2D，假设是单个序列，添加序列维度
            x = x.unsqueeze(1)
        
        # 输入标准化
        x = self.input_norm(x)
        
        # LSTM前向传播
        if hidden is None:
            lstm_out, hidden = self.lstm(x)
        else:
            lstm_out, hidden = self.lstm(x, hidden)
        
        # 注意力机制
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 取最后一个时间步的输出
        last_output = attn_out[:, -1, :]
        
        # 全连接层
        output = last_output
        for layer in self.fc_layers:
            output = layer(output)
        
        # 输出标准化（多输出情况）
        if hasattr(self, 'output_norm'):
            output = self.output_norm(output)
        
        return output
    
    def predict_sequence(self, x: Union[np.ndarray, torch.Tensor], 
                        future_steps: int = 1) -> np.ndarray:
        """
        序列预测，可以预测未来多个时间步
        
        Args:
            x: 输入序列数据
            future_steps: 预测未来的步数
            
        Returns:
            np.ndarray: 预测的序列
        """
        self.eval()
        
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            
            x = x.to(self.device)
            
            if len(x.shape) == 2:
                x = x.unsqueeze(0)  # 添加batch维度
            
            predictions = []
            current_input = x
            
            for _ in range(future_steps):
                pred = self.forward(current_input)
                predictions.append(pred)
                
                # 更新输入序列（滑动窗口）
                # 移除第一个时间步，添加预测结果作为新的最后一个时间步
                if future_steps > 1:
                    # 需要将预测结果转换为与输入相同的特征维度
                    # 这里简化处理，实际应用中可能需要更复杂的策略
                    new_step = torch.zeros_like(current_input[:, -1:, :])
                    new_step[:, :, :self.output_dim] = pred.unsqueeze(1)
                    
                    current_input = torch.cat([
                        current_input[:, 1:, :],  # 移除第一个时间步
                        new_step  # 添加新的预测步
                    ], dim=1)
            
            # 合并预测结果
            result = torch.stack(predictions, dim=1)
            result = result.cpu().numpy()
            
            return result.squeeze(0) if result.shape[0] == 1 else result
    
    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        初始化隐藏状态
        
        Args:
            batch_size: 批次大小
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (h0, c0)
        """
        num_directions = 2 if self.bidirectional else 1
        
        h0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_dim).to(self.device)
        
        return h0, c0


class MultiTaskHealthLSTM(BaseHealthModel):
    """
    多任务健康预测LSTM模型
    同时预测多个健康指标（血压、血糖、心率等）
    """
    
    def __init__(self, 
                 input_dim: int,
                 task_configs: Dict[str, Dict[str, Any]],
                 shared_hidden_dim: int = 128,
                 num_shared_layers: int = 2,
                 sequence_length: int = 7,
                 dropout: float = 0.2,
                 device: Optional[str] = None):
        """
        初始化多任务LSTM模型
        
        Args:
            input_dim: 输入特征维度
            task_configs: 任务配置字典 {'task_name': {'output_dim': int, 'weight': float}}
            shared_hidden_dim: 共享LSTM隐藏层维度
            num_shared_layers: 共享LSTM层数
            sequence_length: 输入序列长度
            dropout: dropout率
            device: 计算设备
        """
        super(MultiTaskHealthLSTM, self).__init__(input_dim, device, "MultiTaskHealthLSTM")
        
        self.task_configs = task_configs
        self.shared_hidden_dim = shared_hidden_dim
        self.num_shared_layers = num_shared_layers
        self.sequence_length = sequence_length
        self.dropout = dropout
        
        # 更新配置
        self.config.update({
            'task_configs': task_configs,
            'shared_hidden_dim': shared_hidden_dim,
            'num_shared_layers': num_shared_layers,
            'sequence_length': sequence_length,
            'dropout': dropout,
        })
        
        # 构建模型
        self._build_model()
        
        # 移动到指定设备
        self.to(self.device)
        
        logger.info(f"MultiTaskHealthLSTM模型初始化完成，任务数: {len(task_configs)}")
    
    def _build_model(self):
        """构建多任务模型"""
        
        # 输入标准化
        self.input_norm = nn.LayerNorm(self.input_dim)
        
        # 共享LSTM层
        self.shared_lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.shared_hidden_dim,
            num_layers=self.num_shared_layers,
            dropout=self.dropout if self.num_shared_layers > 1 else 0,
            batch_first=True
        )
        
        # 任务特定的头部
        self.task_heads = nn.ModuleDict()
        
        for task_name, task_config in self.task_configs.items():
            output_dim = task_config['output_dim']
            
            # 每个任务的专用层
            task_head = nn.Sequential(
                nn.Linear(self.shared_hidden_dim, self.shared_hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.shared_hidden_dim // 2, output_dim)
            )
            
            self.task_heads[task_name] = task_head
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, sequence_length, input_dim]
            
        Returns:
            Dict[str, torch.Tensor]: 各任务的预测结果
        """
        # 确保输入形状正确
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # 输入标准化
        x = self.input_norm(x)
        
        # 共享LSTM层
        shared_out, _ = self.shared_lstm(x)
        
        # 取最后一个时间步的输出
        last_output = shared_out[:, -1, :]
        
        # 各任务预测
        predictions = {}
        for task_name, task_head in self.task_heads.items():
            predictions[task_name] = task_head(last_output)
        
        return predictions
    
    def predict_multi_task(self, x: Union[np.ndarray, torch.Tensor]) -> Dict[str, np.ndarray]:
        """
        多任务预测
        
        Args:
            x: 输入数据
            
        Returns:
            Dict[str, np.ndarray]: 各任务的预测结果
        """
        self.eval()
        
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            
            x = x.to(self.device)
            
            if len(x.shape) == 1:
                x = x.unsqueeze(0).unsqueeze(0)
            elif len(x.shape) == 2:
                x = x.unsqueeze(0)
            
            predictions = self.forward(x)
            
            # 转换为numpy
            results = {}
            for task_name, pred in predictions.items():
                results[task_name] = pred.cpu().numpy()
            
            return results