# 基础模型类定义
# 提供所有深度学习模型的通用接口和功能

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional, Union
from utils.logger import get_logger

logger = get_logger(__name__)

class BaseHealthModel(nn.Module, ABC):
    """
    健康预测模型基类
    定义所有深度学习模型的通用接口
    """
    
    def __init__(self, 
                 input_dim: int,
                 device: Optional[str] = None,
                 model_name: str = "BaseHealthModel"):
        """
        初始化基础模型
        
        Args:
            input_dim: 输入特征维度
            device: 计算设备 ('cuda', 'cpu', None为自动选择)
            model_name: 模型名称
        """
        super(BaseHealthModel, self).__init__()
        
        self.input_dim = input_dim
        self.model_name = model_name
        self.device = self._get_device(device)
        
        # 模型配置
        self.config = {
            'input_dim': input_dim,
            'model_name': model_name,
            'device': str(self.device),
            'created_at': 0.0,  # 使用标准数字而不是tensor
        }
        
        logger.info(f"初始化 {model_name} 模型，输入维度: {input_dim}, 设备: {self.device}")
        
    def _get_device(self, device: Optional[str] = None) -> torch.device:
        """
        获取计算设备，支持GPU/CPU自适应
        
        Args:
            device: 指定设备
            
        Returns:
            torch.device: 计算设备
        """
        if device is not None:
            return torch.device(device)
            
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            logger.info("使用CPU进行计算")
            
        return device
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播（抽象方法）
        
        Args:
            x: 输入张量
            
        Returns:
            torch.Tensor: 输出张量
        """
        pass
    
    def predict(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        预测方法
        
        Args:
            x: 输入数据 (numpy数组或torch张量)
            
        Returns:
            np.ndarray: 预测结果
        """
        self.eval()
        
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            
            x = x.to(self.device)
            
            if len(x.shape) == 1:
                x = x.unsqueeze(0)  # 添加batch维度
                
            output = self.forward(x)
            
            # 转换回numpy并返回
            result = output.cpu().numpy()
            
            return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息字典
        """
        param_count = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'model_name': self.model_name,
            'input_dim': self.input_dim,
            'total_params': param_count,
            'trainable_params': trainable_params,
            'device': str(self.device),
            'model_size_mb': param_count * 4 / (1024 * 1024),  # 假设float32
            'config': self.config
        }
        
        return info
    
    def save_model(self, filepath: str, include_optimizer: bool = False, 
                   optimizer: Optional[torch.optim.Optimizer] = None) -> None:
        """
        保存模型
        
        Args:
            filepath: 保存路径
            include_optimizer: 是否包含优化器状态
            optimizer: 优化器对象
        """
        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_config': self.config,
            'model_info': self.get_model_info()
        }
        
        if include_optimizer and optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
            
        torch.save(save_dict, filepath)
        logger.info(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str, strict: bool = True) -> Dict[str, Any]:
        """
        加载模型
        
        Args:
            filepath: 模型路径
            strict: 是否严格匹配参数
            
        Returns:
            Dict[str, Any]: 加载的模型信息
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        self.to(self.device)
        
        logger.info(f"模型已从 {filepath} 加载")
        
        return checkpoint
    
    def enable_mixed_precision(self) -> None:
        """启用混合精度训练"""
        self.config['mixed_precision'] = True
        logger.info("启用混合精度训练")
    
    def compile_model(self) -> None:
        """
        使用torch.compile优化模型（PyTorch 2.0+）
        """
        try:
            if hasattr(torch, 'compile'):
                self = torch.compile(self)
                self.config['compiled'] = True
                logger.info("模型已编译优化")
            else:
                logger.warning("当前PyTorch版本不支持torch.compile")
        except Exception as e:
            logger.warning(f"模型编译失败: {e}")


class ModelRegistry:
    """
    模型注册表
    管理不同类型的模型实例
    """
    
    def __init__(self):
        self._models = {}
        self._model_configs = {}
    
    def register(self, name: str, model: BaseHealthModel, config: Dict[str, Any] = None):
        """
        注册模型
        
        Args:
            name: 模型名称
            model: 模型实例
            config: 模型配置
        """
        self._models[name] = model
        self._model_configs[name] = config or {}
        logger.info(f"注册模型: {name}")
    
    def get_model(self, name: str) -> Optional[BaseHealthModel]:
        """
        获取模型
        
        Args:
            name: 模型名称
            
        Returns:
            BaseHealthModel: 模型实例
        """
        return self._models.get(name)
    
    def list_models(self) -> List[str]:
        """
        列出所有注册的模型
        
        Returns:
            List[str]: 模型名称列表
        """
        return list(self._models.keys())
    
    def get_model_info(self, name: str) -> Dict[str, Any]:
        """
        获取模型信息
        
        Args:
            name: 模型名称
            
        Returns:
            Dict[str, Any]: 模型信息
        """
        model = self._models.get(name)
        if model is None:
            return {}
            
        info = model.get_model_info()
        info['config'] = self._model_configs.get(name, {})
        
        return info


# 全局模型注册表实例
model_registry = ModelRegistry()
