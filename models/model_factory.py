# 模型工厂和管理器
# 提供统一的模型创建、加载和管理接口

import torch
import torch.nn as nn
import os
import json
import pickle
from typing import Dict, List, Any, Optional, Union, Type, Tuple
from datetime import datetime

from .base_model import BaseHealthModel, model_registry
from .health_lstm import HealthLSTM, MultiTaskHealthLSTM
from .risk_assessment import RiskAssessmentNet, MultiDiseaseRiskNet, EnsembleRiskAssessment
from utils.logger import get_logger

logger = get_logger(__name__)

class ModelFactory:
    """
    模型工厂类
    负责创建和配置各种深度学习模型
    """
    
    # 注册可用的模型类
    MODEL_REGISTRY = {
        'health_lstm': HealthLSTM,
        'multitask_lstm': MultiTaskHealthLSTM,
        'risk_assessment': RiskAssessmentNet,
        'multidisease_risk': MultiDiseaseRiskNet,
        'ensemble_risk': EnsembleRiskAssessment,
    }
    
    @classmethod
    def create_model(cls, model_type: str, config: Dict[str, Any]) -> BaseHealthModel:
        """
        创建模型实例
        
        Args:
            model_type: 模型类型
            config: 模型配置
            
        Returns:
            BaseHealthModel: 模型实例
        """
        if model_type not in cls.MODEL_REGISTRY:
            raise ValueError(f"未知的模型类型: {model_type}。可用类型: {list(cls.MODEL_REGISTRY.keys())}")
        
        model_class = cls.MODEL_REGISTRY[model_type]
        
        try:
            model = model_class(**config)
            logger.info(f"成功创建 {model_type} 模型")
            return model
        except Exception as e:
            logger.error(f"创建模型失败: {e}")
            raise
    
    @classmethod
    def create_health_lstm(cls, 
                          input_dim: int,
                          hidden_dim: int = 128,
                          num_layers: int = 2,
                          output_dim: int = 1,
                          sequence_length: int = 7,
                          dropout: float = 0.2,
                          bidirectional: bool = False,
                          device: Optional[str] = None) -> HealthLSTM:
        """创建健康LSTM模型"""
        config = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'output_dim': output_dim,
            'sequence_length': sequence_length,
            'dropout': dropout,
            'bidirectional': bidirectional,
            'device': device
        }
        return cls.create_model('health_lstm', config)
    
    @classmethod
    def create_risk_assessment(cls,
                              input_dim: int,
                              hidden_dims: List[int] = [256, 128, 64],
                              num_classes: int = 3,
                              dropout: float = 0.3,
                              device: Optional[str] = None) -> RiskAssessmentNet:
        """创建风险评估模型"""
        config = {
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'num_classes': num_classes,
            'dropout': dropout,
            'device': device
        }
        return cls.create_model('risk_assessment', config)
    
    @classmethod
    def create_multitask_lstm(cls,
                             input_dim: int,
                             task_configs: Dict[str, Dict[str, Any]],
                             shared_hidden_dim: int = 128,
                             device: Optional[str] = None) -> MultiTaskHealthLSTM:
        """创建多任务LSTM模型"""
        config = {
            'input_dim': input_dim,
            'task_configs': task_configs,
            'shared_hidden_dim': shared_hidden_dim,
            'device': device
        }
        return cls.create_model('multitask_lstm', config)
    
    @classmethod
    def register_custom_model(cls, model_name: str, model_class: Type[BaseHealthModel]):
        """
        注册自定义模型类
        
        Args:
            model_name: 模型名称
            model_class: 模型类
        """
        cls.MODEL_REGISTRY[model_name] = model_class
        logger.info(f"注册自定义模型: {model_name}")


class ModelManager:
    """
    模型管理器
    负责模型的保存、加载、版本管理等
    """
    
    def __init__(self, model_dir: str = "models/saved"):
        """
        初始化模型管理器
        
        Args:
            model_dir: 模型保存目录
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # 模型元数据
        self.metadata_file = os.path.join(model_dir, "models_metadata.json")
        self._load_metadata()
        
        logger.info(f"模型管理器初始化完成，保存目录: {model_dir}")
    
    def _load_metadata(self):
        """加载模型元数据"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def _save_metadata(self):
        """保存模型元数据"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
    
    def save_model(self, 
                   model: BaseHealthModel,
                   model_name: str,
                   version: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None,
                   optimizer: Optional[torch.optim.Optimizer] = None) -> str:
        """
        保存模型
        
        Args:
            model: 模型实例
            model_name: 模型名称
            version: 版本号（可选，默认使用时间戳）
            metadata: 额外的元数据
            optimizer: 优化器（可选）
            
        Returns:
            str: 模型保存路径
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建版本目录
        version_dir = os.path.join(self.model_dir, model_name, version)
        os.makedirs(version_dir, exist_ok=True)
        
        # 模型文件路径
        model_path = os.path.join(version_dir, "model.pth")
        config_path = os.path.join(version_dir, "config.json")
        
        # 保存模型
        save_dict = {
            'model_state_dict': model.state_dict(),
            'model_config': model.config,
            'model_class': model.__class__.__name__,
            'model_info': model.get_model_info(),
            'save_time': datetime.now().isoformat(),
        }
        
        if optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(save_dict, model_path)
        
        # 保存配置文件（需要处理不可序列化的对象）
        config_for_json = {}
        for key, value in save_dict['model_config'].items():
            if hasattr(value, 'tolist'):  # numpy数组
                config_for_json[key] = value.tolist()
            elif hasattr(value, 'item'):  # torch tensor标量
                config_for_json[key] = value.item()
            elif str(type(value)) == "<class 'torch.Tensor'>":
                config_for_json[key] = value.cpu().numpy().tolist() if value.numel() > 1 else value.item()
            else:
                config_for_json[key] = value
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_for_json, f, ensure_ascii=False, indent=2)
        
        # 更新元数据
        if model_name not in self.metadata:
            self.metadata[model_name] = {'versions': []}
        
        version_metadata = {
            'version': version,
            'model_class': model.__class__.__name__,
            'save_time': save_dict['save_time'],
            'model_path': model_path,
            'config_path': config_path,
            'model_info': save_dict['model_info']
        }
        
        if metadata:
            version_metadata['custom_metadata'] = metadata
        
        # 添加或更新版本信息
        existing_versions = [v['version'] for v in self.metadata[model_name]['versions']]
        if version in existing_versions:
            # 更新现有版本
            for i, v in enumerate(self.metadata[model_name]['versions']):
                if v['version'] == version:
                    self.metadata[model_name]['versions'][i] = version_metadata
                    break
        else:
            # 添加新版本
            self.metadata[model_name]['versions'].append(version_metadata)
        
        # 按时间排序版本
        self.metadata[model_name]['versions'].sort(
            key=lambda x: x['save_time'], reverse=True
        )
        
        self._save_metadata()
        
        logger.info(f"模型已保存: {model_name} v{version} -> {model_path}")
        return model_path
    
    def load_model(self, 
                   model_name: str,
                   version: Optional[str] = None,
                   device: Optional[str] = None) -> Tuple[BaseHealthModel, Dict[str, Any]]:
        """
        加载模型
        
        Args:
            model_name: 模型名称
            version: 版本号（可选，默认加载最新版本）
            device: 设备（可选）
            
        Returns:
            Tuple[BaseHealthModel, Dict[str, Any]]: 模型实例和元数据
        """
        if model_name not in self.metadata:
            raise ValueError(f"模型 {model_name} 不存在")
        
        versions = self.metadata[model_name]['versions']
        if not versions:
            raise ValueError(f"模型 {model_name} 没有可用版本")
        
        # 选择版本
        if version is None:
            version_info = versions[0]  # 最新版本
        else:
            version_info = None
            for v in versions:
                if v['version'] == version:
                    version_info = v
                    break
            
            if version_info is None:
                raise ValueError(f"版本 {version} 不存在")
        
        model_path = version_info['model_path']
        
        # 加载模型检查点
        checkpoint = torch.load(model_path, map_location='cpu')
        model_class_name = checkpoint['model_class']
        model_config = checkpoint['model_config']
        
        # 找到对应的模型类
        model_class = None
        for cls_name, cls in ModelFactory.MODEL_REGISTRY.items():
            if cls.__name__ == model_class_name:
                model_class = cls
                break
        
        if model_class is None:
            raise ValueError(f"未知的模型类: {model_class_name}")
        
        # 创建模型实例
        if device:
            model_config['device'] = device
        
        # 清理配置中不属于模型构造函数的参数
        model_init_config = model_config.copy()
        non_init_params = ['model_name', 'created_at']  # 这些不是构造函数参数
        for param in non_init_params:
            model_init_config.pop(param, None)
        
        model = model_class(**model_init_config)
        
        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 移动到正确的设备
        target_device = model.device if hasattr(model, 'device') else torch.device('cpu')
        model.to(target_device)
        
        logger.info(f"模型已加载: {model_name} v{version_info['version']}")
        
        return model, version_info
    
    def list_models(self) -> Dict[str, List[str]]:
        """
        列出所有可用模型
        
        Returns:
            Dict[str, List[str]]: 模型名称及其版本列表
        """
        result = {}
        for model_name, info in self.metadata.items():
            result[model_name] = [v['version'] for v in info['versions']]
        return result
    
    def delete_model(self, model_name: str, version: Optional[str] = None):
        """
        删除模型
        
        Args:
            model_name: 模型名称
            version: 版本号（可选，删除指定版本；为None时删除所有版本）
        """
        if model_name not in self.metadata:
            raise ValueError(f"模型 {model_name} 不存在")
        
        if version is None:
            # 删除整个模型
            model_dir = os.path.join(self.model_dir, model_name)
            import shutil
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
            
            del self.metadata[model_name]
            logger.info(f"已删除模型: {model_name}")
        else:
            # 删除指定版本
            versions = self.metadata[model_name]['versions']
            version_to_remove = None
            
            for i, v in enumerate(versions):
                if v['version'] == version:
                    version_to_remove = i
                    # 删除文件
                    version_dir = os.path.dirname(v['model_path'])
                    import shutil
                    if os.path.exists(version_dir):
                        shutil.rmtree(version_dir)
                    break
            
            if version_to_remove is not None:
                versions.pop(version_to_remove)
                logger.info(f"已删除模型版本: {model_name} v{version}")
                
                # 如果没有版本了，删除整个模型记录
                if not versions:
                    del self.metadata[model_name]
            else:
                raise ValueError(f"版本 {version} 不存在")
        
        self._save_metadata()
    
    def get_model_info(self, model_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        获取模型信息
        
        Args:
            model_name: 模型名称
            version: 版本号
            
        Returns:
            Dict[str, Any]: 模型信息
        """
        if model_name not in self.metadata:
            raise ValueError(f"模型 {model_name} 不存在")
        
        versions = self.metadata[model_name]['versions']
        
        if version is None:
            return {'model_name': model_name, 'versions': versions}
        else:
            for v in versions:
                if v['version'] == version:
                    return v
            
            raise ValueError(f"版本 {version} 不存在")


# 全局模型管理器实例
model_manager = ModelManager()

# 向后兼容的别名
HealthModelFactory = ModelFactory
