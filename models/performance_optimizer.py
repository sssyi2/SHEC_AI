"""
性能优化模型包装器
提供GPU推理优化、混合精度训练、批量推理等功能
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import time
from typing import Dict, List, Any, Tuple, Optional, Union
from functools import wraps
import concurrent.futures
import threading
from contextlib import contextmanager

from utils.logger import get_logger

logger = get_logger(__name__)

class PerformanceOptimizedModel:
    """性能优化模型包装器"""
    
    def __init__(self, model: nn.Module, device: Optional[str] = None, optimize_level: str = 'medium'):
        """
        初始化性能优化包装器
        
        Args:
            model: 要优化的PyTorch模型
            device: 计算设备
            optimize_level: 优化级别 ('basic', 'medium', 'aggressive')
        """
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimize_level = optimize_level
        
        # 性能统计
        self.inference_stats = {
            'total_inferences': 0,
            'total_time': 0.0,
            'average_time': 0.0,
            'batch_sizes': [],
            'gpu_memory_used': []
        }
        
        # 初始化优化
        self._initialize_optimizations()
    
    def _initialize_optimizations(self):
        """初始化各种优化策略"""
        logger.info(f"初始化性能优化 - 级别: {self.optimize_level}, 设备: {self.device}")
        
        # 1. 设备优化
        self.model.to(self.device)
        
        # 2. 编译优化 (PyTorch 2.0+)
        if hasattr(torch, 'compile') and self.optimize_level in ['medium', 'aggressive']:
            try:
                logger.info("启用torch.compile优化")
                if self.optimize_level == 'aggressive':
                    self.model = torch.compile(self.model, mode='max-autotune')
                else:
                    self.model = torch.compile(self.model, mode='default')
            except Exception as e:
                logger.warning(f"torch.compile优化失败: {e}")
        
        # 3. 混合精度支持
        self.scaler = GradScaler() if self.device == 'cuda' else None
        self.use_amp = self.device == 'cuda' and self.optimize_level in ['medium', 'aggressive']
        
        # 4. 推理模式优化
        if self.optimize_level in ['medium', 'aggressive']:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        
        # 5. GPU内存优化
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        logger.info("性能优化初始化完成")
    
    @contextmanager
    def inference_mode(self):
        """推理模式上下文管理器"""
        with torch.inference_mode():
            yield
    
    def profile_inference(func):
        """推理性能分析装饰器"""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            
            # GPU内存使用记录
            if self.device == 'cuda':
                torch.cuda.synchronize()
                memory_before = torch.cuda.memory_allocated()
            
            # 执行推理
            result = func(self, *args, **kwargs)
            
            # 记录统计信息
            if self.device == 'cuda':
                torch.cuda.synchronize()
                memory_after = torch.cuda.memory_allocated()
                memory_used = memory_after - memory_before
                self.inference_stats['gpu_memory_used'].append(memory_used)
            
            end_time = time.time()
            inference_time = end_time - start_time
            
            # 更新统计
            self.inference_stats['total_inferences'] += 1
            self.inference_stats['total_time'] += inference_time
            self.inference_stats['average_time'] = (
                self.inference_stats['total_time'] / self.inference_stats['total_inferences']
            )
            
            logger.debug(f"推理耗时: {inference_time:.4f}s")
            
            return result
        return wrapper
    
    @profile_inference
    def predict(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """优化的单次预测"""
        with self.inference_mode():
            inputs = inputs.to(self.device, non_blocking=True)
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(inputs, **kwargs)
            else:
                outputs = self.model(inputs, **kwargs)
            
            return outputs
    
    @profile_inference 
    def batch_predict(self, batch_inputs: List[torch.Tensor], batch_size: int = 32) -> List[torch.Tensor]:
        """优化的批量预测"""
        results = []
        
        with self.inference_mode():
            for i in range(0, len(batch_inputs), batch_size):
                batch = batch_inputs[i:i + batch_size]
                
                # 批量数据预处理
                if isinstance(batch[0], torch.Tensor):
                    batch_tensor = torch.stack(batch).to(self.device, non_blocking=True)
                else:
                    batch_tensor = torch.tensor(batch).to(self.device, non_blocking=True)
                
                # 记录批量大小
                self.inference_stats['batch_sizes'].append(len(batch))
                
                # 批量推理
                if self.use_amp:
                    with autocast():
                        batch_outputs = self.model(batch_tensor)
                else:
                    batch_outputs = self.model(batch_tensor)
                
                # 分解批量结果
                for j in range(batch_outputs.size(0)):
                    results.append(batch_outputs[j])
        
        return results
    
    def async_predict(self, inputs: torch.Tensor, **kwargs) -> concurrent.futures.Future:
        """异步预测"""
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self.predict, inputs, **kwargs)
        return future
    
    def warm_up(self, input_shape: Tuple[int, ...], num_warmup: int = 10):
        """模型预热"""
        logger.info(f"开始模型预热 - 输入形状: {input_shape}, 预热次数: {num_warmup}")
        
        dummy_input = torch.randn(*input_shape).to(self.device)
        
        with self.inference_mode():
            for i in range(num_warmup):
                _ = self.predict(dummy_input)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
        
        logger.info("模型预热完成")
    
    def optimize_memory(self):
        """内存优化"""
        if self.device == 'cuda':
            # 清理GPU缓存
            torch.cuda.empty_cache()
            
            # 设置内存分配策略
            torch.cuda.set_per_process_memory_fraction(0.8)
            
            # 启用内存池
            if hasattr(torch.cuda, 'memory_pool'):
                torch.cuda.memory_pool.set_enabled(True)
        
        logger.info("内存优化完成")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        stats = self.inference_stats.copy()
        
        if self.device == 'cuda' and torch.cuda.is_available():
            stats['gpu_info'] = {
                'device_name': torch.cuda.get_device_name(),
                'memory_allocated': torch.cuda.memory_allocated(),
                'memory_cached': torch.cuda.memory_reserved(),
                'memory_free': torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            }
        
        if stats['batch_sizes']:
            stats['average_batch_size'] = sum(stats['batch_sizes']) / len(stats['batch_sizes'])
        
        if stats['gpu_memory_used']:
            stats['average_gpu_memory'] = sum(stats['gpu_memory_used']) / len(stats['gpu_memory_used'])
            stats['max_gpu_memory'] = max(stats['gpu_memory_used'])
        
        return stats
    
    def reset_stats(self):
        """重置性能统计"""
        self.inference_stats = {
            'total_inferences': 0,
            'total_time': 0.0,
            'average_time': 0.0,
            'batch_sizes': [],
            'gpu_memory_used': []
        }

class BatchInferenceOptimizer:
    """批量推理优化器"""
    
    def __init__(self, model: PerformanceOptimizedModel, max_batch_size: int = 64):
        self.model = model
        self.max_batch_size = max_batch_size
        self.pending_requests = []
        self.lock = threading.Lock()
    
    def add_request(self, inputs: torch.Tensor, callback=None):
        """添加推理请求"""
        with self.lock:
            self.pending_requests.append({
                'inputs': inputs,
                'callback': callback,
                'timestamp': time.time()
            })
    
    def process_batch(self):
        """处理批量请求"""
        if not self.pending_requests:
            return
        
        with self.lock:
            # 取出一批请求
            batch = self.pending_requests[:self.max_batch_size]
            self.pending_requests = self.pending_requests[self.max_batch_size:]
        
        if batch:
            # 批量处理
            batch_inputs = [req['inputs'] for req in batch]
            results = self.model.batch_predict(batch_inputs)
            
            # 执行回调
            for i, req in enumerate(batch):
                if req['callback']:
                    req['callback'](results[i])

class ModelEnsembleOptimizer:
    """模型集成优化器"""
    
    def __init__(self, models: List[PerformanceOptimizedModel]):
        self.models = models
        self.weights = [1.0 / len(models)] * len(models)  # 平均权重
    
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """集成预测"""
        predictions = []
        
        # 并行推理
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.models)) as executor:
            futures = [executor.submit(model.predict, inputs) for model in self.models]
            predictions = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # 加权平均
        weighted_sum = torch.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            weighted_sum += self.weights[i] * pred
        
        return weighted_sum
    
    def set_weights(self, weights: List[float]):
        """设置集成权重"""
        if len(weights) != len(self.models):
            raise ValueError("权重数量必须与模型数量一致")
        
        # 归一化权重
        total_weight = sum(weights)
        self.weights = [w / total_weight for w in weights]

def create_optimized_model(model: nn.Module, 
                          device: Optional[str] = None,
                          optimize_level: str = 'medium',
                          warmup_shape: Optional[Tuple[int, ...]] = None) -> PerformanceOptimizedModel:
    """
    创建性能优化的模型
    
    Args:
        model: 原始模型
        device: 计算设备
        optimize_level: 优化级别
        warmup_shape: 预热输入形状
        
    Returns:
        优化后的模型
    """
    optimized_model = PerformanceOptimizedModel(model, device, optimize_level)
    
    # 内存优化
    optimized_model.optimize_memory()
    
    # 预热
    if warmup_shape:
        optimized_model.warm_up(warmup_shape)
    
    return optimized_model
