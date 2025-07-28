"""
API响应优化模块
提供缓存、压缩、异步处理等功能
"""

import time
import json
import gzip
import threading
from typing import Dict, Any, Optional, List, Union, Callable
from functools import wraps
from flask import request, jsonify, make_response, g
import redis
import hashlib
from concurrent.futures import ThreadPoolExecutor
import asyncio
from dataclasses import dataclass, asdict

from utils.logger import get_logger
from utils.redis_client import get_redis_client
from config.settings import get_config

logger = get_logger(__name__)

@dataclass
class CacheConfig:
    """缓存配置"""
    ttl: int = 300  # 缓存时间(秒)
    compression: bool = True  # 是否压缩
    key_prefix: str = "api_cache"
    max_size: int = 1024 * 1024  # 最大缓存大小(字节)

@dataclass
class ResponseMetrics:
    """响应指标"""
    endpoint: str
    method: str
    status_code: int
    response_time: float
    response_size: int
    cache_hit: bool
    timestamp: float

class APIResponseOptimizer:
    """API响应优化器"""
    
    def __init__(self, config_name: str = 'development'):
        self.config = get_config(config_name)
        self.redis_client = get_redis_client()
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # 响应指标
        self.metrics = []
        self.metrics_lock = threading.Lock()
        
        # 缓存统计
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0
        }
        self.stats_lock = threading.Lock()
    
    def cache_response(self, cache_config: Optional[CacheConfig] = None):
        """响应缓存装饰器"""
        if cache_config is None:
            cache_config = CacheConfig()
            
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # 构建缓存键
                cache_key = self._build_cache_key(
                    func.__name__,
                    request.method,
                    request.url,
                    request.get_json(silent=True),
                    cache_config.key_prefix
                )
                
                # 尝试从缓存获取
                cached_response = self._get_cached_response(cache_key, cache_config.compression)
                if cached_response:
                    with self.stats_lock:
                        self.cache_stats['hits'] += 1
                    
                    # 添加缓存头
                    response = make_response(cached_response)
                    response.headers['X-Cache'] = 'HIT'
                    return response
                
                # 缓存未命中，执行原函数
                with self.stats_lock:
                    self.cache_stats['misses'] += 1
                
                start_time = time.time()
                result = func(*args, **kwargs)
                
                # 缓存响应
                if hasattr(result, 'status_code') and result.status_code == 200:
                    self._cache_response(cache_key, result.get_data(as_text=True), cache_config)
                    with self.stats_lock:
                        self.cache_stats['sets'] += 1
                
                # 添加缓存头
                if hasattr(result, 'headers'):
                    result.headers['X-Cache'] = 'MISS'
                
                return result
                
            return wrapper
        return decorator
    
    def performance_monitor(self, func):
        """性能监控装饰器"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            endpoint = request.endpoint or 'unknown'
            method = request.method
            
            try:
                result = func(*args, **kwargs)
                
                # 获取响应信息
                status_code = getattr(result, 'status_code', 200)
                response_size = len(str(result.get_data())) if hasattr(result, 'get_data') else 0
                cache_hit = getattr(result, 'headers', {}).get('X-Cache') == 'HIT'
                
                # 记录指标
                metrics = ResponseMetrics(
                    endpoint=endpoint,
                    method=method,
                    status_code=status_code,
                    response_time=time.time() - start_time,
                    response_size=response_size,
                    cache_hit=cache_hit,
                    timestamp=time.time()
                )
                
                with self.metrics_lock:
                    self.metrics.append(metrics)
                    # 限制指标数量
                    if len(self.metrics) > 1000:
                        self.metrics = self.metrics[-800:]
                
                # 慢响应日志
                response_time = time.time() - start_time
                if response_time > 2.0:  # 超过2秒
                    logger.warning(f"慢响应检测 - {method} {endpoint}: {response_time:.3f}s")
                
                return result
                
            except Exception as e:
                # 记录错误指标
                metrics = ResponseMetrics(
                    endpoint=endpoint,
                    method=method,
                    status_code=500,
                    response_time=time.time() - start_time,
                    response_size=0,
                    cache_hit=False,
                    timestamp=time.time()
                )
                
                with self.metrics_lock:
                    self.metrics.append(metrics)
                
                raise
                
        return wrapper
    
    def compress_response(self, min_size: int = 1024):
        """响应压缩装饰器"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                
                # 检查是否需要压缩
                if not self._should_compress_response(result, min_size):
                    return result
                
                try:
                    # 获取响应数据
                    response_data = result.get_data()
                    
                    # 压缩数据
                    compressed_data = gzip.compress(response_data)
                    
                    # 创建新响应
                    response = make_response(compressed_data)
                    response.headers['Content-Encoding'] = 'gzip'
                    response.headers['Content-Length'] = len(compressed_data)
                    response.headers['Vary'] = 'Accept-Encoding'
                    
                    # 复制原有头信息
                    for key, value in result.headers:
                        if key not in ['Content-Length', 'Content-Encoding']:
                            response.headers[key] = value
                    
                    logger.debug(f"响应压缩 - 原始: {len(response_data)}bytes, 压缩后: {len(compressed_data)}bytes")
                    
                    return response
                    
                except Exception as e:
                    logger.error(f"响应压缩失败: {e}")
                    return result
                    
            return wrapper
        return decorator
    
    def async_processing(self, func):
        """异步处理装饰器"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 检查是否需要异步处理
            if request.args.get('async') == 'true':
                # 生成任务ID
                task_id = self._generate_task_id()
                
                # 提交异步任务
                future = self.executor.submit(func, *args, **kwargs)
                
                # 存储任务信息
                self._store_task_info(task_id, future)
                
                return jsonify({
                    'status': 'accepted',
                    'task_id': task_id,
                    'message': '任务已提交异步处理'
                }), 202
            else:
                return func(*args, **kwargs)
                
        return wrapper
    
    def batch_processing(self, batch_size: int = 100):
        """批处理装饰器"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                data = request.get_json()
                
                # 检查是否为批处理请求
                if isinstance(data, dict) and 'batch' in data and isinstance(data['batch'], list):
                    batch_data = data['batch']
                    
                    # 分批处理
                    results = []
                    for i in range(0, len(batch_data), batch_size):
                        batch_chunk = batch_data[i:i + batch_size]
                        
                        # 处理批次
                        try:
                            # 模拟批处理调用
                            batch_results = []
                            for item in batch_chunk:
                                # 临时修改请求数据
                                original_json = request.get_json
                                request.get_json = lambda: item
                                
                                try:
                                    result = func(*args, **kwargs)
                                    batch_results.append(result.get_json() if hasattr(result, 'get_json') else result)
                                finally:
                                    request.get_json = original_json
                            
                            results.extend(batch_results)
                            
                        except Exception as e:
                            logger.error(f"批处理错误: {e}")
                            results.append({'error': str(e)})
                    
                    return jsonify({
                        'status': 'success',
                        'total': len(batch_data),
                        'results': results
                    })
                else:
                    return func(*args, **kwargs)
                    
            return wrapper
        return decorator
    
    def _build_cache_key(self, func_name: str, method: str, url: str, data: Optional[Dict], prefix: str) -> str:
        """构建缓存键"""
        key_parts = [prefix, func_name, method, url]
        
        if data:
            # 序列化请求数据
            data_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
            key_parts.append(hashlib.md5(data_str.encode()).hexdigest())
        
        return ':'.join(key_parts)
    
    def _get_cached_response(self, cache_key: str, compression: bool) -> Optional[str]:
        """获取缓存响应"""
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                if compression:
                    # 解压数据
                    return gzip.decompress(cached_data).decode('utf-8')
                else:
                    return cached_data.decode('utf-8') if isinstance(cached_data, bytes) else cached_data
        except Exception as e:
            logger.error(f"获取缓存失败: {e}")
        
        return None
    
    def _cache_response(self, cache_key: str, response_data: str, cache_config: CacheConfig):
        """缓存响应"""
        try:
            # 检查数据大小
            if len(response_data.encode()) > cache_config.max_size:
                logger.warning(f"响应数据过大，跳过缓存: {len(response_data.encode())}bytes")
                return
            
            data_to_cache = response_data.encode()
            
            if cache_config.compression:
                # 压缩数据
                data_to_cache = gzip.compress(data_to_cache)
            
            # 存储到Redis
            self.redis_client.setex(cache_key, cache_config.ttl, data_to_cache)
            
        except Exception as e:
            logger.error(f"缓存响应失败: {e}")
    
    def _should_compress_response(self, response, min_size: int) -> bool:
        """判断是否应该压缩响应"""
        if not hasattr(response, 'get_data'):
            return False
        
        # 检查Accept-Encoding头
        accept_encoding = request.headers.get('Accept-Encoding', '')
        if 'gzip' not in accept_encoding:
            return False
        
        # 检查响应大小
        response_data = response.get_data()
        if len(response_data) < min_size:
            return False
        
        # 检查Content-Type
        content_type = response.headers.get('Content-Type', '')
        compressible_types = [
            'application/json',
            'text/html',
            'text/plain',
            'text/css',
            'application/javascript'
        ]
        
        return any(ct in content_type for ct in compressible_types)
    
    def _generate_task_id(self) -> str:
        """生成任务ID"""
        return f"task_{int(time.time() * 1000)}_{threading.get_ident()}"
    
    def _store_task_info(self, task_id: str, future):
        """存储任务信息"""
        try:
            task_info = {
                'id': task_id,
                'status': 'running',
                'created_at': time.time()
            }
            
            self.redis_client.setex(f"task:{task_id}", 3600, json.dumps(task_info))
            
            # 异步更新任务状态
            def update_task_status():
                try:
                    result = future.result(timeout=300)  # 5分钟超时
                    task_info.update({
                        'status': 'completed',
                        'result': result.get_json() if hasattr(result, 'get_json') else str(result),
                        'completed_at': time.time()
                    })
                except Exception as e:
                    task_info.update({
                        'status': 'failed',
                        'error': str(e),
                        'completed_at': time.time()
                    })
                finally:
                    self.redis_client.setex(f"task:{task_id}", 3600, json.dumps(task_info))
            
            threading.Thread(target=update_task_status, daemon=True).start()
            
        except Exception as e:
            logger.error(f"存储任务信息失败: {e}")
    
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """获取任务状态"""
        try:
            task_data = self.redis_client.get(f"task:{task_id}")
            if task_data:
                return json.loads(task_data)
        except Exception as e:
            logger.error(f"获取任务状态失败: {e}")
        
        return None
    
    def clear_cache(self, pattern: str = None):
        """清理缓存"""
        try:
            if pattern:
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
                    with self.stats_lock:
                        self.cache_stats['deletes'] += len(keys)
            else:
                # 清理所有API缓存
                keys = self.redis_client.keys("api_cache:*")
                if keys:
                    self.redis_client.delete(*keys)
                    with self.stats_lock:
                        self.cache_stats['deletes'] += len(keys)
        except Exception as e:
            logger.error(f"清理缓存失败: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        with self.metrics_lock:
            metrics = self.metrics.copy()
        
        if not metrics:
            return {}
        
        # 统计分析
        total_requests = len(metrics)
        total_response_time = sum(m.response_time for m in metrics)
        avg_response_time = total_response_time / total_requests if total_requests > 0 else 0
        
        # 按端点统计
        endpoint_stats = {}
        for m in metrics:
            if m.endpoint not in endpoint_stats:
                endpoint_stats[m.endpoint] = {
                    'count': 0,
                    'total_time': 0.0,
                    'avg_time': 0.0,
                    'cache_hits': 0,
                    'status_codes': {}
                }
            
            stats = endpoint_stats[m.endpoint]
            stats['count'] += 1
            stats['total_time'] += m.response_time
            
            if m.cache_hit:
                stats['cache_hits'] += 1
            
            status_code = str(m.status_code)
            stats['status_codes'][status_code] = stats['status_codes'].get(status_code, 0) + 1
        
        # 计算平均时间
        for endpoint in endpoint_stats:
            stats = endpoint_stats[endpoint]
            stats['avg_time'] = stats['total_time'] / stats['count']
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['count']
        
        with self.stats_lock:
            cache_stats = self.cache_stats.copy()
        
        return {
            'total_requests': total_requests,
            'average_response_time': avg_response_time,
            'endpoint_statistics': endpoint_stats,
            'cache_statistics': cache_stats
        }
    
    def reset_metrics(self):
        """重置性能指标"""
        with self.metrics_lock:
            self.metrics.clear()
        
        with self.stats_lock:
            self.cache_stats = {
                'hits': 0,
                'misses': 0,
                'sets': 0,
                'deletes': 0
            }

# 单例实例
_api_optimizer = None
_optimizer_lock = threading.Lock()

def get_api_optimizer(config_name: str = 'development') -> APIResponseOptimizer:
    """获取API优化器单例"""
    global _api_optimizer
    
    if _api_optimizer is None:
        with _optimizer_lock:
            if _api_optimizer is None:
                _api_optimizer = APIResponseOptimizer(config_name)
    
    return _api_optimizer
