# 缓存策略管理模块
# 提供统一的缓存管理服务，包括预测结果缓存、模型权重缓存、用户会话缓存等

import json
import pickle
import hashlib
import gzip
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

from utils.redis_client import get_redis_client, init_redis
from utils.logger import get_logger
from config.settings import get_config

logger = get_logger(__name__)
config = get_config()

class CacheType(Enum):
    """缓存类型枚举"""
    PREDICTION = "prediction"
    MODEL = "model"
    SESSION = "session"
    USER_DATA = "user_data"
    STATISTICS = "statistics"
    TEMPORARY = "temporary"

@dataclass
class CacheConfig:
    """缓存配置"""
    default_ttl: int = 3600  # 默认过期时间（秒）
    max_memory_size: int = 100 * 1024 * 1024  # 最大内存使用（100MB）
    compression_threshold: int = 1024  # 压缩阈值（字节）
    enable_compression: bool = True
    enable_memory_cache: bool = True
    
    # 不同类型的缓存配置
    cache_ttl: Dict[CacheType, int] = None
    
    def __post_init__(self):
        if self.cache_ttl is None:
            self.cache_ttl = {
                CacheType.PREDICTION: 1800,      # 30分钟
                CacheType.MODEL: 7200,           # 2小时
                CacheType.SESSION: 3600,         # 1小时
                CacheType.USER_DATA: 600,        # 10分钟
                CacheType.STATISTICS: 300,       # 5分钟
                CacheType.TEMPORARY: 60          # 1分钟
            }

class CacheManager:
    """统一缓存管理器"""
    
    def __init__(self, config: CacheConfig = None):
        """初始化缓存管理器"""
        self.config = config or CacheConfig()
        
        # 初始化Redis连接
        try:
            init_redis()
            self.redis_client = get_redis_client()
            if self.redis_client:
                logger.info(f"缓存管理器Redis客户端初始化成功，类型: {type(self.redis_client)}")
            else:
                logger.warning("缓存管理器Redis客户端为None，将使用Mock客户端")
        except Exception as e:
            logger.error(f"缓存管理器Redis客户端初始化失败: {e}")
            self.redis_client = None
        
        # 内存缓存
        self.memory_cache = {} if self.config.enable_memory_cache else None
        self.memory_cache_size = 0
        self.cache_lock = threading.RLock()
        
        # 缓存统计
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'memory_hits': 0,
            'redis_hits': 0
        }
        
        # 后台清理任务
        self.cleanup_executor = ThreadPoolExecutor(max_workers=1)
        self._start_cleanup_task()
        
        logger.info("缓存管理器初始化完成")
    
    def _generate_key(self, cache_type: CacheType, identifier: str, 
                     namespace: str = None) -> str:
        """生成缓存键"""
        parts = [cache_type.value]
        if namespace:
            parts.append(namespace)
        parts.append(identifier)
        return ":".join(parts)
    
    def _serialize_data(self, data: Any) -> bytes:
        """序列化数据"""
        try:
            # 尝试JSON序列化（更快，兼容性好）
            if isinstance(data, (dict, list, str, int, float, bool, type(None))):
                serialized = json.dumps(data, default=str, ensure_ascii=False).encode('utf-8')
            else:
                # 使用pickle处理复杂对象
                serialized = pickle.dumps(data)
            
            # 压缩大数据
            if (self.config.enable_compression and 
                len(serialized) > self.config.compression_threshold):
                serialized = gzip.compress(serialized)
                return b'compressed:' + serialized
            
            return serialized
            
        except Exception as e:
            logger.error(f"数据序列化失败: {str(e)}")
            raise
    
    def _deserialize_data(self, data: bytes) -> Any:
        """反序列化数据"""
        try:
            if data.startswith(b'compressed:'):
                # 解压缩
                data = gzip.decompress(data[11:])
            
            # 尝试JSON反序列化
            try:
                return json.loads(data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # 使用pickle
                return pickle.loads(data)
                
        except Exception as e:
            logger.error(f"数据反序列化失败: {str(e)}")
            raise
    
    def _estimate_size(self, data: Any) -> int:
        """估算数据大小"""
        try:
            if isinstance(data, (str, bytes)):
                return len(data)
            elif isinstance(data, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) 
                          for k, v in data.items())
            elif isinstance(data, (list, tuple)):
                return sum(self._estimate_size(item) for item in data)
            else:
                # 使用pickle序列化大小作为估算
                return len(pickle.dumps(data))
        except:
            return 1024  # 默认估算1KB
    
    def set(self, cache_type: CacheType, identifier: str, data: Any,
            ttl: Optional[int] = None, namespace: str = None,
            force_redis: bool = False) -> bool:
        """设置缓存"""
        try:
            key = self._generate_key(cache_type, identifier, namespace)
            ttl = ttl or self.config.cache_ttl.get(cache_type, self.config.default_ttl)
            
            # 序列化数据
            serialized_data = self._serialize_data(data)
            data_size = len(serialized_data)
            
            # 内存缓存策略
            if (self.memory_cache is not None and not force_redis and
                data_size < self.config.compression_threshold):
                
                with self.cache_lock:
                    # 检查内存使用限制
                    if self.memory_cache_size + data_size > self.config.max_memory_size:
                        self._cleanup_memory_cache()
                    
                    # 存储到内存
                    self.memory_cache[key] = {
                        'data': data,
                        'timestamp': datetime.now(),
                        'ttl': ttl,
                        'size': data_size
                    }
                    self.memory_cache_size += data_size
            
            # Redis缓存
            success = self.redis_client.setex(key, ttl, serialized_data)
            
            self.stats['sets'] += 1
            
            if success:
                logger.debug(f"缓存设置成功: {key}, TTL: {ttl}s, 大小: {data_size}bytes")
                return True
            else:
                logger.warning(f"Redis缓存设置失败: {key}")
                return False
                
        except Exception as e:
            logger.error(f"设置缓存失败: {str(e)}")
            return False
    
    def get(self, cache_type: CacheType, identifier: str,
            namespace: str = None) -> Optional[Any]:
        """获取缓存"""
        try:
            key = self._generate_key(cache_type, identifier, namespace)
            
            # 先检查内存缓存
            if self.memory_cache is not None:
                with self.cache_lock:
                    if key in self.memory_cache:
                        cache_item = self.memory_cache[key]
                        
                        # 检查是否过期
                        if (datetime.now() - cache_item['timestamp']).seconds < cache_item['ttl']:
                            self.stats['hits'] += 1
                            self.stats['memory_hits'] += 1
                            logger.debug(f"内存缓存命中: {key}")
                            return cache_item['data']
                        else:
                            # 过期，删除
                            self.memory_cache_size -= cache_item['size']
                            del self.memory_cache[key]
            
            # Redis缓存
            cached_data = self.redis_client.get(key)
            if cached_data is not None:
                data = self._deserialize_data(cached_data)
                self.stats['hits'] += 1
                self.stats['redis_hits'] += 1
                logger.debug(f"Redis缓存命中: {key}")
                return data
            
            self.stats['misses'] += 1
            return None
            
        except Exception as e:
            logger.error(f"获取缓存失败: {str(e)}")
            self.stats['misses'] += 1
            return None
    
    def delete(self, cache_type: CacheType, identifier: str,
               namespace: str = None) -> bool:
        """删除缓存"""
        try:
            key = self._generate_key(cache_type, identifier, namespace)
            
            # 删除内存缓存
            if self.memory_cache is not None:
                with self.cache_lock:
                    if key in self.memory_cache:
                        self.memory_cache_size -= self.memory_cache[key]['size']
                        del self.memory_cache[key]
            
            # 删除Redis缓存
            result = self.redis_client.delete(key)
            self.stats['deletes'] += 1
            
            logger.debug(f"缓存删除: {key}")
            return bool(result)
            
        except Exception as e:
            logger.error(f"删除缓存失败: {str(e)}")
            return False
    
    def exists(self, cache_type: CacheType, identifier: str,
               namespace: str = None) -> bool:
        """检查缓存是否存在"""
        try:
            key = self._generate_key(cache_type, identifier, namespace)
            
            # 检查内存缓存
            if self.memory_cache is not None:
                with self.cache_lock:
                    if key in self.memory_cache:
                        cache_item = self.memory_cache[key]
                        if (datetime.now() - cache_item['timestamp']).seconds < cache_item['ttl']:
                            return True
                        else:
                            # 过期，删除
                            self.memory_cache_size -= cache_item['size']
                            del self.memory_cache[key]
            
            # 检查Redis缓存
            return bool(self.redis_client.exists(key))
            
        except Exception as e:
            logger.error(f"检查缓存存在性失败: {str(e)}")
            return False
    
    def get_or_set(self, cache_type: CacheType, identifier: str,
                   factory_func, ttl: Optional[int] = None,
                   namespace: str = None, **kwargs) -> Any:
        """获取缓存，如果不存在则调用工厂函数生成并缓存"""
        # 先尝试获取缓存
        cached_data = self.get(cache_type, identifier, namespace)
        if cached_data is not None:
            return cached_data
        
        # 生成新数据
        try:
            if asyncio.iscoroutinefunction(factory_func):
                # 异步函数需要特殊处理
                raise ValueError("异步工厂函数需要使用 async_get_or_set")
            
            new_data = factory_func(**kwargs)
            
            # 缓存新数据
            self.set(cache_type, identifier, new_data, ttl, namespace)
            return new_data
            
        except Exception as e:
            logger.error(f"工厂函数执行失败: {str(e)}")
            raise
    
    async def async_get_or_set(self, cache_type: CacheType, identifier: str,
                              async_factory_func, ttl: Optional[int] = None,
                              namespace: str = None, **kwargs) -> Any:
        """异步版本的获取或设置缓存"""
        # 先尝试获取缓存
        cached_data = self.get(cache_type, identifier, namespace)
        if cached_data is not None:
            return cached_data
        
        # 生成新数据
        try:
            new_data = await async_factory_func(**kwargs)
            
            # 缓存新数据
            self.set(cache_type, identifier, new_data, ttl, namespace)
            return new_data
            
        except Exception as e:
            logger.error(f"异步工厂函数执行失败: {str(e)}")
            raise
    
    def clear_by_pattern(self, pattern: str) -> int:
        """根据模式清除缓存"""
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                deleted = self.redis_client.delete(*keys)
                
                # 同时清理内存缓存
                if self.memory_cache is not None:
                    with self.cache_lock:
                        # 处理Redis键的编码问题
                        redis_keys = []
                        for key in keys:
                            if isinstance(key, bytes):
                                redis_keys.append(key.decode())
                            else:
                                redis_keys.append(str(key))
                        
                        memory_keys_to_delete = [
                            key for key in self.memory_cache.keys() 
                            if key in redis_keys
                        ]
                        for key in memory_keys_to_delete:
                            self.memory_cache_size -= self.memory_cache[key]['size']
                            del self.memory_cache[key]
                
                logger.info(f"清除缓存: 模式 {pattern}, 删除 {deleted} 个键")
                return deleted
            return 0
            
        except Exception as e:
            logger.error(f"按模式清除缓存失败: {str(e)}")
            return 0
            return 0
    
    def clear_by_type(self, cache_type: CacheType, namespace: str = None) -> int:
        """根据类型清除缓存"""
        if namespace:
            pattern = f"{cache_type.value}:{namespace}:*"
        else:
            pattern = f"{cache_type.value}:*"
        
        return self.clear_by_pattern(pattern)
    
    def _cleanup_memory_cache(self):
        """清理内存缓存（LRU策略）"""
        if self.memory_cache is None:
            return
        
        try:
            # 按时间戳排序，删除最旧的缓存项
            sorted_items = sorted(
                self.memory_cache.items(),
                key=lambda x: x[1]['timestamp']
            )
            
            # 删除一半的缓存项
            items_to_delete = len(sorted_items) // 2
            for i in range(items_to_delete):
                key, item = sorted_items[i]
                self.memory_cache_size -= item['size']
                del self.memory_cache[key]
            
            logger.info(f"内存缓存清理完成，删除 {items_to_delete} 个项目")
            
        except Exception as e:
            logger.error(f"内存缓存清理失败: {str(e)}")
    
    def _start_cleanup_task(self):
        """启动后台清理任务"""
        def cleanup_worker():
            while True:
                try:
                    # 每5分钟清理一次过期的内存缓存
                    if self.memory_cache is not None:
                        with self.cache_lock:
                            current_time = datetime.now()
                            expired_keys = []
                            
                            for key, item in self.memory_cache.items():
                                if (current_time - item['timestamp']).seconds >= item['ttl']:
                                    expired_keys.append(key)
                            
                            for key in expired_keys:
                                self.memory_cache_size -= self.memory_cache[key]['size']
                                del self.memory_cache[key]
                            
                            if expired_keys:
                                logger.debug(f"清理过期内存缓存: {len(expired_keys)} 个项目")
                    
                    # 等待5分钟
                    threading.Event().wait(300)
                    
                except Exception as e:
                    logger.error(f"后台清理任务异常: {str(e)}")
                    threading.Event().wait(60)  # 出错后等待1分钟
        
        # 启动后台线程
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        memory_info = {}
        if self.memory_cache is not None:
            with self.cache_lock:
                memory_info = {
                    'memory_cache_items': len(self.memory_cache),
                    'memory_cache_size_bytes': self.memory_cache_size,
                    'memory_cache_size_mb': round(self.memory_cache_size / 1024 / 1024, 2)
                }
        
        return {
            'total_requests': total_requests,
            'hit_rate_percent': round(hit_rate, 2),
            'memory_hit_rate_percent': round(
                (self.stats['memory_hits'] / total_requests * 100) if total_requests > 0 else 0, 2
            ),
            'redis_hit_rate_percent': round(
                (self.stats['redis_hits'] / total_requests * 100) if total_requests > 0 else 0, 2
            ),
            **self.stats,
            **memory_info,
            'redis_info': self._get_redis_info()
        }
    
    def _get_redis_info(self) -> Dict[str, Any]:
        """获取Redis信息"""
        try:
            info = self.redis_client.info()
            return {
                'used_memory_human': info.get('used_memory_human', 'N/A'),
                'connected_clients': info.get('connected_clients', 0),
                'total_connections_received': info.get('total_connections_received', 0),
                'total_commands_processed': info.get('total_commands_processed', 0)
            }
        except Exception:
            return {'error': 'Redis信息获取失败'}
    
    def clear_all(self):
        """清空所有缓存"""
        try:
            # 清空Redis
            self.redis_client.flushdb()
            
            # 清空内存缓存
            if self.memory_cache is not None:
                with self.cache_lock:
                    self.memory_cache.clear()
                    self.memory_cache_size = 0
            
            # 重置统计
            self.stats = {
                'hits': 0,
                'misses': 0,
                'sets': 0,
                'deletes': 0,
                'memory_hits': 0,
                'redis_hits': 0
            }
            
            logger.info("所有缓存已清空")
            
        except Exception as e:
            logger.error(f"清空缓存失败: {str(e)}")
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'cleanup_executor'):
            self.cleanup_executor.shutdown(wait=True)


# 全局缓存管理器实例
cache_manager = CacheManager()


# 便捷的缓存装饰器
def cache_result(cache_type: CacheType, identifier_func=None, 
                ttl: Optional[int] = None, namespace: str = None):
    """缓存结果装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 生成缓存标识符
            if identifier_func:
                identifier = identifier_func(*args, **kwargs)
            else:
                # 使用函数名和参数哈希作为标识符
                func_name = func.__name__
                args_str = str(args) + str(sorted(kwargs.items()))
                identifier = f"{func_name}_{hashlib.md5(args_str.encode()).hexdigest()[:8]}"
            
            return cache_manager.get_or_set(
                cache_type, identifier, func, ttl, namespace,
                *args, **kwargs
            )
        
        async def async_wrapper(*args, **kwargs):
            # 异步版本
            if identifier_func:
                identifier = identifier_func(*args, **kwargs)
            else:
                func_name = func.__name__
                args_str = str(args) + str(sorted(kwargs.items()))
                identifier = f"{func_name}_{hashlib.md5(args_str.encode()).hexdigest()[:8]}"
            
            return await cache_manager.async_get_or_set(
                cache_type, identifier, func, ttl, namespace,
                *args, **kwargs
            )
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator
