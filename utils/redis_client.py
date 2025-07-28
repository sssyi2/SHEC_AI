"""
Redis客户端管理模块
提供Redis连接和缓存操作功能，支持Docker环境
"""

import redis
import json
import logging
from typing import Any, Optional, Dict
import pickle
import time

# 全局Redis客户端
redis_client = None

class MockRedisClient:
    """模拟Redis客户端（用于测试和开发环境）"""
    
    def __init__(self):
        self._data = {}
        self._ttl = {}
        self.connected = True
    
    def ping(self):
        """测试连接"""
        return True
    
    def set(self, key: str, value: Any, ex: Optional[int] = None):
        """设置键值"""
        try:
            self._data[key] = value
            if ex:
                self._ttl[key] = time.time() + ex
            return True
        except Exception:
            return False
    
    def setex(self, key: str, time_seconds: int, value: Any):
        """设置键值和过期时间"""
        return self.set(key, value, ex=time_seconds)
    
    def get(self, key: str):
        """获取值"""
        try:
            # 检查是否过期
            if key in self._ttl:
                if time.time() > self._ttl[key]:
                    del self._data[key]
                    del self._ttl[key]
                    return None
            
            return self._data.get(key)
        except Exception:
            return None
    
    def exists(self, key: str):
        """检查键是否存在"""
        return key in self._data
    
    def delete(self, *keys):
        """删除键"""
        deleted = 0
        for key in keys:
            if key in self._data:
                del self._data[key]
                if key in self._ttl:
                    del self._ttl[key]
                deleted += 1
        return deleted
    
    def keys(self, pattern: str = "*"):
        """获取匹配的键"""
        import fnmatch
        # 由于我们模拟decode_responses=True的行为，返回字符串而不是字节
        return [key for key in self._data.keys() if fnmatch.fnmatch(key, pattern)]
    
    def flushdb(self):
        """清空数据库"""
        self._data.clear()
        self._ttl.clear()
        return True
    
    def info(self):
        """获取服务器信息"""
        return {
            'used_memory_human': '1MB',
            'connected_clients': 1,
            'total_connections_received': 10,
            'total_commands_processed': 100
        }

def init_redis(app=None):
    """初始化Redis连接"""
    global redis_client
    
    try:
        if app and hasattr(app, 'config'):
            # Flask应用配置 - 压力测试优化
            redis_config = {
                'host': app.config.get('REDIS_HOST', 'localhost'),
                'port': app.config.get('REDIS_PORT', 6379),
                'db': app.config.get('REDIS_DB', 0),
                'password': app.config.get('REDIS_PASSWORD'),
                'decode_responses': True,
                'socket_connect_timeout': 10,    # 增加连接超时
                'socket_timeout': 10,            # 增加操作超时
                'retry_on_timeout': True,
                'max_connections': 100,          # 增加最大连接数
                'socket_keepalive': True,        # 启用keepalive
                'socket_keepalive_options': {},
                'health_check_interval': 30      # 健康检查间隔
            }
            
            # 创建连接池
            connection_pool = redis.ConnectionPool(**redis_config)
            redis_client = redis.Redis(connection_pool=connection_pool)
            
            # 测试连接
            redis_client.ping()
            
            if app.logger:
                app.logger.info("Redis连接初始化成功")
        else:
            # 直接连接（用于测试和开发）
            try:
                # 优先尝试连接Docker中的Redis
                docker_configs = [
                    {'host': 'localhost', 'port': 6379},  # Docker端口映射
                    {'host': 'shec_redis', 'port': 6379}, # Docker内部网络
                    {'host': '127.0.0.1', 'port': 6379},  # 本地回环
                ]
                
                redis_client = None
                for config in docker_configs:
                    try:
                        print(f"尝试连接Redis: {config['host']}:{config['port']}")
                        test_client = redis.Redis(
                            host=config['host'], 
                            port=config['port'], 
                            db=0, 
                            decode_responses=True,
                            socket_connect_timeout=3,
                            socket_timeout=3
                        )
                        test_client.ping()
                        redis_client = test_client
                        print(f"✅ Redis连接成功: {config['host']}:{config['port']}")
                        break
                    except Exception as e:
                        print(f"❌ Redis连接失败 {config['host']}:{config['port']}: {e}")
                        continue
                
                if redis_client is None:
                    raise Exception("所有Redis连接配置都失败")
                    
            except Exception as e:
                # 如果Redis服务不可用，使用Mock客户端
                print(f"⚠️ Redis连接失败: {str(e)}，使用Mock Redis客户端")
                redis_client = MockRedisClient()
        
    except Exception as e:
        print(f"❌ Redis初始化异常: {str(e)}，使用Mock Redis客户端")
        redis_client = MockRedisClient()

def get_redis_client():
    """获取Redis客户端"""
    if redis_client is None:
        init_redis()
    return redis_client

class CacheManager:
    """缓存管理类"""
    
    @staticmethod
    def set(key: str, value: Any, timeout: int = 300) -> bool:
        """设置缓存"""
        if not redis_client:
            return False
            
        try:
            # 序列化值
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, (int, float, str, bool)):
                serialized_value = str(value)
            else:
                # 对于复杂对象使用pickle
                serialized_value = pickle.dumps(value)
                key = f"pickle:{key}"
            
            return redis_client.setex(key, timeout, serialized_value)
            
        except Exception as e:
            logging.error(f"设置缓存失败 {key}: {str(e)}")
            return False
    
    @staticmethod
    def get(key: str) -> Optional[Any]:
        """获取缓存"""
        if not redis_client:
            return None
            
        try:
            value = redis_client.get(key)
            if value is None:
                return None
            
            # 检查是否是pickle序列化的
            if key.startswith("pickle:"):
                return pickle.loads(value)
            
            # 尝试JSON反序列化
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                # 返回原始字符串
                return value
                
        except Exception as e:
            logging.error(f"获取缓存失败 {key}: {str(e)}")
            return None
    
    @staticmethod
    def delete(key: str) -> bool:
        """删除缓存"""
        if not redis_client:
            return False
            
        try:
            return bool(redis_client.delete(key))
        except Exception as e:
            logging.error(f"删除缓存失败 {key}: {str(e)}")
            return False
    
    @staticmethod
    def exists(key: str) -> bool:
        """检查缓存是否存在"""
        if not redis_client:
            return False
            
        try:
            return bool(redis_client.exists(key))
        except Exception as e:
            logging.error(f"检查缓存存在性失败 {key}: {str(e)}")
            return False
    
    @staticmethod
    def set_hash(name: str, mapping: dict, timeout: int = 300) -> bool:
        """设置哈希缓存"""
        if not redis_client:
            return False
            
        try:
            # 序列化哈希值
            serialized_mapping = {}
            for k, v in mapping.items():
                if isinstance(v, (dict, list)):
                    serialized_mapping[k] = json.dumps(v, ensure_ascii=False)
                else:
                    serialized_mapping[k] = str(v)
            
            redis_client.hmset(name, serialized_mapping)
            redis_client.expire(name, timeout)
            return True
            
        except Exception as e:
            logging.error(f"设置哈希缓存失败 {name}: {str(e)}")
            return False
    
    @staticmethod
    def get_hash(name: str, key: str = None) -> Optional[Any]:
        """获取哈希缓存"""
        if not redis_client:
            return None
            
        try:
            if key:
                value = redis_client.hget(name, key)
                if value:
                    try:
                        return json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        return value
                return None
            else:
                values = redis_client.hgetall(name)
                result = {}
                for k, v in values.items():
                    try:
                        result[k] = json.loads(v)
                    except (json.JSONDecodeError, TypeError):
                        result[k] = v
                return result
                
        except Exception as e:
            logging.error(f"获取哈希缓存失败 {name}: {str(e)}")
            return None
    
    @staticmethod
    def clear_pattern(pattern: str) -> int:
        """按模式清除缓存"""
        if not redis_client:
            return 0
            
        try:
            keys = redis_client.keys(pattern)
            if keys:
                return redis_client.delete(*keys)
            return 0
            
        except Exception as e:
            logging.error(f"按模式清除缓存失败 {pattern}: {str(e)}")
            return 0

# 缓存装饰器
def cache_result(timeout: int = 300, key_prefix: str = ""):
    """缓存函数结果的装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = f"{key_prefix}:{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # 尝试从缓存获取
            cached_result = CacheManager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            CacheManager.set(cache_key, result, timeout)
            
            return result
        return wrapper
    return decorator
