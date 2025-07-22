"""
Redis管理器模块
提供Redis缓存管理功能
"""

from utils.redis_client import get_redis_client
import json
from typing import Any, Optional

class RedisManager:
    """Redis管理器"""
    
    def __init__(self):
        self.client = get_redis_client()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if not self.client:
            return None
        
        try:
            value = self.client.get(key)
            if value:
                return json.loads(value.decode('utf-8'))
        except Exception as e:
            print(f"Redis get error: {e}")
        return None
    
    def set(self, key: str, value: Any, expire: int = None) -> bool:
        """设置缓存值"""
        if not self.client:
            return False
        
        try:
            serialized_value = json.dumps(value, ensure_ascii=False)
            if expire:
                return self.client.setex(key, expire, serialized_value)
            else:
                return self.client.set(key, serialized_value)
        except Exception as e:
            print(f"Redis set error: {e}")
        return False
    
    def delete(self, key: str) -> bool:
        """删除缓存值"""
        if not self.client:
            return False
        
        try:
            return self.client.delete(key)
        except Exception as e:
            print(f"Redis delete error: {e}")
        return False

# 全局实例
redis_manager = RedisManager()