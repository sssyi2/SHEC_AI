"""
数据库性能优化模块
提供连接池、查询优化、索引管理等功能
"""

import mysql.connector
from mysql.connector import pooling, Error
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
from contextlib import contextmanager
from functools import wraps
import logging
from dataclasses import dataclass

from utils.logger import get_logger
from config.settings import get_config

logger = get_logger(__name__)

@dataclass
class QueryPerformanceStats:
    """查询性能统计"""
    query_type: str
    execution_time: float
    rows_affected: int
    query_hash: str
    timestamp: float

class DatabasePerformanceOptimizer:
    """数据库性能优化器"""
    
    def __init__(self, config_name: str = 'development'):
        self.config = get_config(config_name)
        self.connection_pool = None
        self.query_stats = []
        self.slow_query_threshold = 1.0  # 慢查询阈值(秒)
        self.stats_lock = threading.Lock()
        
        # 初始化连接池
        self._initialize_connection_pool()
        
        # 查询缓存
        self.query_cache = {}
        self.cache_lock = threading.Lock()
        
    def _initialize_connection_pool(self):
        """初始化数据库连接池"""
        try:
            pool_config = {
                'pool_name': 'shec_ai_pool',
                'pool_size': 20,  # 连接池大小
                'pool_reset_session': True,
                'host': self.config.MYSQL_HOST,
                'port': self.config.MYSQL_PORT,
                'database': self.config.MYSQL_DATABASE,
                'user': self.config.MYSQL_USER,
                'password': self.config.MYSQL_PASSWORD,
                'charset': 'utf8mb4',
                'use_unicode': True,
                'autocommit': True,
                'time_zone': '+08:00',
                # 性能优化参数
                'connect_timeout': 10,
                'buffered': True,
                'raise_on_warnings': False,
                'sql_mode': 'STRICT_TRANS_TABLES,NO_ZERO_DATE,NO_ZERO_IN_DATE,ERROR_FOR_DIVISION_BY_ZERO'
            }
            
            self.connection_pool = pooling.MySQLConnectionPool(**pool_config)
            logger.info(f"数据库连接池初始化成功 - 池大小: {pool_config['pool_size']}")
            
        except Error as e:
            logger.error(f"数据库连接池初始化失败: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """获取数据库连接"""
        connection = None
        try:
            connection = self.connection_pool.get_connection()
            yield connection
        except Error as e:
            logger.error(f"获取数据库连接失败: {e}")
            if connection:
                connection.rollback()
            raise
        finally:
            if connection and connection.is_connected():
                connection.close()
    
    def query_performance_monitor(self, query_type: str = 'unknown'):
        """查询性能监控装饰器"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    rows_affected = getattr(result, 'rowcount', 0) if hasattr(result, 'rowcount') else len(result) if isinstance(result, list) else 1
                    
                    execution_time = time.time() - start_time
                    
                    # 记录性能统计
                    stats = QueryPerformanceStats(
                        query_type=query_type,
                        execution_time=execution_time,
                        rows_affected=rows_affected,
                        query_hash=hash(str(args) + str(kwargs)),
                        timestamp=time.time()
                    )
                    
                    with self.stats_lock:
                        self.query_stats.append(stats)
                        
                        # 慢查询日志
                        if execution_time > self.slow_query_threshold:
                            logger.warning(f"慢查询检测 - 类型: {query_type}, 耗时: {execution_time:.3f}s")
                    
                    return result
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    logger.error(f"查询执行失败 - 类型: {query_type}, 耗时: {execution_time:.3f}s, 错误: {e}")
                    raise
                    
            return wrapper
        return decorator
    
    @query_performance_monitor('select')
    def execute_select(self, query: str, params: Optional[Tuple] = None, use_cache: bool = True) -> List[Dict]:
        """执行SELECT查询"""
        cache_key = hash(query + str(params)) if use_cache else None
        
        # 检查查询缓存
        if cache_key and use_cache:
            with self.cache_lock:
                if cache_key in self.query_cache:
                    cache_entry = self.query_cache[cache_key]
                    if time.time() - cache_entry['timestamp'] < 300:  # 5分钟缓存
                        logger.debug("使用查询缓存")
                        return cache_entry['data']
        
        with self.get_connection() as connection:
            cursor = connection.cursor(dictionary=True, buffered=True)
            cursor.execute(query, params)
            results = cursor.fetchall()
            cursor.close()
            
            # 更新缓存
            if cache_key and use_cache and len(results) < 1000:  # 只缓存小结果集
                with self.cache_lock:
                    self.query_cache[cache_key] = {
                        'data': results,
                        'timestamp': time.time()
                    }
            
            return results
    
    @query_performance_monitor('insert')
    def execute_insert(self, query: str, params: Optional[Union[Tuple, List[Tuple]]] = None, batch: bool = False) -> int:
        """执行INSERT查询"""
        with self.get_connection() as connection:
            cursor = connection.cursor()
            
            if batch and isinstance(params, list):
                cursor.executemany(query, params)
            else:
                cursor.execute(query, params)
            
            affected_rows = cursor.rowcount
            last_insert_id = cursor.lastrowid
            cursor.close()
            
            return last_insert_id if not batch else affected_rows
    
    @query_performance_monitor('update')
    def execute_update(self, query: str, params: Optional[Tuple] = None) -> int:
        """执行UPDATE查询"""
        with self.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(query, params)
            affected_rows = cursor.rowcount
            cursor.close()
            
            # 清理相关缓存
            self._invalidate_cache()
            
            return affected_rows
    
    @query_performance_monitor('delete')
    def execute_delete(self, query: str, params: Optional[Tuple] = None) -> int:
        """执行DELETE查询"""
        with self.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(query, params)
            affected_rows = cursor.rowcount
            cursor.close()
            
            # 清理相关缓存
            self._invalidate_cache()
            
            return affected_rows
    
    def execute_transaction(self, operations: List[Dict[str, Any]]) -> bool:
        """执行事务"""
        with self.get_connection() as connection:
            cursor = connection.cursor()
            
            try:
                connection.start_transaction()
                
                for operation in operations:
                    query = operation.get('query')
                    params = operation.get('params')
                    cursor.execute(query, params)
                
                connection.commit()
                cursor.close()
                
                # 清理缓存
                self._invalidate_cache()
                
                return True
                
            except Error as e:
                connection.rollback()
                cursor.close()
                logger.error(f"事务执行失败: {e}")
                raise
    
    def _invalidate_cache(self):
        """清理查询缓存"""
        with self.cache_lock:
            self.query_cache.clear()
    
    def optimize_table(self, table_name: str):
        """优化表"""
        query = f"OPTIMIZE TABLE {table_name}"
        with self.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            cursor.close()
            logger.info(f"表 {table_name} 优化完成: {result}")
    
    def analyze_table(self, table_name: str):
        """分析表"""
        query = f"ANALYZE TABLE {table_name}"
        with self.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            cursor.close()
            logger.info(f"表 {table_name} 分析完成: {result}")
    
    def create_index(self, table_name: str, index_name: str, columns: List[str], unique: bool = False):
        """创建索引"""
        index_type = "UNIQUE" if unique else ""
        columns_str = ", ".join(columns)
        query = f"CREATE {index_type} INDEX {index_name} ON {table_name} ({columns_str})"
        
        with self.get_connection() as connection:
            cursor = connection.cursor()
            try:
                cursor.execute(query)
                logger.info(f"索引 {index_name} 创建成功")
            except Error as e:
                if "Duplicate key name" in str(e):
                    logger.info(f"索引 {index_name} 已存在")
                else:
                    logger.error(f"创建索引失败: {e}")
                    raise
            finally:
                cursor.close()
    
    def drop_index(self, table_name: str, index_name: str):
        """删除索引"""
        query = f"DROP INDEX {index_name} ON {table_name}"
        with self.get_connection() as connection:
            cursor = connection.cursor()
            try:
                cursor.execute(query)
                logger.info(f"索引 {index_name} 删除成功")
            except Error as e:
                logger.error(f"删除索引失败: {e}")
                raise
            finally:
                cursor.close()
    
    def get_table_indexes(self, table_name: str) -> List[Dict]:
        """获取表的索引信息"""
        query = f"SHOW INDEX FROM {table_name}"
        return self.execute_select(query, use_cache=False)
    
    def get_query_performance_stats(self) -> Dict[str, Any]:
        """获取查询性能统计"""
        with self.stats_lock:
            stats = self.query_stats.copy()
        
        if not stats:
            return {}
        
        # 统计分析
        total_queries = len(stats)
        total_time = sum(s.execution_time for s in stats)
        avg_time = total_time / total_queries if total_queries > 0 else 0
        
        slow_queries = [s for s in stats if s.execution_time > self.slow_query_threshold]
        slow_query_count = len(slow_queries)
        
        query_types = {}
        for s in stats:
            if s.query_type not in query_types:
                query_types[s.query_type] = {
                    'count': 0,
                    'total_time': 0.0,
                    'avg_time': 0.0
                }
            query_types[s.query_type]['count'] += 1
            query_types[s.query_type]['total_time'] += s.execution_time
        
        for qtype in query_types:
            query_types[qtype]['avg_time'] = (
                query_types[qtype]['total_time'] / query_types[qtype]['count']
            )
        
        return {
            'total_queries': total_queries,
            'total_execution_time': total_time,
            'average_execution_time': avg_time,
            'slow_queries': slow_query_count,
            'slow_query_rate': slow_query_count / total_queries if total_queries > 0 else 0,
            'query_types': query_types,
            'cache_size': len(self.query_cache)
        }
    
    def reset_performance_stats(self):
        """重置性能统计"""
        with self.stats_lock:
            self.query_stats.clear()
    
    def setup_performance_indexes(self):
        """设置性能优化索引"""
        logger.info("开始设置数据库性能索引")
        
        # 常用索引配置
        indexes_config = [
            {
                'table': 'users',
                'indexes': [
                    {'name': 'idx_users_username', 'columns': ['username'], 'unique': True},
                    {'name': 'idx_users_created_at', 'columns': ['created_at']},
                ]
            },
            {
                'table': 'health_records',
                'indexes': [
                    {'name': 'idx_health_user_id', 'columns': ['user_id']},
                    {'name': 'idx_health_record_date', 'columns': ['record_date']},
                    {'name': 'idx_health_user_date', 'columns': ['user_id', 'record_date']},
                ]
            },
            {
                'table': 'predictions',
                'indexes': [
                    {'name': 'idx_pred_user_id', 'columns': ['user_id']},
                    {'name': 'idx_pred_created_at', 'columns': ['created_at']},
                    {'name': 'idx_pred_model_name', 'columns': ['model_name']},
                ]
            }
        ]
        
        for table_config in indexes_config:
            table_name = table_config['table']
            for index_config in table_config['indexes']:
                try:
                    self.create_index(
                        table_name=table_name,
                        index_name=index_config['name'],
                        columns=index_config['columns'],
                        unique=index_config.get('unique', False)
                    )
                except Exception as e:
                    logger.warning(f"创建索引 {index_config['name']} 失败: {e}")
        
        logger.info("数据库性能索引设置完成")

# 单例实例
_db_optimizer = None
_optimizer_lock = threading.Lock()

def get_db_optimizer(config_name: str = 'development') -> DatabasePerformanceOptimizer:
    """获取数据库优化器单例"""
    global _db_optimizer
    
    if _db_optimizer is None:
        with _optimizer_lock:
            if _db_optimizer is None:
                _db_optimizer = DatabasePerformanceOptimizer(config_name)
    
    return _db_optimizer
