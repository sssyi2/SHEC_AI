"""
Prometheus指标收集模块
提供应用性能和业务指标的收集功能
"""

import time
from typing import Dict, Any, Optional, List
from flask import Flask, request, g
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CollectorRegistry
from prometheus_client.metrics import MetricWrapperBase
import threading
from functools import wraps
from dataclasses import dataclass
import psutil
import GPUtil

from utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class MetricConfig:
    """指标配置"""
    name: str
    description: str
    labels: List[str] = None
    buckets: List[float] = None

class PrometheusMetricsCollector:
    """Prometheus指标收集器"""
    
    def __init__(self, app: Optional[Flask] = None):
        self.app = app
        self.registry = CollectorRegistry()
        
        # API请求指标
        self.request_count = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )
        
        # 预测相关指标
        self.prediction_count = Counter(
            'predictions_total',
            'Total number of predictions made',
            ['model_name', 'prediction_type', 'status'],
            registry=self.registry
        )
        
        self.prediction_duration = Histogram(
            'prediction_duration_seconds',
            'Time spent on predictions',
            ['model_name', 'prediction_type'],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
            registry=self.registry
        )
        
        # 模型性能指标
        self.model_accuracy = Gauge(
            'model_accuracy_score',
            'Current model accuracy score',
            ['model_name'],
            registry=self.registry
        )
        
        self.model_inference_time = Gauge(
            'model_inference_time_seconds',
            'Average model inference time',
            ['model_name'],
            registry=self.registry
        )
        
        # 缓存指标
        self.cache_hits = Counter(
            'cache_hits_total',
            'Total cache hits',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'cache_misses_total',
            'Total cache misses',
            ['cache_type'],
            registry=self.registry
        )
        
        # 数据库指标
        self.db_connections_active = Gauge(
            'database_connections_active',
            'Active database connections',
            registry=self.registry
        )
        
        self.db_query_duration = Histogram(
            'database_query_duration_seconds',
            'Database query execution time',
            ['query_type'],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
            registry=self.registry
        )
        
        # 系统资源指标
        self.system_cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.system_memory_usage = Gauge(
            'system_memory_usage_bytes',
            'System memory usage in bytes',
            registry=self.registry
        )
        
        self.system_disk_usage = Gauge(
            'system_disk_usage_percent',
            'System disk usage percentage',
            ['device'],
            registry=self.registry
        )
        
        # GPU指标（如果可用）
        self.gpu_usage = Gauge(
            'gpu_usage_percent',
            'GPU usage percentage',
            ['gpu_id'],
            registry=self.registry
        )
        
        self.gpu_memory_usage = Gauge(
            'gpu_memory_usage_bytes',
            'GPU memory usage in bytes',
            ['gpu_id'],
            registry=self.registry
        )
        
        # 应用信息
        self.app_info = Info(
            'shec_ai_info',
            'SHEC AI application information',
            registry=self.registry
        )
        
        if app:
            self.init_app(app)
    
    def init_app(self, app: Flask):
        """初始化Flask应用"""
        self.app = app
        
        # 设置应用信息
        self.app_info.info({
            'version': '4.1.0',
            'environment': app.config.get('ENV', 'development'),
            'debug': str(app.debug)
        })
        
        # 注册请求监控
        app.before_request(self._before_request)
        app.after_request(self._after_request)
        
        # 添加指标端点
        @app.route('/metrics')
        def metrics():
            return generate_latest(self.registry), 200, {'Content-Type': 'text/plain'}
        
        logger.info("Prometheus指标收集器初始化完成")
    
    def _before_request(self):
        """请求开始前的处理"""
        g.start_time = time.time()
    
    def _after_request(self, response):
        """请求结束后的处理"""
        if hasattr(g, 'start_time'):
            duration = time.time() - g.start_time
            
            # 记录请求指标
            method = request.method
            endpoint = request.endpoint or 'unknown'
            status = str(response.status_code)
            
            self.request_count.labels(method=method, endpoint=endpoint, status=status).inc()
            self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
        
        return response
    
    def record_prediction(self, model_name: str, prediction_type: str, 
                         duration: float, success: bool = True):
        """记录预测指标"""
        status = 'success' if success else 'error'
        
        self.prediction_count.labels(
            model_name=model_name,
            prediction_type=prediction_type,
            status=status
        ).inc()
        
        if success:
            self.prediction_duration.labels(
                model_name=model_name,
                prediction_type=prediction_type
            ).observe(duration)
    
    def update_model_metrics(self, model_name: str, accuracy: float, 
                           inference_time: float):
        """更新模型性能指标"""
        self.model_accuracy.labels(model_name=model_name).set(accuracy)
        self.model_inference_time.labels(model_name=model_name).set(inference_time)
    
    def record_http_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """记录HTTP请求指标"""
        # 记录请求总数
        self.request_count.labels(
            method=method, 
            endpoint=endpoint, 
            status=str(status_code)
        ).inc()
        
        # 记录请求响应时间
        self.request_duration.labels(
            method=method, 
            endpoint=endpoint
        ).observe(duration)
    
    def record_cache_operation(self, operation_type: str, cache_type: str):
        """记录缓存操作"""
        if operation_type == "hit":
            self.record_cache_hit(cache_type)
        elif operation_type == "miss":
            self.record_cache_miss(cache_type)
    
    def record_cache_hit(self, cache_type: str):
        """记录缓存命中"""
        self.cache_hits.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str):
        """记录缓存未命中"""
        self.cache_misses.labels(cache_type=cache_type).inc()
    
    def record_db_query(self, query_type: str, duration: float):
        """记录数据库查询"""
        self.db_query_duration.labels(query_type=query_type).observe(duration)
    
    def update_db_connections(self, active_count: int):
        """更新数据库连接数"""
        self.db_connections_active.set(active_count)
    
    def update_system_metrics(self):
        """更新系统资源指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            self.system_cpu_usage.set(cpu_percent)
            
            # 内存使用
            memory = psutil.virtual_memory()
            self.system_memory_usage.set(memory.used)
            
            # 磁盘使用
            disk_usage = psutil.disk_usage('/')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            self.system_disk_usage.labels(device='/').set(disk_percent)
            
            # GPU指标（如果可用）
            try:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    self.gpu_usage.labels(gpu_id=str(i)).set(gpu.load * 100)
                    self.gpu_memory_usage.labels(gpu_id=str(i)).set(
                        gpu.memoryUsed * 1024 * 1024  # 转换为字节
                    )
            except Exception:
                # GPU不可用时忽略
                pass
                
        except Exception as e:
            logger.error(f"更新系统指标失败: {e}")
    
    def monitoring_decorator(self, model_name: str = None, prediction_type: str = None):
        """监控装饰器"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                success = True
                
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    success = False
                    raise
                finally:
                    duration = time.time() - start_time
                    
                    if model_name and prediction_type:
                        self.record_prediction(model_name, prediction_type, duration, success)
                    
            return wrapper
        return decorator

# 系统监控线程
class SystemMonitorThread(threading.Thread):
    """系统监控线程"""
    
    def __init__(self, metrics_collector: PrometheusMetricsCollector, 
                 interval: int = 30):
        super().__init__(daemon=True)
        self.metrics_collector = metrics_collector
        self.interval = interval
        self.running = True
        
    def run(self):
        """运行系统监控"""
        logger.info(f"系统监控线程启动，间隔: {self.interval}秒")
        
        while self.running:
            try:
                self.metrics_collector.update_system_metrics()
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"系统监控线程错误: {e}")
                time.sleep(self.interval)
    
    def stop(self):
        """停止监控"""
        self.running = False

# 全局指标收集器实例
_metrics_collector = None
_monitor_thread = None

def get_metrics_collector() -> PrometheusMetricsCollector:
    """获取全局指标收集器"""
    global _metrics_collector
    
    if _metrics_collector is None:
        _metrics_collector = PrometheusMetricsCollector()
    
    return _metrics_collector

def init_prometheus_metrics(app: Flask, start_system_monitor: bool = True):
    """初始化Prometheus指标收集系统"""
    global _metrics_collector, _monitor_thread
    
    _metrics_collector = PrometheusMetricsCollector(app)
    
    if start_system_monitor:
        _monitor_thread = SystemMonitorThread(_metrics_collector, interval=30)
        _monitor_thread.start()
    
    logger.info("Prometheus指标收集系统初始化完成")
    return _metrics_collector

def init_monitoring(app: Flask, start_system_monitor: bool = True):
    """初始化监控系统（兼容性别名）"""
    return init_prometheus_metrics(app, start_system_monitor)
    
    logger.info("监控系统初始化完成")
    
    return _metrics_collector

def stop_monitoring():
    """停止监控系统"""
    global _monitor_thread
    
    if _monitor_thread:
        _monitor_thread.stop()
        _monitor_thread = None
    
    logger.info("监控系统已停止")
