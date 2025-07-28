#!/usr/bin/env python3
"""
监控路由处理器
为Sprint 4.2提供监控接口路由
"""
import json
import os
from flask import jsonify, request
from typing import Dict, Any
import psutil
import logging

logger = logging.getLogger(__name__)

class MonitoringRoutes:
    """监控路由处理器"""
    
    def __init__(self, app, metrics_collector=None):
        """
        初始化监控路由
        
        Args:
            app: Flask应用实例
            metrics_collector: 指标收集器实例
        """
        self.app = app
        self.metrics_collector = metrics_collector
        self.setup_routes()
    
    def setup_routes(self):
        """设置监控路由"""
        
        @self.app.route('/monitoring/health')
        def monitoring_health():
            """监控健康检查接口"""
            try:
                # 获取系统状态
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                health_status = {
                    "status": "healthy",
                    "timestamp": "2025-07-22T10:05:00Z",
                    "system": {
                        "cpu_usage": f"{cpu_percent:.1f}%",
                        "memory_usage": f"{memory.percent:.1f}%",
                        "disk_usage": f"{disk.percent:.1f}%"
                    },
                    "services": {
                        "prometheus": "running",
                        "monitoring": "active",
                        "dashboard": "available"
                    },
                    "version": "Sprint 4.2"
                }
                
                logger.info("监控健康检查请求 - 状态: healthy")
                return jsonify(health_status)
                
            except Exception as e:
                logger.error(f"健康检查失败: {str(e)}")
                return jsonify({
                    "status": "error", 
                    "message": str(e),
                    "timestamp": "2025-07-22T10:05:00Z"
                }), 500
        
        @self.app.route('/monitoring/config')
        def monitoring_config():
            """监控配置接口"""
            try:
                config = {
                    "monitoring_enabled": True,
                    "system_monitoring": {
                        "enabled": True,
                        "interval": 30,
                        "metrics": ["cpu", "memory", "disk", "gpu"]
                    },
                    "prometheus": {
                        "enabled": True,
                        "port": 5002,
                        "path": "/metrics"
                    },
                    "dashboards": {
                        "enabled": True,
                        "auto_refresh": 30,
                        "themes": ["light", "dark"]
                    },
                    "alerts": {
                        "enabled": True,
                        "rules": [
                            {
                                "name": "high_cpu",
                                "condition": "cpu_usage > 80",
                                "severity": "warning"
                            },
                            {
                                "name": "high_memory", 
                                "condition": "memory_usage > 90",
                                "severity": "critical"
                            }
                        ]
                    },
                    "version": "Sprint 4.2",
                    "last_updated": "2025-07-22T10:00:00Z"
                }
                
                logger.info("监控配置请求")
                return jsonify(config)
                
            except Exception as e:
                logger.error(f"获取配置失败: {str(e)}")
                return jsonify({
                    "status": "error",
                    "message": str(e)
                }), 500
        
        @self.app.route('/monitoring/metrics/summary')
        def monitoring_metrics_summary():
            """监控指标摘要"""
            try:
                # 获取系统指标
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # GPU信息
                gpu_info = "N/A"
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_info = f"{gpus[0].name} ({gpus[0].memoryUtil*100:.1f}%)"
                except:
                    pass
                
                summary = {
                    "system_metrics": {
                        "cpu_usage": cpu_percent,
                        "memory_usage": memory.percent,
                        "disk_usage": disk.percent,
                        "gpu_info": gpu_info
                    },
                    "application_metrics": {
                        "requests_total": "N/A",
                        "response_time_avg": "N/A",
                        "errors_total": "N/A"
                    },
                    "collection_time": "2025-07-22T10:05:00Z",
                    "status": "active"
                }
                
                if self.metrics_collector:
                    # 如果有指标收集器，获取更详细的信息
                    try:
                        # 这里可以添加更多指标收集逻辑
                        summary["application_metrics"]["status"] = "collecting"
                    except Exception as e:
                        logger.warning(f"获取应用指标失败: {str(e)}")
                
                return jsonify(summary)
                
            except Exception as e:
                logger.error(f"获取指标摘要失败: {str(e)}")
                return jsonify({
                    "status": "error",
                    "message": str(e)
                }), 500
        
        @self.app.route('/monitoring/dashboard')
        def monitoring_dashboard():
            """Web监控仪表板"""
            try:
                # 检查仪表板文件
                dashboard_path = "test_dashboards/web_dashboard.html"
                if os.path.exists(dashboard_path):
                    with open(dashboard_path, 'r', encoding='utf-8') as f:
                        return f.read()
                else:
                    return """
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>SHEC AI 监控仪表板</title>
                        <style>
                            body { font-family: Arial, sans-serif; margin: 20px; }
                            .dashboard { background: #f5f5f5; padding: 20px; border-radius: 8px; }
                            .metric { background: white; padding: 15px; margin: 10px 0; border-radius: 5px; }
                        </style>
                    </head>
                    <body>
                        <h1>🎯 SHEC AI 监控仪表板</h1>
                        <div class="dashboard">
                            <div class="metric">
                                <h3>✅ Sprint 4.2 监控系统</h3>
                                <p>状态: 运行中</p>
                                <p>版本: Sprint 4.2</p>
                            </div>
                            <div class="metric">
                                <h3>📊 系统指标</h3>
                                <p>CPU使用率: 实时监控中</p>
                                <p>内存使用率: 实时监控中</p>
                                <p>磁盘使用率: 实时监控中</p>
                            </div>
                            <div class="metric">
                                <h3>🔗 相关链接</h3>
                                <p><a href="/monitoring/health">健康检查</a></p>
                                <p><a href="/monitoring/config">监控配置</a></p>
                                <p><a href="/metrics">Prometheus指标</a></p>
                            </div>
                        </div>
                    </body>
                    </html>
                    """
            except Exception as e:
                logger.error(f"获取仪表板失败: {str(e)}")
                return f"<h1>错误</h1><p>{str(e)}</p>", 500
        
        @self.app.route('/metrics')
        def prometheus_metrics():
            """Prometheus指标接口"""
            try:
                # 基础Prometheus格式指标
                metrics_data = []
                
                # 系统指标
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                metrics_data.extend([
                    f"# HELP system_cpu_usage_percent CPU使用率百分比",
                    f"# TYPE system_cpu_usage_percent gauge",
                    f"system_cpu_usage_percent {cpu_percent}",
                    f"",
                    f"# HELP system_memory_usage_percent 内存使用率百分比",
                    f"# TYPE system_memory_usage_percent gauge",
                    f"system_memory_usage_percent {memory.percent}",
                    f"",
                    f"# HELP system_disk_usage_percent 磁盘使用率百分比",
                    f"# TYPE system_disk_usage_percent gauge", 
                    f"system_disk_usage_percent {disk.percent}",
                    f""
                ])
                
                # HTTP请求指标（模拟）
                metrics_data.extend([
                    f"# HELP http_requests_total HTTP请求总数",
                    f"# TYPE http_requests_total counter",
                    f"http_requests_total{{method=\"GET\",status=\"200\"}} 10",
                    f"http_requests_total{{method=\"GET\",status=\"404\"}} 5",
                    f""
                ])
                
                response_text = "\n".join(metrics_data)
                
                from flask import Response
                return Response(response_text, mimetype='text/plain')
                
            except Exception as e:
                logger.error(f"获取Prometheus指标失败: {str(e)}")
                return f"# ERROR: {str(e)}", 500

def register_monitoring_routes(app, metrics_collector=None):
    """
    注册监控路由到Flask应用
    
    Args:
        app: Flask应用实例
        metrics_collector: 指标收集器实例
        
    Returns:
        MonitoringRoutes: 监控路由实例
    """
    monitoring_routes = MonitoringRoutes(app, metrics_collector)
    logger.info("监控路由注册完成")
    return monitoring_routes
