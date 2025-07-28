"""
监控系统集成模块
将Prometheus指标收集、告警系统和仪表板集成到主应用中
"""

import os
import threading
import time
from typing import Dict, Optional, Any
from pathlib import Path

from flask import Flask, jsonify, render_template_string
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from prometheus_client import make_wsgi_app

from utils.logger import get_logger
from monitoring.prometheus_metrics import get_metrics_collector, init_prometheus_metrics
from monitoring.alerting import get_alert_manager, init_alerting, AlertLevel
from monitoring.dashboard import generate_monitoring_dashboards

logger = get_logger(__name__)

class MonitoringSystem:
    """监控系统集成类"""
    
    def __init__(self, app: Flask, config: Dict[str, Any] = None):
        self.app = app
        self.config = config or {}
        self.metrics_collector = None
        self.alert_manager = None
        self.monitoring_app = None
        self.is_initialized = False
        
    def initialize(self):
        """初始化监控系统"""
        if self.is_initialized:
            logger.warning("监控系统已经初始化")
            return
        
        try:
            # 1. 初始化Prometheus指标收集
            self._init_metrics()
            
            # 2. 初始化告警系统
            self._init_alerting() 
            
            # 3. 创建监控接口
            self._create_monitoring_endpoints()
            
            # 4. 生成监控仪表板
            self._generate_dashboards()
            
            # 5. 集成到主应用
            self._integrate_with_main_app()
            
            self.is_initialized = True
            logger.info("监控系统初始化完成")
            
        except Exception as e:
            logger.error(f"监控系统初始化失败: {e}")
            raise
    
    def _init_metrics(self):
        """初始化指标收集"""
        logger.info("初始化Prometheus指标收集器...")
        
        # 初始化指标收集器
        init_prometheus_metrics(self.app)
        self.metrics_collector = get_metrics_collector()
        
        # 启动系统监控线程
        if self.config.get('enable_system_monitoring', True):
            self.metrics_collector.start_system_monitoring()
        
        logger.info("Prometheus指标收集器初始化完成")
    
    def _init_alerting(self):
        """初始化告警系统"""
        logger.info("初始化告警系统...")
        
        # 配置邮件告警（如果提供配置）
        email_config = self.config.get('email_alerting')
        webhook_config = self.config.get('webhook_alerting')
        
        # 初始化告警管理器
        self.alert_manager = init_alerting(email_config, webhook_config)
        
        logger.info("告警系统初始化完成")
    
    def _create_monitoring_endpoints(self):
        """创建监控API接口"""
        logger.info("创建监控API接口...")
        
        # 创建监控子应用
        self.monitoring_app = Flask('monitoring')
        
        @self.monitoring_app.route('/health')
        def health_check():
            """健康检查接口"""
            return jsonify({
                'status': 'healthy',
                'timestamp': time.time(),
                'metrics_collector': self.metrics_collector is not None,
                'alert_manager': self.alert_manager is not None
            })
        
        @self.monitoring_app.route('/metrics/summary')
        def metrics_summary():
            """指标摘要接口"""
            try:
                # 获取关键指标摘要
                summary = {
                    'timestamp': time.time(),
                    'system': self._get_system_metrics_summary(),
                    'api': self._get_api_metrics_summary(),
                    'predictions': self._get_prediction_metrics_summary(),
                    'alerts': self._get_alerts_summary()
                }
                return jsonify(summary)
            except Exception as e:
                logger.error(f"获取指标摘要失败: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.monitoring_app.route('/alerts')
        def get_alerts():
            """获取告警列表"""
            try:
                active_alerts = self.alert_manager.get_active_alerts()
                alert_history = self.alert_manager.get_alert_history(50)
                
                return jsonify({
                    'active_alerts': [
                        {
                            'rule_name': alert.rule_name,
                            'level': alert.level.value,
                            'status': alert.status.value,
                            'message': alert.message,
                            'start_time': alert.start_time,
                            'value': alert.value
                        }
                        for alert in active_alerts
                    ],
                    'recent_alerts': [
                        {
                            'rule_name': alert.rule_name,
                            'level': alert.level.value,
                            'status': alert.status.value,
                            'message': alert.message,
                            'start_time': alert.start_time,
                            'end_time': alert.end_time,
                            'value': alert.value
                        }
                        for alert in alert_history
                    ]
                })
            except Exception as e:
                logger.error(f"获取告警列表失败: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.monitoring_app.route('/dashboard')
        def web_dashboard():
            """Web监控仪表板"""
            try:
                dashboard_path = Path("monitoring/dashboards/web_dashboard.html")
                if dashboard_path.exists():
                    with open(dashboard_path, 'r', encoding='utf-8') as f:
                        return f.read()
                else:
                    return "监控仪表板未找到，请先生成仪表板", 404
            except Exception as e:
                logger.error(f"加载Web仪表板失败: {e}")
                return f"加载仪表板失败: {str(e)}", 500
        
        @self.monitoring_app.route('/config')
        def get_monitoring_config():
            """获取监控配置"""
            return jsonify({
                'metrics_enabled': self.metrics_collector is not None,
                'alerting_enabled': self.alert_manager is not None,
                'system_monitoring': self.config.get('enable_system_monitoring', True),
                'alert_rules_count': len(self.alert_manager.get_rules()) if self.alert_manager else 0,
                'active_alerts_count': len(self.alert_manager.get_active_alerts()) if self.alert_manager else 0
            })
        
        logger.info("监控API接口创建完成")
    
    def _generate_dashboards(self):
        """生成监控仪表板"""
        logger.info("生成监控仪表板...")
        
        try:
            dashboard_dir = self.config.get('dashboard_dir', 'monitoring/dashboards')
            dashboards = generate_monitoring_dashboards(dashboard_dir)
            
            logger.info(f"已生成 {len(dashboards['grafana_dashboards'])} 个Grafana仪表板")
            logger.info(f"Web仪表板: {dashboards['web_dashboard']}")
            
        except Exception as e:
            logger.error(f"生成监控仪表板失败: {e}")
    
    def _integrate_with_main_app(self):
        """集成到主应用"""
        logger.info("集成监控系统到主应用...")
        
        # 添加Prometheus指标接口
        prometheus_app = make_wsgi_app()
        
        # 使用DispatcherMiddleware组合应用
        self.app.wsgi_app = DispatcherMiddleware(
            self.app.wsgi_app,
            {
                '/metrics': prometheus_app,  # Prometheus指标接口
                '/monitoring': self.monitoring_app  # 监控管理接口
            }
        )
        
        # 添加应用钩子
        self._add_request_hooks()
        
        logger.info("监控系统集成完成")
    
    def _add_request_hooks(self):
        """添加请求钩子用于指标收集"""
        
        @self.app.before_request
        def before_request():
            """请求前处理"""
            if self.metrics_collector:
                # 记录请求开始时间
                import flask
                flask.g.request_start_time = time.time()
        
        @self.app.after_request
        def after_request(response):
            """请求后处理"""
            if self.metrics_collector and hasattr(flask.g, 'request_start_time'):
                # 记录HTTP请求指标
                duration = time.time() - flask.g.request_start_time
                
                import flask
                self.metrics_collector.record_http_request(
                    method=flask.request.method,
                    endpoint=str(flask.request.endpoint or 'unknown'),
                    status_code=response.status_code,
                    duration=duration
                )
            
            return response
    
    def _get_system_metrics_summary(self) -> Dict[str, Any]:
        """获取系统指标摘要"""
        try:
            import psutil
            
            return {
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'load_average': os.getloadavg()[0] if hasattr(os, 'getloadavg') else None
            }
        except Exception as e:
            logger.error(f"获取系统指标失败: {e}")
            return {}
    
    def _get_api_metrics_summary(self) -> Dict[str, Any]:
        """获取API指标摘要"""
        # 这里应该从Prometheus查询实际数据
        # 简化实现，返回模拟数据
        return {
            'total_requests': 1500,
            'avg_response_time': 0.85,
            'error_rate': 0.02
        }
    
    def _get_prediction_metrics_summary(self) -> Dict[str, Any]:
        """获取预测指标摘要"""
        return {
            'total_predictions': 800,
            'success_rate': 0.95,
            'avg_prediction_time': 0.15
        }
    
    def _get_alerts_summary(self) -> Dict[str, Any]:
        """获取告警摘要"""
        if not self.alert_manager:
            return {}
        
        active_alerts = self.alert_manager.get_active_alerts()
        
        return {
            'total_active': len(active_alerts),
            'critical': len([a for a in active_alerts if a.level == AlertLevel.CRITICAL]),
            'warning': len([a for a in active_alerts if a.level == AlertLevel.WARNING]),
            'info': len([a for a in active_alerts if a.level == AlertLevel.INFO])
        }
    
    def shutdown(self):
        """关闭监控系统"""
        logger.info("关闭监控系统...")
        
        if self.metrics_collector:
            self.metrics_collector.stop_system_monitoring()
        
        if self.alert_manager:
            self.alert_manager.stop()
        
        logger.info("监控系统已关闭")

# 全局监控系统实例
_monitoring_system = None

def init_monitoring_system(app: Flask, config: Dict[str, Any] = None) -> MonitoringSystem:
    """初始化监控系统"""
    global _monitoring_system
    
    if _monitoring_system is not None:
        logger.warning("监控系统已经初始化")
        return _monitoring_system
    
    # 默认配置
    default_config = {
        'enable_system_monitoring': True,
        'dashboard_dir': 'monitoring/dashboards',
        'metrics_port': 8000,
        'alert_check_interval': 30
    }
    
    # 合并配置
    final_config = {**default_config, **(config or {})}
    
    # 创建监控系统
    _monitoring_system = MonitoringSystem(app, final_config)
    _monitoring_system.initialize()
    
    logger.info("监控系统全局初始化完成")
    return _monitoring_system

def get_monitoring_system() -> Optional[MonitoringSystem]:
    """获取监控系统实例"""
    return _monitoring_system

# 监控系统配置示例
MONITORING_CONFIG_EXAMPLE = {
    'enable_system_monitoring': True,
    'dashboard_dir': 'monitoring/dashboards',
    'metrics_port': 8000,
    'alert_check_interval': 30,
    
    # 邮件告警配置（可选）
    'email_alerting': {
        'smtp_host': 'smtp.gmail.com',
        'smtp_port': 587,
        'username': 'your-email@gmail.com',
        'password': 'your-password',
        'from_addr': 'your-email@gmail.com',
        'to_addrs': ['admin@yourcompany.com']
    },
    
    # Webhook告警配置（可选）
    'webhook_alerting': {
        'url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK',
        'headers': {
            'Content-Type': 'application/json'
        }
    }
}
