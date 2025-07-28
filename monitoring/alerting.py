"""
告警规则配置和管理模块
提供自定义告警规则和通知机制
"""

import time
import json
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import smtplib
import email.mime.text
import email.mime.multipart
import requests

from utils.logger import get_logger
from monitoring.prometheus_metrics import get_metrics_collector

logger = get_logger(__name__)

class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class AlertStatus(Enum):
    """告警状态"""
    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"

@dataclass
class AlertRule:
    """告警规则"""
    name: str
    description: str
    metric_name: str
    threshold: float
    comparison: str  # '>', '<', '>=', '<=', '==', '!='
    duration: int  # 持续时间(秒)
    level: AlertLevel
    labels: Dict[str, str] = None
    annotations: Dict[str, str] = None
    enabled: bool = True

@dataclass
class Alert:
    """告警实例"""
    rule_name: str
    level: AlertLevel
    status: AlertStatus
    message: str
    start_time: float
    end_time: Optional[float] = None
    labels: Dict[str, str] = None
    annotations: Dict[str, str] = None
    value: Optional[float] = None

class AlertNotifier:
    """告警通知器基类"""
    
    def send(self, alert: Alert) -> bool:
        """发送告警通知"""
        raise NotImplementedError

class EmailNotifier(AlertNotifier):
    """邮件告警通知器"""
    
    def __init__(self, smtp_host: str, smtp_port: int, username: str, 
                 password: str, from_addr: str, to_addrs: List[str]):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs
    
    def send(self, alert: Alert) -> bool:
        """发送邮件告警"""
        try:
            msg = email.mime.multipart.MIMEMultipart()
            msg['From'] = self.from_addr
            msg['To'] = ', '.join(self.to_addrs)
            msg['Subject'] = f"[{alert.level.value.upper()}] {alert.rule_name}"
            
            body = self._format_alert_message(alert)
            msg.attach(email.mime.text.MimeText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"邮件告警发送成功: {alert.rule_name}")
            return True
            
        except Exception as e:
            logger.error(f"邮件告警发送失败: {e}")
            return False
    
    def _format_alert_message(self, alert: Alert) -> str:
        """格式化告警消息"""
        message = f"""
告警名称: {alert.rule_name}
告警级别: {alert.level.value.upper()}
告警状态: {alert.status.value.upper()}
告警时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.start_time))}
告警消息: {alert.message}
"""
        
        if alert.value is not None:
            message += f"当前值: {alert.value}\n"
        
        if alert.labels:
            message += f"标签: {json.dumps(alert.labels, indent=2)}\n"
        
        if alert.annotations:
            message += f"注释: {json.dumps(alert.annotations, indent=2)}\n"
        
        return message

class WebhookNotifier(AlertNotifier):
    """Webhook告警通知器"""
    
    def __init__(self, url: str, headers: Dict[str, str] = None):
        self.url = url
        self.headers = headers or {}
    
    def send(self, alert: Alert) -> bool:
        """发送Webhook告警"""
        try:
            payload = {
                'rule_name': alert.rule_name,
                'level': alert.level.value,
                'status': alert.status.value,
                'message': alert.message,
                'start_time': alert.start_time,
                'end_time': alert.end_time,
                'labels': alert.labels or {},
                'annotations': alert.annotations or {},
                'value': alert.value
            }
            
            response = requests.post(
                self.url, 
                json=payload, 
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            
            logger.info(f"Webhook告警发送成功: {alert.rule_name}")
            return True
            
        except Exception as e:
            logger.error(f"Webhook告警发送失败: {e}")
            return False

class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notifiers: List[AlertNotifier] = []
        self.running = False
        self.check_interval = 30  # 检查间隔(秒)
        self._thread = None
        
        # 预定义告警规则
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """设置默认告警规则"""
        default_rules = [
            # API响应时间告警
            AlertRule(
                name="high_api_response_time",
                description="API响应时间过高",
                metric_name="http_request_duration_seconds",
                threshold=2.0,
                comparison=">",
                duration=60,
                level=AlertLevel.WARNING,
                annotations={
                    "summary": "API响应时间超过2秒",
                    "description": "API响应时间过高，可能影响用户体验"
                }
            ),
            
            # 预测错误率告警
            AlertRule(
                name="high_prediction_error_rate",
                description="预测错误率过高",
                metric_name="predictions_total",
                threshold=0.1,  # 10%错误率
                comparison=">",
                duration=300,
                level=AlertLevel.CRITICAL,
                annotations={
                    "summary": "预测错误率过高",
                    "description": "预测服务错误率超过10%，需要立即检查"
                }
            ),
            
            # 系统CPU使用率告警
            AlertRule(
                name="high_cpu_usage",
                description="CPU使用率过高",
                metric_name="system_cpu_usage_percent",
                threshold=85.0,
                comparison=">",
                duration=180,
                level=AlertLevel.WARNING,
                annotations={
                    "summary": "系统CPU使用率过高",
                    "description": "CPU使用率超过85%，系统负载较高"
                }
            ),
            
            # 内存使用告警
            AlertRule(
                name="high_memory_usage",
                description="内存使用率过高",
                metric_name="system_memory_usage_bytes",
                threshold=0.9,  # 90%
                comparison=">",
                duration=120,
                level=AlertLevel.CRITICAL,
                annotations={
                    "summary": "系统内存使用率过高",
                    "description": "内存使用率超过90%，可能导致系统不稳定"
                }
            ),
            
            # 数据库连接数告警
            AlertRule(
                name="high_db_connections",
                description="数据库连接数过多",
                metric_name="database_connections_active",
                threshold=18,  # 连接池大小的90%
                comparison=">",
                duration=60,
                level=AlertLevel.WARNING,
                annotations={
                    "summary": "数据库连接数过多",
                    "description": "活跃数据库连接数接近连接池上限"
                }
            ),
            
            # GPU使用率告警（如果有GPU）
            AlertRule(
                name="high_gpu_usage",
                description="GPU使用率过高",
                metric_name="gpu_usage_percent",
                threshold=95.0,
                comparison=">",
                duration=300,
                level=AlertLevel.WARNING,
                annotations={
                    "summary": "GPU使用率过高",
                    "description": "GPU使用率超过95%，可能存在资源竞争"
                }
            )
        ]
        
        for rule in default_rules:
            self.add_rule(rule)
        
        logger.info(f"已加载 {len(default_rules)} 个默认告警规则")
    
    def add_rule(self, rule: AlertRule):
        """添加告警规则"""
        self.rules[rule.name] = rule
        logger.info(f"添加告警规则: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """删除告警规则"""
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info(f"删除告警规则: {rule_name}")
    
    def enable_rule(self, rule_name: str):
        """启用告警规则"""
        if rule_name in self.rules:
            self.rules[rule_name].enabled = True
            logger.info(f"启用告警规则: {rule_name}")
    
    def disable_rule(self, rule_name: str):
        """禁用告警规则"""
        if rule_name in self.rules:
            self.rules[rule_name].enabled = False
            logger.info(f"禁用告警规则: {rule_name}")
    
    def add_notifier(self, notifier: AlertNotifier):
        """添加告警通知器"""
        self.notifiers.append(notifier)
        logger.info(f"添加告警通知器: {type(notifier).__name__}")
    
    def start(self):
        """启动告警监控"""
        if self.running:
            return
        
        self.running = True
        self._thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._thread.start()
        
        logger.info("告警监控已启动")
    
    def stop(self):
        """停止告警监控"""
        self.running = False
        if self._thread:
            self._thread.join()
        
        logger.info("告警监控已停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        metrics_collector = get_metrics_collector()
        
        while self.running:
            try:
                for rule_name, rule in self.rules.items():
                    if not rule.enabled:
                        continue
                    
                    self._check_rule(rule, metrics_collector)
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"告警监控循环错误: {e}")
                time.sleep(self.check_interval)
    
    def _check_rule(self, rule: AlertRule, metrics_collector):
        """检查告警规则"""
        try:
            # 这里应该从Prometheus获取指标值
            # 简化实现，使用模拟数据
            current_value = self._get_metric_value(rule.metric_name, metrics_collector)
            
            if current_value is None:
                return
            
            # 评估告警条件
            is_firing = self._evaluate_condition(current_value, rule.threshold, rule.comparison)
            
            if is_firing:
                self._handle_firing_alert(rule, current_value)
            else:
                self._handle_resolved_alert(rule)
                
        except Exception as e:
            logger.error(f"检查告警规则失败 {rule.name}: {e}")
    
    def _get_metric_value(self, metric_name: str, metrics_collector) -> Optional[float]:
        """获取指标值（简化实现）"""
        # 这里应该从Prometheus查询API获取实际值
        # 为了演示，返回一些模拟值
        if metric_name == "http_request_duration_seconds":
            return 1.5  # 模拟响应时间
        elif metric_name == "system_cpu_usage_percent":
            import psutil
            return psutil.cpu_percent()
        elif metric_name == "system_memory_usage_bytes":
            import psutil
            memory = psutil.virtual_memory()
            return memory.percent / 100.0
        elif metric_name == "database_connections_active":
            return 10  # 模拟连接数
        
        return None
    
    def _evaluate_condition(self, value: float, threshold: float, comparison: str) -> bool:
        """评估告警条件"""
        if comparison == '>':
            return value > threshold
        elif comparison == '<':
            return value < threshold
        elif comparison == '>=':
            return value >= threshold
        elif comparison == '<=':
            return value <= threshold
        elif comparison == '==':
            return value == threshold
        elif comparison == '!=':
            return value != threshold
        
        return False
    
    def _handle_firing_alert(self, rule: AlertRule, value: float):
        """处理告警触发"""
        alert_key = f"{rule.name}"
        
        if alert_key in self.active_alerts:
            # 更新现有告警
            alert = self.active_alerts[alert_key]
            alert.value = value
        else:
            # 创建新告警
            alert = Alert(
                rule_name=rule.name,
                level=rule.level,
                status=AlertStatus.FIRING,
                message=f"{rule.description}: 当前值 {value}, 阈值 {rule.threshold}",
                start_time=time.time(),
                labels=rule.labels or {},
                annotations=rule.annotations or {},
                value=value
            )
            
            self.active_alerts[alert_key] = alert
            self.alert_history.append(alert)
            
            # 发送告警通知
            self._send_notifications(alert)
            
            logger.warning(f"告警触发: {rule.name} - {alert.message}")
    
    def _handle_resolved_alert(self, rule: AlertRule):
        """处理告警解决"""
        alert_key = f"{rule.name}"
        
        if alert_key in self.active_alerts:
            alert = self.active_alerts[alert_key]
            alert.status = AlertStatus.RESOLVED
            alert.end_time = time.time()
            
            # 发送解决通知
            self._send_notifications(alert)
            
            # 从活跃告警中移除
            del self.active_alerts[alert_key]
            
            logger.info(f"告警解决: {rule.name}")
    
    def _send_notifications(self, alert: Alert):
        """发送告警通知"""
        for notifier in self.notifiers:
            try:
                notifier.send(alert)
            except Exception as e:
                logger.error(f"发送告警通知失败 {type(notifier).__name__}: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警列表"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """获取告警历史"""
        return self.alert_history[-limit:]
    
    def get_rules(self) -> Dict[str, AlertRule]:
        """获取所有告警规则"""
        return self.rules.copy()

# 全局告警管理器实例
_alert_manager = None

def get_alert_manager() -> AlertManager:
    """获取全局告警管理器"""
    global _alert_manager
    
    if _alert_manager is None:
        _alert_manager = AlertManager()
    
    return _alert_manager

def init_alerting(email_config: Dict = None, webhook_config: Dict = None):
    """初始化告警系统"""
    alert_manager = get_alert_manager()
    
    # 配置邮件通知器
    if email_config:
        email_notifier = EmailNotifier(**email_config)
        alert_manager.add_notifier(email_notifier)
    
    # 配置Webhook通知器
    if webhook_config:
        webhook_notifier = WebhookNotifier(**webhook_config)
        alert_manager.add_notifier(webhook_notifier)
    
    # 启动告警监控
    alert_manager.start()
    
    logger.info("告警系统初始化完成")
    
    return alert_manager
