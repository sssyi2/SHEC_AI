"""
监控仪表板生成器
为Grafana和Web界面生成监控仪表板配置
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class Panel:
    """仪表板面板配置"""
    id: int
    title: str
    type: str
    datasource: str
    targets: List[Dict[str, Any]]
    x: int = 0
    y: int = 0
    width: int = 12
    height: int = 8
    options: Dict[str, Any] = None
    field_config: Dict[str, Any] = None

@dataclass
class Dashboard:
    """仪表板配置"""
    id: Optional[int]
    title: str
    description: str
    tags: List[str]
    panels: List[Panel]
    refresh: str = "30s"
    time_from: str = "now-1h"
    time_to: str = "now"

class GrafanaDashboardGenerator:
    """Grafana仪表板生成器"""
    
    def __init__(self, datasource_name: str = "Prometheus"):
        self.datasource_name = datasource_name
        
    def create_system_overview_dashboard(self) -> Dashboard:
        """创建系统概览仪表板"""
        panels = []
        panel_id = 1
        
        # API请求总数
        panels.append(Panel(
            id=panel_id,
            title="API请求总数",
            type="stat",
            datasource=self.datasource_name,
            targets=[{
                "expr": "sum(increase(http_requests_total[5m]))",
                "format": "time_series",
                "legendFormat": "总请求数"
            }],
            x=0, y=0, width=6, height=4,
            field_config={
                "defaults": {
                    "color": {"mode": "value"},
                    "mappings": [],
                    "thresholds": {
                        "steps": [
                            {"color": "green", "value": None},
                            {"color": "red", "value": 1000}
                        ]
                    }
                }
            }
        ))
        panel_id += 1
        
        # API响应时间
        panels.append(Panel(
            id=panel_id,
            title="平均响应时间",
            type="stat", 
            datasource=self.datasource_name,
            targets=[{
                "expr": "avg(http_request_duration_seconds)",
                "format": "time_series",
                "legendFormat": "平均响应时间"
            }],
            x=6, y=0, width=6, height=4,
            field_config={
                "defaults": {
                    "unit": "s",
                    "color": {"mode": "value"},
                    "thresholds": {
                        "steps": [
                            {"color": "green", "value": None},
                            {"color": "yellow", "value": 1},
                            {"color": "red", "value": 2}
                        ]
                    }
                }
            }
        ))
        panel_id += 1
        
        # API请求率时间序列
        panels.append(Panel(
            id=panel_id,
            title="API请求率",
            type="graph",
            datasource=self.datasource_name,
            targets=[{
                "expr": "sum(rate(http_requests_total[5m])) by (method, endpoint)",
                "format": "time_series",
                "legendFormat": "{{method}} {{endpoint}}"
            }],
            x=0, y=4, width=12, height=8,
            options={
                "legend": {"displayMode": "table", "placement": "right"},
                "tooltip": {"mode": "multi"}
            }
        ))
        panel_id += 1
        
        # HTTP状态码分布
        panels.append(Panel(
            id=panel_id,
            title="HTTP状态码分布",
            type="piechart",
            datasource=self.datasource_name,
            targets=[{
                "expr": "sum(increase(http_requests_total[5m])) by (status)",
                "format": "time_series", 
                "legendFormat": "{{status}}"
            }],
            x=0, y=12, width=6, height=8
        ))
        panel_id += 1
        
        # 预测服务指标
        panels.append(Panel(
            id=panel_id,
            title="预测服务统计",
            type="table",
            datasource=self.datasource_name,
            targets=[
                {
                    "expr": "sum(increase(predictions_total[5m]))",
                    "format": "table",
                    "legendFormat": "总预测数"
                },
                {
                    "expr": "sum(increase(predictions_total{status=\"success\"}[5m]))",
                    "format": "table", 
                    "legendFormat": "成功预测数"
                },
                {
                    "expr": "avg(prediction_duration_seconds)",
                    "format": "table",
                    "legendFormat": "平均预测时间"
                }
            ],
            x=6, y=12, width=6, height=8
        ))
        panel_id += 1
        
        return Dashboard(
            id=None,
            title="SHEC AI - 系统概览",
            description="SHEC AI系统监控概览仪表板",
            tags=["shec-ai", "overview", "api"],
            panels=panels
        )
    
    def create_performance_dashboard(self) -> Dashboard:
        """创建性能监控仪表板"""
        panels = []
        panel_id = 1
        
        # CPU使用率
        panels.append(Panel(
            id=panel_id,
            title="CPU使用率",
            type="graph",
            datasource=self.datasource_name,
            targets=[{
                "expr": "system_cpu_usage_percent",
                "format": "time_series",
                "legendFormat": "CPU使用率"
            }],
            x=0, y=0, width=6, height=8,
            field_config={
                "defaults": {
                    "unit": "percent",
                    "min": 0,
                    "max": 100
                }
            }
        ))
        panel_id += 1
        
        # 内存使用率
        panels.append(Panel(
            id=panel_id,
            title="内存使用率",
            type="graph",
            datasource=self.datasource_name,
            targets=[{
                "expr": "system_memory_usage_bytes / system_memory_total_bytes * 100",
                "format": "time_series",
                "legendFormat": "内存使用率"
            }],
            x=6, y=0, width=6, height=8,
            field_config={
                "defaults": {
                    "unit": "percent",
                    "min": 0,
                    "max": 100
                }
            }
        ))
        panel_id += 1
        
        # 磁盘使用率
        panels.append(Panel(
            id=panel_id,
            title="磁盘使用率", 
            type="graph",
            datasource=self.datasource_name,
            targets=[{
                "expr": "system_disk_usage_bytes / system_disk_total_bytes * 100",
                "format": "time_series",
                "legendFormat": "磁盘使用率"
            }],
            x=0, y=8, width=6, height=8,
            field_config={
                "defaults": {
                    "unit": "percent",
                    "min": 0,
                    "max": 100
                }
            }
        ))
        panel_id += 1
        
        # GPU使用率（如果有GPU）
        panels.append(Panel(
            id=panel_id,
            title="GPU使用率",
            type="graph", 
            datasource=self.datasource_name,
            targets=[{
                "expr": "gpu_usage_percent",
                "format": "time_series",
                "legendFormat": "GPU {{gpu_id}}"
            }],
            x=6, y=8, width=6, height=8,
            field_config={
                "defaults": {
                    "unit": "percent",
                    "min": 0,
                    "max": 100
                }
            }
        ))
        panel_id += 1
        
        # 数据库连接池
        panels.append(Panel(
            id=panel_id,
            title="数据库连接池",
            type="graph",
            datasource=self.datasource_name,
            targets=[
                {
                    "expr": "database_connections_active",
                    "format": "time_series",
                    "legendFormat": "活跃连接"
                },
                {
                    "expr": "database_connections_idle",
                    "format": "time_series", 
                    "legendFormat": "空闲连接"
                }
            ],
            x=0, y=16, width=6, height=8
        ))
        panel_id += 1
        
        # 缓存性能
        panels.append(Panel(
            id=panel_id,
            title="缓存命中率",
            type="stat",
            datasource=self.datasource_name,
            targets=[{
                "expr": "cache_hits_total / (cache_hits_total + cache_misses_total) * 100",
                "format": "time_series",
                "legendFormat": "缓存命中率"
            }],
            x=6, y=16, width=6, height=8,
            field_config={
                "defaults": {
                    "unit": "percent",
                    "color": {"mode": "value"},
                    "thresholds": {
                        "steps": [
                            {"color": "red", "value": None},
                            {"color": "yellow", "value": 70},
                            {"color": "green", "value": 90}
                        ]
                    }
                }
            }
        ))
        panel_id += 1
        
        return Dashboard(
            id=None,
            title="SHEC AI - 性能监控",
            description="系统性能指标监控仪表板", 
            tags=["shec-ai", "performance", "resources"],
            panels=panels
        )
    
    def create_ml_model_dashboard(self) -> Dashboard:
        """创建机器学习模型监控仪表板"""
        panels = []
        panel_id = 1
        
        # 模型预测延迟
        panels.append(Panel(
            id=panel_id,
            title="模型预测延迟",
            type="graph",
            datasource=self.datasource_name,
            targets=[{
                "expr": "histogram_quantile(0.95, sum(rate(prediction_duration_seconds_bucket[5m])) by (le, model_name))",
                "format": "time_series",
                "legendFormat": "{{model_name}} P95"
            }, {
                "expr": "histogram_quantile(0.50, sum(rate(prediction_duration_seconds_bucket[5m])) by (le, model_name))",
                "format": "time_series",
                "legendFormat": "{{model_name}} P50"
            }],
            x=0, y=0, width=12, height=8,
            field_config={
                "defaults": {"unit": "s"}
            }
        ))
        panel_id += 1
        
        # 模型准确性指标
        panels.append(Panel(
            id=panel_id,
            title="模型准确性",
            type="graph",
            datasource=self.datasource_name,
            targets=[{
                "expr": "model_accuracy_score",
                "format": "time_series",
                "legendFormat": "{{model_name}} 准确率"
            }],
            x=0, y=8, width=6, height=8,
            field_config={
                "defaults": {
                    "unit": "percentunit",
                    "min": 0,
                    "max": 1
                }
            }
        ))
        panel_id += 1
        
        # 模型预测分布
        panels.append(Panel(
            id=panel_id,
            title="预测结果分布",
            type="piechart",
            datasource=self.datasource_name,
            targets=[{
                "expr": "sum(increase(predictions_total[5m])) by (prediction_type)",
                "format": "time_series",
                "legendFormat": "{{prediction_type}}"
            }],
            x=6, y=8, width=6, height=8
        ))
        panel_id += 1
        
        # 模型负载
        panels.append(Panel(
            id=panel_id,
            title="模型请求量",
            type="graph",
            datasource=self.datasource_name,
            targets=[{
                "expr": "sum(rate(predictions_total[5m])) by (model_name)",
                "format": "time_series",
                "legendFormat": "{{model_name}}"
            }],
            x=0, y=16, width=12, height=8
        ))
        panel_id += 1
        
        return Dashboard(
            id=None,
            title="SHEC AI - 模型监控",
            description="机器学习模型性能监控仪表板",
            tags=["shec-ai", "ml", "models"],
            panels=panels
        )
    
    def generate_dashboard_json(self, dashboard: Dashboard) -> Dict[str, Any]:
        """生成Grafana仪表板JSON配置"""
        panels_json = []
        
        for panel in dashboard.panels:
            panel_json = {
                "id": panel.id,
                "title": panel.title,
                "type": panel.type,
                "datasource": {"type": "prometheus", "uid": panel.datasource},
                "targets": panel.targets,
                "gridPos": {
                    "h": panel.height,
                    "w": panel.width,
                    "x": panel.x,
                    "y": panel.y
                }
            }
            
            if panel.options:
                panel_json["options"] = panel.options
            
            if panel.field_config:
                panel_json["fieldConfig"] = panel.field_config
            
            panels_json.append(panel_json)
        
        dashboard_json = {
            "id": dashboard.id,
            "title": dashboard.title,
            "description": dashboard.description,
            "tags": dashboard.tags,
            "refresh": dashboard.refresh,
            "time": {
                "from": dashboard.time_from,
                "to": dashboard.time_to
            },
            "panels": panels_json,
            "schemaVersion": 36,
            "version": 1,
            "editable": True
        }
        
        return dashboard_json

class WebDashboardGenerator:
    """Web监控仪表板生成器"""
    
    def generate_html_dashboard(self, title: str = "SHEC AI 监控仪表板") -> str:
        """生成HTML监控仪表板"""
        html_template = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .metric-card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-title {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }}
        .metric-value {{
            font-size: 36px;
            font-weight: bold;
            color: #007bff;
            margin-bottom: 10px;
        }}
        .metric-description {{
            color: #666;
            font-size: 14px;
        }}
        .status-good {{ color: #28a745; }}
        .status-warning {{ color: #ffc107; }}
        .status-error {{ color: #dc3545; }}
        .chart-container {{
            height: 400px;
            margin: 20px 0;
        }}
        .alerts-panel {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .alert-item {{
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
        }}
        .alert-critical {{ background-color: #f8d7da; border-left: 4px solid #dc3545; }}
        .alert-warning {{ background-color: #fff3cd; border-left: 4px solid #ffc107; }}
        .alert-info {{ background-color: #d1ecf1; border-left: 4px solid #17a2b8; }}
        .refresh-btn {{
            background-color: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }}
        .refresh-btn:hover {{
            background-color: #0056b3;
        }}
        .last-update {{
            color: #666;
            font-size: 14px;
            margin-left: 10px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <button class="refresh-btn" onclick="refreshDashboard()">刷新数据</button>
        <span class="last-update" id="lastUpdate">最后更新: 加载中...</span>
    </div>

    <div class="dashboard-grid">
        <div class="metric-card">
            <div class="metric-title">API请求总数</div>
            <div class="metric-value" id="totalRequests">-</div>
            <div class="metric-description">过去5分钟的API请求总数</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-title">平均响应时间</div>
            <div class="metric-value" id="avgResponseTime">-</div>
            <div class="metric-description">API平均响应时间（秒）</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-title">系统CPU使用率</div>
            <div class="metric-value" id="cpuUsage">-</div>
            <div class="metric-description">当前CPU使用百分比</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-title">内存使用率</div>
            <div class="metric-value" id="memoryUsage">-</div>
            <div class="metric-description">当前内存使用百分比</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-title">预测成功率</div>
            <div class="metric-value" id="predictionSuccessRate">-</div>
            <div class="metric-description">模型预测成功率</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-title">缓存命中率</div>
            <div class="metric-value" id="cacheHitRate">-</div>
            <div class="metric-description">Redis缓存命中率</div>
        </div>
    </div>

    <div class="metric-card">
        <div class="metric-title">API响应时间趋势</div>
        <div id="responseTimeChart" class="chart-container"></div>
    </div>

    <div class="metric-card">
        <div class="metric-title">系统资源使用趋势</div>
        <div id="resourceChart" class="chart-container"></div>
    </div>

    <div class="alerts-panel">
        <div class="metric-title">活跃告警</div>
        <div id="alertsContainer">
            <div class="alert-item alert-info">暂无活跃告警</div>
        </div>
    </div>

    <script>
        let lastUpdateTime = Date.now();
        
        // 模拟数据更新
        function refreshDashboard() {{
            // 更新最后更新时间
            document.getElementById('lastUpdate').textContent = 
                '最后更新: ' + new Date().toLocaleString('zh-CN');
            
            // 模拟指标数据
            updateMetrics();
            updateCharts();
            updateAlerts();
        }}
        
        function updateMetrics() {{
            // 模拟实时指标数据
            document.getElementById('totalRequests').textContent = 
                Math.floor(Math.random() * 1000 + 500);
            
            const responseTime = (Math.random() * 2 + 0.5).toFixed(2);
            document.getElementById('avgResponseTime').textContent = responseTime + 's';
            
            const cpuUsage = (Math.random() * 30 + 20).toFixed(1);
            document.getElementById('cpuUsage').textContent = cpuUsage + '%';
            
            const memoryUsage = (Math.random() * 40 + 30).toFixed(1);
            document.getElementById('memoryUsage').textContent = memoryUsage + '%';
            
            const successRate = (Math.random() * 10 + 90).toFixed(1);
            document.getElementById('predictionSuccessRate').textContent = successRate + '%';
            
            const cacheHitRate = (Math.random() * 20 + 75).toFixed(1);
            document.getElementById('cacheHitRate').textContent = cacheHitRate + '%';
        }}
        
        function updateCharts() {{
            // 生成响应时间趋势图
            const now = Date.now();
            const timeLabels = [];
            const responseTimeData = [];
            
            for (let i = 20; i >= 0; i--) {{
                timeLabels.push(new Date(now - i * 60000).toLocaleTimeString());
                responseTimeData.push(Math.random() * 2 + 0.5);
            }}
            
            const responseTimeTrace = {{
                x: timeLabels,
                y: responseTimeData,
                type: 'scatter',
                mode: 'lines+markers',
                name: '响应时间',
                line: {{color: '#007bff'}}
            }};
            
            Plotly.newPlot('responseTimeChart', [responseTimeTrace], {{
                title: 'API响应时间趋势',
                xaxis: {{title: '时间'}},
                yaxis: {{title: '响应时间 (秒)'}}
            }});
            
            // 生成系统资源趋势图
            const cpuData = [];
            const memoryData = [];
            
            for (let i = 20; i >= 0; i--) {{
                cpuData.push(Math.random() * 30 + 20);
                memoryData.push(Math.random() * 40 + 30);
            }}
            
            const cpuTrace = {{
                x: timeLabels,
                y: cpuData,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'CPU使用率',
                line: {{color: '#28a745'}}
            }};
            
            const memoryTrace = {{
                x: timeLabels,
                y: memoryData,
                type: 'scatter',
                mode: 'lines+markers',
                name: '内存使用率',
                line: {{color: '#ffc107'}}
            }};
            
            Plotly.newPlot('resourceChart', [cpuTrace, memoryTrace], {{
                title: '系统资源使用趋势',
                xaxis: {{title: '时间'}},
                yaxis: {{title: '使用率 (%)'}}
            }});
        }}
        
        function updateAlerts() {{
            const alertsContainer = document.getElementById('alertsContainer');
            
            // 模拟告警数据
            const alerts = [
                {{level: 'warning', message: 'CPU使用率较高 (75%)', time: '2分钟前'}},
                {{level: 'info', message: '数据库连接池使用率正常', time: '5分钟前'}}
            ];
            
            if (alerts.length === 0) {{
                alertsContainer.innerHTML = '<div class="alert-item alert-info">暂无活跃告警</div>';
                return;
            }}
            
            alertsContainer.innerHTML = alerts.map(alert => 
                `<div class="alert-item alert-${{alert.level}}">
                    <strong>${{alert.message}}</strong> - ${{alert.time}}
                </div>`
            ).join('');
        }}
        
        // 定期更新数据
        setInterval(refreshDashboard, 30000); // 每30秒更新一次
        
        // 初始化加载
        refreshDashboard();
    </script>
</body>
</html>
        """
        
        return html_template

class DashboardManager:
    """仪表板管理器"""
    
    def __init__(self, output_dir: str = "monitoring/dashboards"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.grafana_generator = GrafanaDashboardGenerator()
        self.web_generator = WebDashboardGenerator()
    
    def generate_all_grafana_dashboards(self):
        """生成所有Grafana仪表板"""
        dashboards = [
            ("system_overview", self.grafana_generator.create_system_overview_dashboard()),
            ("performance", self.grafana_generator.create_performance_dashboard()), 
            ("ml_models", self.grafana_generator.create_ml_model_dashboard())
        ]
        
        for name, dashboard in dashboards:
            json_config = self.grafana_generator.generate_dashboard_json(dashboard)
            
            output_path = self.output_dir / f"{name}_dashboard.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"已生成Grafana仪表板: {output_path}")
    
    def generate_web_dashboard(self):
        """生成Web监控仪表板"""
        html_content = self.web_generator.generate_html_dashboard()
        
        output_path = self.output_dir / "web_dashboard.html"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"已生成Web监控仪表板: {output_path}")
        return output_path
    
    def generate_all_dashboards(self):
        """生成所有监控仪表板"""
        self.generate_all_grafana_dashboards()
        web_dashboard_path = self.generate_web_dashboard()
        
        logger.info("所有监控仪表板已生成完成")
        return {
            "grafana_dashboards": list(self.output_dir.glob("*_dashboard.json")),
            "web_dashboard": web_dashboard_path
        }

# 便捷函数
def generate_monitoring_dashboards(output_dir: str = "monitoring/dashboards") -> Dict[str, Any]:
    """生成监控仪表板"""
    manager = DashboardManager(output_dir)
    return manager.generate_all_dashboards()

def create_grafana_dashboard_config() -> Dict[str, Any]:
    """创建Grafana配置示例"""
    return {
        "datasources": [
            {
                "name": "Prometheus",
                "type": "prometheus",
                "url": "http://localhost:9090",
                "access": "proxy",
                "isDefault": True
            }
        ],
        "dashboard_import_urls": [
            "/monitoring/dashboards/system_overview_dashboard.json",
            "/monitoring/dashboards/performance_dashboard.json", 
            "/monitoring/dashboards/ml_models_dashboard.json"
        ]
    }
