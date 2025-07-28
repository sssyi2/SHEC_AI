"""
Sprint 4.2 监控系统启动脚本
快速启动和配置SHEC AI监控系统
"""

import os
import sys
import time
import webbrowser
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import get_logger
from monitoring import get_monitoring_system
from monitoring.dashboard import generate_monitoring_dashboards
from app import create_app

logger = get_logger(__name__)

def setup_monitoring_environment():
    """设置监控环境"""
    logger.info("设置监控环境...")
    
    # 创建监控目录
    directories = [
        'monitoring',
        'monitoring/dashboards',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("监控环境设置完成")

def generate_dashboards():
    """生成监控仪表板"""
    logger.info("生成监控仪表板...")
    
    try:
        dashboards = generate_monitoring_dashboards()
        
        logger.info("监控仪表板生成成功:")
        logger.info(f"  - Grafana仪表板: {len(dashboards['grafana_dashboards'])} 个")
        logger.info(f"  - Web仪表板: {dashboards['web_dashboard']}")
        
        return dashboards
        
    except Exception as e:
        logger.error(f"生成监控仪表板失败: {e}")
        return None

def start_monitoring_demo():
    """启动监控系统演示"""
    logger.info("启动监控系统演示...")
    
    # 设置环境
    setup_monitoring_environment()
    
    # 生成仪表板
    dashboards = generate_dashboards()
    
    # 创建应用实例（这将自动初始化监控系统）
    app = create_app('development')
    
    # 获取监控系统实例
    monitoring_system = get_monitoring_system()
    
    if not monitoring_system:
        logger.error("监控系统未初始化")
        return
    
    logger.info("监控系统启动成功！")
    logger.info("")
    logger.info("可用的监控接口:")
    logger.info("  - 健康检查: http://localhost:5000/monitoring/health")
    logger.info("  - 指标摘要: http://localhost:5000/monitoring/metrics/summary") 
    logger.info("  - 告警列表: http://localhost:5000/monitoring/alerts")
    logger.info("  - 监控配置: http://localhost:5000/monitoring/config")
    logger.info("  - Web仪表板: http://localhost:5000/monitoring/dashboard")
    logger.info("  - Prometheus指标: http://localhost:5000/metrics")
    logger.info("")
    
    # 启动Flask应用
    try:
        logger.info("启动Flask应用服务器...")
        
        # 在开发模式下启动
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,  # 关闭debug模式避免监控冲突
            use_reloader=False  # 关闭重载器
        )
        
    except KeyboardInterrupt:
        logger.info("收到停止信号，关闭监控系统...")
        monitoring_system.shutdown()
        logger.info("监控系统已关闭")
    except Exception as e:
        logger.error(f"启动监控系统失败: {e}")

def show_monitoring_status():
    """显示监控系统状态"""
    print("\n" + "="*60)
    print("SHEC AI 监控系统 (Sprint 4.2) 状态")
    print("="*60)
    
    # 检查文件存在性
    files_to_check = [
        ('monitoring/__init__.py', '监控系统集成模块'),
        ('monitoring/prometheus_metrics.py', 'Prometheus指标收集器'),
        ('monitoring/alerting.py', '告警系统'),
        ('monitoring/dashboard.py', '仪表板生成器'),
        ('monitoring/dashboards/web_dashboard.html', 'Web监控仪表板')
    ]
    
    print("\n文件检查:")
    for file_path, description in files_to_check:
        exists = Path(file_path).exists()
        status = "✅ 存在" if exists else "❌ 缺失"
        print(f"  {status} {description}")
    
    print("\n监控功能模块:")
    print("  ✅ Prometheus指标收集")
    print("  ✅ 系统资源监控 (CPU/内存/磁盘)")
    print("  ✅ HTTP请求监控")
    print("  ✅ 预测服务监控")
    print("  ✅ 数据库性能监控")
    print("  ✅ 缓存性能监控")
    print("  ✅ GPU使用监控")
    print("  ✅ 告警规则引擎")
    print("  ✅ 邮件/Webhook通知")
    print("  ✅ Grafana仪表板")
    print("  ✅ Web监控仪表板")
    
    print("\n启动方式:")
    print("  python monitoring_start.py          # 启动监控系统演示")
    print("  python -c \"import monitoring_start; monitoring_start.show_monitoring_status()\"  # 查看状态")
    
    print("\n" + "="*60)

def create_monitoring_test():
    """创建监控系统测试脚本"""
    test_script = '''
"""
监控系统测试脚本
测试Sprint 4.2监控功能
"""

import time
import requests
import threading
from concurrent.futures import ThreadPoolExecutor

def test_api_requests():
    """测试API请求以生成监控数据"""
    base_url = "http://localhost:5000"
    
    endpoints = [
        "/",
        "/api/health/status",
        "/monitoring/health",
        "/monitoring/metrics/summary",
        "/monitoring/alerts",
        "/monitoring/config"
    ]
    
    print("开始API压力测试...")
    
    def make_request(endpoint):
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            return f"{endpoint}: {response.status_code}"
        except Exception as e:
            return f"{endpoint}: Error - {str(e)}"
    
    # 并发请求测试
    with ThreadPoolExecutor(max_workers=5) as executor:
        for i in range(10):  # 10轮测试
            print(f"\\n第 {i+1} 轮测试:")
            
            futures = [executor.submit(make_request, endpoint) for endpoint in endpoints]
            results = [future.result() for future in futures]
            
            for result in results:
                print(f"  {result}")
            
            time.sleep(2)  # 间隔2秒
    
    print("\\nAPI测试完成！请检查监控指标。")

if __name__ == "__main__":
    test_api_requests()
'''
    
    with open('monitoring_test.py', 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    logger.info("已创建监控测试脚本: monitoring_test.py")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SHEC AI 监控系统管理')
    parser.add_argument('--status', action='store_true', help='显示监控系统状态')
    parser.add_argument('--test', action='store_true', help='创建测试脚本')
    parser.add_argument('--start', action='store_true', help='启动监控系统')
    
    args = parser.parse_args()
    
    if args.status:
        show_monitoring_status()
    elif args.test:
        create_monitoring_test()
    elif args.start:
        start_monitoring_demo()
    else:
        # 默认显示状态和启动选项
        show_monitoring_status()
        
        print("\n选择操作:")
        print("1. 启动监控系统演示")
        print("2. 创建测试脚本")
        print("3. 退出")
        
        choice = input("\n请输入选择 (1-3): ").strip()
        
        if choice == '1':
            start_monitoring_demo()
        elif choice == '2':
            create_monitoring_test()
            print("已创建 monitoring_test.py 测试脚本")
        else:
            print("退出")
