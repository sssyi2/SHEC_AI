#!/usr/bin/env python3
"""
SHEC AI 本地开发启动脚本
使用本地Python环境 + Docker数据库的高效开发方式
"""

import os
import sys
import time
from pathlib import Path

def setup_environment():
    """设置开发环境变量"""
    print("🔧 配置开发环境...")
    
    # 开发环境配置
    os.environ.update({
        # Flask开发配置
        "FLASK_ENV": "development",
        "FLASK_DEBUG": "1",
        
        # 数据库配置（连接Docker服务）
        "DB_HOST": "localhost",
        "DB_PORT": "3307",
        "DB_USER": "shec_user",
        "DB_PASSWORD": "shec_password",
        "DB_NAME": "shec_psims",
        
        # Redis配置（连接Docker服务）
        "REDIS_HOST": "localhost", 
        "REDIS_PORT": "6379",
        # "REDIS_PASSWORD": "",  # Redis无密码
        
        # 应用配置
        "APP_ENV": "development",
        "LOG_LEVEL": "DEBUG",
        "SECRET_KEY": "dev-secret-key-change-in-production",
        
        # AI模型配置
        "MODEL_PATH": "./models",
        "CACHE_ENABLED": "true",
        
        # 跨域配置
        "CORS_ORIGINS": "*"
    })
    
    print("✅ 环境变量已设置")

def check_dependencies():
    """检查Python依赖"""
    print("📦 检查Python依赖...")
    
    required_packages = [
        'flask', 'marshmallow', 'pymysql', 
        'redis', 'psutil', 'gunicorn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt -r requirements-base.txt")
        return False
    
    print("✅ Python依赖检查完成")
    return True

def check_database_connection():
    """检查数据库连接"""
    print("🔌 检查数据库连接...")
    
    import socket
    
    # 检查MySQL连接
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex(('localhost', 3307))
        sock.close()
        if result != 0:
            print("❌ MySQL连接失败 (localhost:3307)")
            print("请确保Docker数据库服务已启动:")
            print("docker-compose -f docker-compose.dev.yml up -d mysql redis")
            return False
    except Exception as e:
        print(f"❌ MySQL连接检查失败: {e}")
        return False
    
    # 检查Redis连接
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        sock.settimeout(3)
        result = sock.connect_ex(('localhost', 6379))
        sock.close()
        if result != 0:
            print("❌ Redis连接失败 (localhost:6379)")
            return False
    except Exception as e:
        print(f"❌ Redis连接检查失败: {e}")
        return False
    
    print("✅ 数据库连接正常")
    return True

def start_application():
    """启动Flask应用"""
    print("🚀 启动SHEC AI应用...")
    print("=" * 50)
    print("开发模式启动:")
    print("- 代码修改后自动重载")
    print("- 调试模式开启") 
    print("- 访问地址: http://localhost:5000")
    print("- 健康检查: http://localhost:5000/api/health")
    print("- 按 Ctrl+C 停止服务")
    print("=" * 50)
    
    try:
        # 导入并启动应用
        from app import create_app
        app = create_app()
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            use_reloader=True,  # 开启热重载
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n👋 开发服务已停止")
    except Exception as e:
        print(f"❌ 应用启动失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("🎯 SHEC AI 本地开发环境启动")
    print("=" * 50)
    
    # 切换到项目根目录
    os.chdir(Path(__file__).parent)
    
    # 设置环境
    setup_environment()
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 检查数据库
    if not check_database_connection():
        print("\n💡 提示: 请先启动数据库服务:")
        print("cd E:\\vuework\\SHEC-PSIMS")
        print("docker-compose -f SHEC_AI/docker-compose.dev.yml up -d mysql redis")
        sys.exit(1)
    
    # 启动应用
    start_application()

if __name__ == "__main__":
    main()
