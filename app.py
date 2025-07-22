#!/usr/bin/env python3
"""
SHEC AI 智能健康预测系统
Flask 主应用程序

Author: SHEC AI Team
Created: 2025-07-18
Version: v2.0
"""

import os
import sys
from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime
import logging
from config.settings import get_config
from utils.logger import setup_logger
from utils.database import init_database
from utils.redis_client import init_redis

def create_app(config_name=None):
    """创建Flask应用工厂函数"""
    
    # 创建Flask应用实例
    app = Flask(__name__)
    
    # 加载配置
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'development')
    
    config = get_config(config_name)
    app.config.from_object(config)
    
    # 设置日志
    setup_logger(app)
    app.logger.info(f"启动 SHEC AI 应用 - 环境: {config_name}")
    
    # 启用CORS
    CORS(app, resources={
        r"/api/*": {
            "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
    
    # 初始化数据库连接
    try:
        init_database(app)
        app.logger.info("数据库连接初始化成功")
    except Exception as e:
        app.logger.error(f"数据库连接初始化失败: {str(e)}")
    
    # 初始化Redis连接
    try:
        init_redis(app)
        app.logger.info("Redis连接初始化成功")
    except Exception as e:
        app.logger.error(f"Redis连接初始化失败: {str(e)}")
    
    # 注册蓝图
    register_blueprints(app)
    
    # 注册错误处理器
    register_error_handlers(app)
    
    # 注册请求钩子
    register_hooks(app)
    
    return app

def register_blueprints(app):
    """注册蓝图 - 兼容版本"""
    
    # 注册根路由
    @app.route('/')
    def index():
        return {
            "message": "SHEC AI 服务正在运行",
            "version": "1.0.0",
            "status": "running",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    try:
        from api.health import health_bp
        app.register_blueprint(health_bp)
        app.logger.info("健康检查蓝图注册成功")
    except ImportError as e:
        app.logger.error(f"健康检查蓝图导入失败: {str(e)}")
    
    try:
        from api.data_compatible import data_bp
        app.register_blueprint(data_bp)
        app.logger.info("数据管理蓝图注册成功 (兼容模式)")
    except ImportError as e:
        app.logger.error(f"数据管理蓝图导入失败: {str(e)}")
    
    try:
        from api.predict_compatible import predict_bp
        app.register_blueprint(predict_bp)
        app.logger.info("预测服务蓝图注册成功 (兼容模式)")
    except ImportError as e:
        app.logger.error(f"预测服务蓝图导入失败: {str(e)}")
    
    try:
        from api.models import models_bp
        app.register_blueprint(models_bp)
        app.logger.info("模型管理蓝图注册成功")
    except ImportError as e:
        app.logger.error(f"模型管理蓝图导入失败: {str(e)}")
    
    app.logger.info("所有蓝图注册完成")

def register_error_handlers(app):
    """注册错误处理器"""
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'error': 'Not Found',
            'message': '请求的资源不存在',
            'status_code': 404,
            'timestamp': datetime.utcnow().isoformat()
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error(f"内部服务器错误: {str(error)}")
        return jsonify({
            'error': 'Internal Server Error',
            'message': '服务器内部错误，请稍后重试',
            'status_code': 500,
            'timestamp': datetime.utcnow().isoformat()
        }), 500
    
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({
            'error': 'Bad Request',
            'message': '请求参数错误',
            'status_code': 400,
            'timestamp': datetime.utcnow().isoformat()
        }), 400

def register_hooks(app):
    """注册请求钩子"""
    
    @app.before_request
    def before_request():
        """请求前处理"""
        # 记录请求信息
        app.logger.info(f"请求: {request.method} {request.url}")
        
        # 设置请求开始时间
        request.start_time = datetime.utcnow()
    
    @app.after_request
    def after_request(response):
        """请求后处理"""
        # 计算请求处理时间
        if hasattr(request, 'start_time'):
            duration = (datetime.utcnow() - request.start_time).total_seconds()
            app.logger.info(f"响应: {response.status_code} - 耗时: {duration:.3f}s")
        
        # 设置安全头
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        
        return response

# 创建应用实例
app = create_app()

if __name__ == '__main__':
    # 开发环境直接运行
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
