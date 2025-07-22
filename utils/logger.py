"""
日志配置模块
统一管理应用日志
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

def setup_logger(app):
    """设置应用日志"""
    
    # 确保日志目录存在
    log_dir = os.path.dirname(app.config.get('LOG_FILE', 'logs/shec_ai.log'))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # 设置日志级别
    log_level = getattr(logging, app.config.get('LOG_LEVEL', 'INFO').upper())
    app.logger.setLevel(log_level)
    
    # 如果不是调试模式，设置文件日志处理器
    if not app.debug:
        # 文件日志处理器
        file_handler = RotatingFileHandler(
            app.config.get('LOG_FILE', 'logs/shec_ai.log'),
            maxBytes=app.config.get('LOG_MAX_BYTES', 10 * 1024 * 1024),
            backupCount=app.config.get('LOG_BACKUP_COUNT', 5),
            encoding='utf-8'
        )
        
        # 设置日志格式
        file_formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(log_level)
        
        # 添加处理器
        app.logger.addHandler(file_handler)
    
    # 控制台日志处理器（开发环境）
    if app.debug:
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(log_level)
        app.logger.addHandler(console_handler)
    
    # 记录启动信息
    app.logger.info(f"SHEC AI 应用启动 - {datetime.now()}")
    app.logger.info(f"日志级别: {app.config.get('LOG_LEVEL', 'INFO')}")
    app.logger.info(f"运行模式: {'调试' if app.debug else '生产'}")

def get_logger(name):
    """获取指定名称的日志器"""
    return logging.getLogger(name)
