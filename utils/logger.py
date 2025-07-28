"""
日志配置模块
统一管理应用日志
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import sys

# 全局日志配置
_logger_configured = False

def configure_logging():
    """配置全局日志设置"""
    global _logger_configured
    
    if _logger_configured:
        return
    
    # 创建日志目录
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # 配置根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 清除现有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 文件处理器
    try:
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, 'shec_ai.log'),
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        
        # 为文件处理器使用不包含中文的格式器
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        print(f"无法创建文件日志处理器: {e}")
    
    _logger_configured = True
    
    # 记录配置完成信息
    root_logger.info("日志系统配置完成")

def setup_logger(app):
    """设置应用日志"""
    
    # 确保全局日志已配置
    configure_logging()
    
    # 确保日志目录存在
    log_dir = os.path.dirname(app.config.get('LOG_FILE', 'logs/shec_ai.log'))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # 设置日志级别
    log_level = getattr(logging, app.config.get('LOG_LEVEL', 'INFO').upper())
    app.logger.setLevel(log_level)
    
    # 记录启动信息
    app.logger.info(f"SHEC AI 应用启动 - {datetime.now()}")
    app.logger.info(f"日志级别: {app.config.get('LOG_LEVEL', 'INFO')}")
    app.logger.info(f"运行模式: {'调试' if app.debug else '生产'}")

def get_logger(name):
    """获取指定名称的日志器"""
    # 确保日志系统已配置
    configure_logging()
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    return logger
