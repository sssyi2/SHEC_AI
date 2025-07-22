"""
配置管理模块
支持开发、测试、生产环境的配置
"""

import os
from datetime import timedelta

# 安全导入PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

class BaseConfig:
    """基础配置类"""
    
    # Flask基础配置
    SECRET_KEY = os.getenv('SECRET_KEY', 'shec-ai-secret-key-change-in-production')
    
    # 数据库配置
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = int(os.getenv('DB_PORT', 3306))
    DB_NAME = os.getenv('DB_NAME', 'shec_psims')
    DB_USER = os.getenv('DB_USER', 'root')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '123456')
    DB_CHARSET = 'utf8mb4'
    
    # Redis配置
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB = int(os.getenv('REDIS_DB', 0))
    REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)
    
    # JWT配置
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', SECRET_KEY)
    JWT_EXPIRATION_DELTA = timedelta(hours=24)
    
    # 日志配置
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = 'logs/shec_ai.log'
    LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT = 5
    
    # AI模型配置
    MODEL_PATH = 'models/saved_models'
    if TORCH_AVAILABLE:
        DEVICE = 'cuda' if torch.cuda.is_available() and not os.getenv('FORCE_CPU') else 'cpu'
    else:
        DEVICE = 'cpu'
    TORCH_AVAILABLE = TORCH_AVAILABLE
    
    # API配置
    API_VERSION = 'v1'
    MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB
    
    # 缓存配置
    CACHE_DEFAULT_TIMEOUT = 300  # 5分钟
    CACHE_PREDICTION_TIMEOUT = 3600  # 1小时
    
    @staticmethod
    def init_app(app):
        """初始化应用配置"""
        pass

class DevelopmentConfig(BaseConfig):
    """开发环境配置"""
    
    DEBUG = True
    TESTING = False
    
    # 开发环境使用更详细的日志
    LOG_LEVEL = 'DEBUG'
    
    # 开发环境缓存时间较短
    CACHE_DEFAULT_TIMEOUT = 60
    
    @staticmethod
    def init_app(app):
        BaseConfig.init_app(app)
        
        # 开发环境特殊配置
        import logging
        logging.basicConfig(level=logging.DEBUG)

class TestingConfig(BaseConfig):
    """测试环境配置"""
    
    DEBUG = True
    TESTING = True
    
    # 测试环境使用内存数据库
    DB_NAME = 'shec_psims_test'
    REDIS_DB = 1  # 使用不同的Redis数据库
    
    # 测试环境不缓存
    CACHE_DEFAULT_TIMEOUT = 1
    
    @staticmethod
    def init_app(app):
        BaseConfig.init_app(app)

class ProductionConfig(BaseConfig):
    """生产环境配置"""
    
    DEBUG = False
    TESTING = False
    
    # 生产环境必须设置的环境变量
    SECRET_KEY = os.getenv('SECRET_KEY')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    
    # 生产环境安全配置
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    
    # 生产环境日志配置
    LOG_LEVEL = 'WARNING'
    
    @staticmethod
    def init_app(app):
        BaseConfig.init_app(app)
        
        # 生产环境错误邮件通知（可选）
        if not app.debug and not app.testing:
            # 配置错误邮件处理器
            pass

class DockerConfig(BaseConfig):
    """Docker环境配置"""
    
    DEBUG = False
    
    # Docker环境下的服务地址
    DB_HOST = os.getenv('DB_HOST', 'host.docker.internal')
    REDIS_HOST = os.getenv('REDIS_HOST', 'host.docker.internal')
    
    # Docker环境特殊配置
    RUNNING_IN_DOCKER = True
    
    @staticmethod
    def init_app(app):
        BaseConfig.init_app(app)

# 配置字典
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'docker': DockerConfig,
    'default': DevelopmentConfig
}

# 向后兼容的别名
Config = BaseConfig

def get_config(config_name=None):
    """获取配置类"""
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'default')
    
    # Docker环境检测
    if os.getenv('RUNNING_IN_DOCKER'):
        config_name = 'docker'
    
    return config.get(config_name, config['default'])
