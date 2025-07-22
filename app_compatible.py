"""
SHEC AI 健康预测系统主应用
完全兼容现有SHEC-PSIMS数据库结构
"""

import os
import sys
from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime

# 配置Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import get_config
from utils.logger import setup_logging, get_logger
from utils.database_adapter import db_adapter

def create_app(config_name=None):
    """应用工厂函数"""
    app = Flask(__name__)
    
    # 配置设置
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'development')
    
    config = get_config(config_name)
    app.config.from_object(config)
    
    # 设置日志
    setup_logging(app.config.get('LOG_LEVEL', 'INFO'))
    logger = get_logger(__name__)
    
    # 启用CORS
    CORS(app, resources={
        r"/api/*": {
            "origins": ["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:3000"],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
    
    # 注册兼容版本的蓝图
    try:
        from api.health import health_bp
        from api.data_compatible import data_bp
        from api.predict_compatible import predict_bp
        from api.models import models_bp
        
        app.register_blueprint(health_bp)
        app.register_blueprint(data_bp)
        app.register_blueprint(predict_bp)
        app.register_blueprint(models_bp)
        
        logger.info("所有API蓝图注册成功（兼容模式）")
        
    except ImportError as e:
        logger.error(f"蓝图导入失败: {str(e)}")
        # 降级处理，只注册可用的蓝图
        try:
            from api.health import health_bp
            app.register_blueprint(health_bp)
            logger.warning("只注册了健康检查蓝图")
        except ImportError:
            logger.error("无法导入任何蓝图")
    
    # 根路由
    @app.route('/')
    def index():
        """根路径欢迎页面"""
        return jsonify({
            'message': 'SHEC AI 健康预测系统',
            'version': '2.0.0',
            'status': 'running',
            'compatibility_mode': 'SHEC-PSIMS',
            'timestamp': datetime.utcnow().isoformat(),
            'endpoints': {
                'health_check': '/api/health',
                'data_management': '/api/data',
                'prediction_services': '/api/predict',
                'model_management': '/api/models'
            }
        })
    
    # 全局错误处理
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'error': 'Not Found',
            'message': '请求的资源不存在',
            'status_code': 404
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"内部服务器错误: {str(error)}")
        return jsonify({
            'error': 'Internal Server Error',
            'message': '服务器内部错误',
            'status_code': 500
        }), 500
    
    @app.errorhandler(Exception)
    def handle_exception(e):
        logger.error(f"未处理异常: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Unexpected Error',
            'message': '系统发生未知错误',
            'status_code': 500
        }), 500
    
    # 请求/响应日志
    @app.before_request
    def log_request():
        if request.endpoint and not request.endpoint.startswith('static'):
            logger.info(f"请求: {request.method} {request.path} - 来源: {request.remote_addr}")
    
    @app.after_request
    def log_response(response):
        if request.endpoint and not request.endpoint.startswith('static'):
            logger.info(f"响应: {response.status_code} - {request.method} {request.path}")
        return response
    
    # 启动时初始化检查
    with app.app_context():
        def initialize_app():
            """应用首次启动时的初始化"""
            logger.info("=== SHEC AI 系统启动 ===")
            logger.info(f"配置环境: {config_name}")
            logger.info(f"兼容模式: SHEC-PSIMS数据库")
            
            # 数据库连接测试
            try:
                db_status = db_adapter.db.test_connection()
                if db_status:
                    logger.info("✅ 数据库连接成功")
                else:
                    logger.warning("⚠️ 数据库连接失败")
            except Exception as e:
                logger.error(f"❌ 数据库连接异常: {str(e)}")
            
            # GPU检查（如果可用）
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    logger.info(f"✅ GPU可用: {gpu_name}")
                    logger.info(f"CUDA版本: {torch.version.cuda}")
                    logger.info(f"PyTorch版本: {torch.__version__}")
                else:
                    logger.info("ℹ️ 使用CPU模式")
            except ImportError:
                logger.info("ℹ️ PyTorch未安装，跳过GPU检查")
            except Exception as e:
                logger.warning(f"⚠️ GPU检查异常: {str(e)}")
            
            logger.info("=== 系统初始化完成 ===")
        
        # 立即执行初始化
        initialize_app()
    
    return app

# 为向后兼容保留的全局app实例
app = create_app()

if __name__ == '__main__':
    # 开发环境运行
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    print(f"""
=== SHEC AI 健康预测系统启动 ===
🏥 系统: SHEC AI Health Prediction System
📊 版本: 2.0.0 (兼容SHEC-PSIMS)
🌐 地址: http://localhost:{port}
🔧 调试模式: {debug}
💾 数据库: MySQL (兼容现有结构)
🧠 AI功能: 健康风险预测、疾病预测
⚡ GPU支持: 自动检测
=======================================
    """)
    
    try:
        app.run(
            host='0.0.0.0',
            port=port,
            debug=debug,
            threaded=True
        )
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"应用启动失败: {str(e)}")
        sys.exit(1)
