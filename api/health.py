"""
健康检查API模块
提供系统健康状态检查接口
"""

from flask import Blueprint, jsonify, current_app
from datetime import datetime
import torch
import psutil
import os
from utils.database import get_db_connection
from utils.redis_client import get_redis_client

health_bp = Blueprint('health', __name__)

@health_bp.route('/health', methods=['GET'])
def health_check():
    """系统健康检查接口"""
    
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '2.0.0',
        'services': {},
        'system': {}
    }
    
    # 检查数据库连接
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
        health_status['services']['database'] = {
            'status': 'healthy',
            'type': 'MySQL'
        }
    except Exception as e:
        health_status['services']['database'] = {
            'status': 'unhealthy',
            'error': str(e),
            'type': 'MySQL'
        }
        health_status['status'] = 'unhealthy'
    
    # 检查Redis连接
    redis_client = get_redis_client()
    if redis_client:
        try:
            redis_client.ping()
            health_status['services']['redis'] = {
                'status': 'healthy',
                'type': 'Redis'
            }
        except Exception as e:
            health_status['services']['redis'] = {
                'status': 'unhealthy',
                'error': str(e),
                'type': 'Redis'
            }
    else:
        health_status['services']['redis'] = {
            'status': 'unavailable',
            'type': 'Redis'
        }
    
    # 检查PyTorch和GPU
    try:
        health_status['services']['pytorch'] = {
            'status': 'healthy',
            'version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        if torch.cuda.is_available():
            health_status['services']['pytorch']['devices'] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    except Exception as e:
        health_status['services']['pytorch'] = {
            'status': 'unhealthy',
            'error': str(e)
        }
    
    # 系统信息
    try:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        health_status['system'] = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory': {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent
            },
            'disk': {
                'total': disk.total,
                'free': disk.free,
                'percent': (disk.used / disk.total) * 100
            },
            'python_version': f"{psutil.version_info}",
            'process_id': os.getpid()
        }
    except Exception as e:
        current_app.logger.error(f"获取系统信息失败: {str(e)}")
    
    # 确定总体状态
    if health_status['status'] == 'healthy':
        status_code = 200
    else:
        status_code = 503
    
    return jsonify(health_status), status_code

@health_bp.route('/ping', methods=['GET'])
def ping():
    """简单的ping接口"""
    return jsonify({
        'message': 'pong',
        'timestamp': datetime.utcnow().isoformat()
    })

@health_bp.route('/version', methods=['GET'])
def version():
    """版本信息接口"""
    return jsonify({
        'application': 'SHEC AI',
        'version': '2.0.0',
        'build_date': '2025-07-18',
        'environment': current_app.config.get('ENV', 'development'),
        'python_version': os.sys.version,
        'pytorch_version': torch.__version__
    })
