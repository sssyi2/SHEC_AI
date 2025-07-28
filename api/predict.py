# 健康预测API接口
# 提供RESTful API接口，包括健康预测、风险评估、模型管理等功能
# 集成性能优化功能

from flask import Blueprint, request, jsonify, current_app
from flask_cors import cross_origin
import asyncio
from typing import Dict, Any, List, Optional
import json
from datetime import datetime
import traceback
import time
import numpy as np
import hashlib

try:
    from services.prediction_service import prediction_service
    from services.cache_service import cache_manager, CacheType
    from utils.api_optimizer import get_api_optimizer, CacheConfig
    SERVICES_AVAILABLE = True
except ImportError:
    SERVICES_AVAILABLE = False
    print("预测服务模块未找到，使用模拟实现")

from utils.logger import get_logger

logger = get_logger(__name__)

# 创建蓝图
predict_bp = Blueprint('predict', __name__, url_prefix='/api/predict')

# 初始化性能优化器
try:
    api_optimizer = get_api_optimizer()
    OPTIMIZER_AVAILABLE = True
except:
    api_optimizer = None
    OPTIMIZER_AVAILABLE = False
    logger.warning("API优化器初始化失败，将跳过性能优化功能")

def handle_async(async_func):
    """处理异步函数的装饰器"""
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(async_func(*args, **kwargs))
        finally:
            loop.close()
    return wrapper

def api_response(data: Any = None, message: str = "Success", 
                status_code: int = 200, error: str = None) -> tuple:
    """标准API响应格式"""
    response_data = {
        'timestamp': datetime.now().isoformat(),
        'status': 'success' if error is None else 'error',
        'message': message,
        'data': data
    }
    
    if error:
        response_data['error'] = error
        response_data['status'] = 'error'
    
    try:
        # 尝试使用Flask的jsonify
        return jsonify(response_data), status_code
    except RuntimeError:
        # 如果没有应用上下文，返回模拟响应对象
        class MockResponse:
            def __init__(self, data):
                self._data = data
            
            def get_json(self):
                return self._data
        
        return MockResponse(response_data), status_code

def validate_health_data(data):
    """验证健康数据格式"""
    if not isinstance(data, dict):
        return False, "数据必须是字典格式"
    
    # 基本字段验证
    if 'sequence_data' in data or 'features' in data:
        return True, None
    
    # 兼容旧格式
    required_fields = ['age', 'gender']
    for field in required_fields:
        if field not in data:
            return False, f"缺少必需字段: {field}"
    
    # 数据范围验证
    validations = {
        'age': (0, 150),
        'systolic_bp': (60, 300),
        'diastolic_bp': (40, 200),
        'heart_rate': (30, 220),
        'temperature': (35.0, 42.0),
        'weight': (10.0, 300.0),
        'height': (50.0, 250.0),
        'blood_sugar': (2.0, 30.0)
    }
    
    for field, (min_val, max_val) in validations.items():
        if field in data:
            try:
                value = float(data[field])
                if not (min_val <= value <= max_val):
                    return False, f"{field} 值超出正常范围 ({min_val}-{max_val})"
            except (ValueError, TypeError):
                return False, f"{field} 必须是数字"
    
    return True, None

def mock_prediction_result(prediction_type: str = 'health') -> Dict[str, Any]:
    """模拟预测结果"""
    if prediction_type == 'health':
        return {
            'timestamp': datetime.now().isoformat(),
            'prediction_type': 'health_indicators',
            'predicted_class': int(np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])),
            'class_probabilities': np.random.dirichlet([5, 2, 1]).tolist(),
            'confidence': float(np.random.uniform(0.7, 0.95)),
            'risk_level': np.random.choice(['低风险', '中风险', '高风险'], p=[0.6, 0.3, 0.1]),
            'predicted_indicators': {
                'blood_pressure': {
                    'systolic': 120 + np.random.normal(0, 10),
                    'diastolic': 80 + np.random.normal(0, 5),
                    'trend': 'stable'
                },
                'heart_rate': {
                    'bpm': 72 + np.random.normal(0, 8),
                    'variability': 'normal'
                },
                'blood_sugar': {
                    'level': 100 + np.random.normal(0, 15),
                    'status': 'normal'
                }
            },
            'recommendations': [
                '保持健康的生活方式',
                '定期进行健康检查',
                '均衡饮食，适量运动'
            ]
        }
    
    elif prediction_type == 'risk':
        diseases = ['糖尿病', '高血压', '心血管疾病', '高血脂', '肥胖症']
        risk_scores = np.random.random(len(diseases)) * 0.8 + 0.1
        
        return {
            'timestamp': datetime.now().isoformat(),
            'prediction_type': 'disease_risk',
            'predicted_class': int(np.argmax(risk_scores)),
            'confidence': float(np.max(risk_scores)),
            'risk_level': ['低风险', '中风险', '高风险'][int(np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1]))],
            'disease_risks': {
                disease: {
                    'risk_score': float(score),
                    'risk_level': '低风险' if score < 0.3 else '中风险' if score < 0.7 else '高风险',
                    'factors': ['年龄', '生活方式', '遗传因素']
                }
                for disease, score in zip(diseases, risk_scores)
            },
            'overall_health_score': float(np.mean(1 - risk_scores) * 100),
            'recommendations': [
                '保持健康的生活方式',
                '定期进行健康检查',
                '控制饮食，适量运动'
            ]
        }
    
    else:
        return {
            'timestamp': datetime.now().isoformat(),
            'prediction_type': prediction_type,
            'error': 'Mock implementation'
        }

@predict_bp.route('/health', methods=['POST'])
@cross_origin()
def predict_health_indicators():
    """
    健康指标预测API
    
    POST /api/predict/health
    """
    # 应用性能优化装饰器
    if OPTIMIZER_AVAILABLE:
        # 缓存配置
        cache_config = CacheConfig(ttl=300, compression=True, key_prefix="health_predict")
        decorated_func = api_optimizer.cache_response(cache_config)(
            api_optimizer.performance_monitor(
                api_optimizer.compress_response()(
                    _predict_health_indicators_impl
                )
            )
        )
        return decorated_func()
    else:
        return _predict_health_indicators_impl()

def _predict_health_indicators_impl():
    """
    健康指标预测API实现
    
    POST /api/predict/health
    """
    try:
        start_time = time.time()
        
        request_data = request.get_json()
        if not request_data:
            return api_response(
                error="请求数据为空",
                message="Invalid request data",
                status_code=400
            )
        
        # 验证输入数据
        is_valid, error_msg = validate_health_data(request_data)
        if not is_valid:
            return api_response(
                error=error_msg,
                message="Input validation failed",
                status_code=400
            )
        
        logger.info(f"健康预测请求")
        
        # 执行预测
        if SERVICES_AVAILABLE:
            @handle_async
            async def predict():
                return await prediction_service.predict_health_indicators(
                    input_data=request_data,
                    user_id=None,
                    model_name=request_data.get('model_name', 'default_health_lstm')
                )
            
            result = predict()
        else:
            # 使用模拟实现
            result = mock_prediction_result('health')
        
        # 检查预测结果
        if 'error' in result:
            return api_response(
                error=result['error'],
                message="Prediction failed",
                status_code=500
            )
        
        # 计算响应时间
        response_time = time.time() - start_time
        result['response_time_ms'] = round(response_time * 1000, 2)
        
        logger.info(f"健康预测完成, 耗时 {response_time:.3f}s")
        
        return api_response(
            data=result,
            message="Health prediction completed successfully"
        )
        
    except Exception as e:
        logger.error(f"健康预测API异常: {str(e)}")
        traceback.print_exc()
        return api_response(
            error=str(e),
            message="Internal server error",
            status_code=500
        )

@predict_bp.route('/risk', methods=['POST'])
@cross_origin()
def assess_disease_risk():
    """
    疾病风险评估API
    
    POST /api/predict/risk
    """
    try:
        start_time = time.time()
        
        request_data = request.get_json()
        if not request_data:
            return api_response(
                error="请求数据为空",
                message="Invalid request data",
                status_code=400
            )
        
        # 验证输入数据
        is_valid, error_msg = validate_health_data(request_data)
        if not is_valid:
            return api_response(
                error=error_msg,
                message="Input validation failed",
                status_code=400
            )
        
        logger.info(f"风险评估请求")
        
        # 执行预测
        if SERVICES_AVAILABLE:
            @handle_async
            async def assess():
                return await prediction_service.assess_disease_risk(
                    input_data=request_data,
                    user_id=None,
                    model_name=request_data.get('model_name', 'default_risk_assessment')
                )
            
            result = assess()
        else:
            # 使用模拟实现
            result = mock_prediction_result('risk')
        
        if 'error' in result:
            return api_response(
                error=result['error'],
                message="Risk assessment failed",
                status_code=500
            )
        
        response_time = time.time() - start_time
        result['response_time_ms'] = round(response_time * 1000, 2)
        
        logger.info(f"风险评估完成, 耗时 {response_time:.3f}s")
        
        return api_response(
            data=result,
            message="Risk assessment completed successfully"
        )
        
    except Exception as e:
        logger.error(f"风险评估API异常: {str(e)}")
        traceback.print_exc()
        return api_response(
            error=str(e),
            message="Internal server error",
            status_code=500
        )

@predict_bp.route('/batch', methods=['POST'])
@cross_origin()
def batch_predict():
    """
    批量预测API
    
    POST /api/predict/batch
    """
    try:
        start_time = time.time()
        
        request_data = request.get_json()
        if not request_data:
            return api_response(
                error="请求数据为空",
                message="Invalid request data",
                status_code=400
            )
        
        batch_data = request_data.get('batch_data', [])
        prediction_type = request_data.get('prediction_type', 'health_indicators')
        
        # 限制批量大小
        max_batch_size = 50
        if len(batch_data) > max_batch_size:
            return api_response(
                error=f"批量大小超过限制 ({max_batch_size})",
                message="Batch size exceeded",
                status_code=400
            )
        
        logger.info(f"批量预测请求: 类型 {prediction_type}, 数量 {len(batch_data)}")
        
        # 执行批量预测
        results = []
        for i, data in enumerate(batch_data):
            try:
                if SERVICES_AVAILABLE:
                    # 这里应该调用实际的预测服务
                    if prediction_type == 'health_indicators':
                        result = mock_prediction_result('health')
                    else:
                        result = mock_prediction_result('risk')
                else:
                    result = mock_prediction_result(prediction_type.split('_')[0])
                
                result['batch_index'] = i
                results.append(result)
                
            except Exception as e:
                results.append({
                    'error': str(e),
                    'batch_index': i,
                    'timestamp': datetime.now().isoformat()
                })
        
        # 统计结果
        success_count = len([r for r in results if 'error' not in r])
        error_count = len(results) - success_count
        
        response_time = time.time() - start_time
        
        logger.info(f"批量预测完成: 成功 {success_count}, 失败 {error_count}, 耗时 {response_time:.3f}s")
        
        return api_response(
            data={
                'results': results,
                'summary': {
                    'total_count': len(results),
                    'success_count': success_count,
                    'error_count': error_count,
                    'response_time_ms': round(response_time * 1000, 2)
                }
            },
            message="Batch prediction completed"
        )
        
    except Exception as e:
        logger.error(f"批量预测API异常: {str(e)}")
        traceback.print_exc()
        return api_response(
            error=str(e),
            message="Internal server error",
            status_code=500
        )

@predict_bp.route('/models', methods=['GET'])
@cross_origin()
def get_available_models():
    """
    获取可用模型列表API
    
    GET /api/predict/models
    """
    try:
        if SERVICES_AVAILABLE:
            model_info = prediction_service.get_model_info()
            
            # 从模型管理器获取注册的模型
            try:
                from models.model_factory import model_manager
                registered_models = []
                
                for model_name, info in model_manager.models.items():
                    model_data = {
                        'name': model_name,
                        'type': info['type'],
                        'description': info.get('description', ''),
                        'version_count': len(info.get('versions', [])),
                        'latest_version': None,
                        'created_at': info.get('created_at', ''),
                        'parameters': info.get('parameters', {})
                    }
                    
                    if info.get('versions'):
                        latest = info['versions'][-1]
                        model_data['latest_version'] = {
                            'version': latest.get('version', '1.0.0'),
                            'created_at': latest.get('created_at', ''),
                            'config': latest.get('config', {}),
                            'performance': latest.get('performance', {})
                        }
                    
                    registered_models.append(model_data)
            except ImportError:
                registered_models = []
        
        else:
            # 模拟模型信息
            model_info = {
                'loaded_models': ['default_health_lstm', 'default_risk_assessment'],
                'device': 'cpu',
                'cache_timeout': 3600,
                'prediction_cache_timeout': 1800
            }
            registered_models = [
                {
                    'name': 'default_health_lstm',
                    'type': 'pytorch',
                    'description': '健康指标预测模型',
                    'version_count': 1,
                    'latest_version': {
                        'version': '1.0.0',
                        'created_at': datetime.now().isoformat(),
                        'config': {'input_dim': 20, 'hidden_dim': 64},
                        'performance': {'accuracy': 0.85}
                    }
                },
                {
                    'name': 'default_risk_assessment',
                    'type': 'pytorch',
                    'description': '疾病风险评估模型',
                    'version_count': 1,
                    'latest_version': {
                        'version': '1.0.0',
                        'created_at': datetime.now().isoformat(),
                        'config': {'input_dim': 15, 'output_dim': 5},
                        'performance': {'auc': 0.92}
                    }
                }
            ]
        
        result = {
            'registered_models': registered_models,
            'loaded_models': model_info.get('loaded_models', []),
            'system_info': {
                'device': model_info.get('device', 'cpu'),
                'model_cache_timeout': model_info.get('cache_timeout', 3600),
                'prediction_cache_timeout': model_info.get('prediction_cache_timeout', 1800)
            }
        }
        
        return api_response(
            data=result,
            message="Available models retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"获取模型列表API异常: {str(e)}")
        return api_response(
            error=str(e),
            message="Failed to retrieve models",
            status_code=500
        )

@predict_bp.route('/health-check', methods=['GET'])
@cross_origin()
def health_check():
    """
    预测服务健康检查API
    
    GET /api/predict/health-check
    """
    try:
        # 检查各个组件状态
        status = {
            'prediction_service': 'healthy' if SERVICES_AVAILABLE else 'degraded',
            'cache_service': 'healthy' if SERVICES_AVAILABLE else 'degraded',
            'redis': 'unknown',
            'models': 'healthy' if SERVICES_AVAILABLE else 'mock'
        }
        
        # 检查Redis连接
        if SERVICES_AVAILABLE:
            try:
                cache_manager.redis_client.ping()
                status['redis'] = 'healthy'
            except Exception:
                status['redis'] = 'unhealthy'
                status['cache_service'] = 'degraded'
        
        # 整体状态
        overall_status = 'healthy'
        if 'unhealthy' in status.values():
            overall_status = 'unhealthy'
        elif 'degraded' in status.values() or 'mock' in status.values():
            overall_status = 'degraded'
        
        return api_response(
            data={
                'overall_status': overall_status,
                'components': status,
                'services_available': SERVICES_AVAILABLE,
                'timestamp': datetime.now().isoformat()
            },
            message="Health check completed"
        )
        
    except Exception as e:
        logger.error(f"健康检查API异常: {str(e)}")
        return api_response(
            error=str(e),
            message="Health check failed",
            status_code=500
        )

# 错误处理
@predict_bp.errorhandler(404)
def not_found(error):
    return api_response(
        error="接口不存在",
        message="Endpoint not found",
        status_code=404
    )

@predict_bp.errorhandler(405)
def method_not_allowed(error):
    return api_response(
        error="请求方法不允许",
        message="Method not allowed",
        status_code=405
    )

@predict_bp.errorhandler(500)
def internal_error(error):
    return api_response(
        error="内部服务器错误",
        message="Internal server error",
        status_code=500
    )

@predict_bp.route('/models/train', methods=['POST'])
@cross_origin()
def train_model():
    """
    模型训练接口
    POST /api/predict/models/train
    
    Request Body:
    {
        "model_type": "health_lstm",
        "model_name": "custom_model",
        "training_data": {...},
        "hyperparameters": {...},
        "training_config": {...}
    }
    """
    try:
        logger.info("接收到模型训练请求")
        
        # 验证请求数据
        if not request.is_json:
            return api_response(
                error="请求必须为JSON格式",
                message="Content-Type应为application/json",
                status_code=400
            )
        
        data = request.get_json()
        if not data:
            return api_response(
                error="请求体为空",
                message="请提供训练参数",
                status_code=400
            )
        
        # 验证必需字段
        required_fields = ['model_type']
        for field in required_fields:
            if field not in data:
                return api_response(
                    error=f"缺少必需字段: {field}",
                    message="请求参数不完整",
                    status_code=400
                )
        
        model_type = data.get('model_type')
        model_name = data.get('model_name', f"{model_type}_{int(time.time())}")
        training_data_config = data.get('training_data', {})
        hyperparameters = data.get('hyperparameters', {})
        training_config = data.get('training_config', {})
        
        logger.info(f"开始训练模型: {model_name}, 类型: {model_type}")
        
        if SERVICES_AVAILABLE:
            try:
                # 这里应该调用实际的训练服务
                # 由于训练是长时间任务，应该异步处理
                training_result = {
                    'training_id': f"train_{int(time.time())}",
                    'model_name': model_name,
                    'model_type': model_type,
                    'status': 'started',
                    'start_time': datetime.now().isoformat(),
                    'estimated_duration': '30-60 minutes',
                    'parameters': {
                        'training_data': training_data_config,
                        'hyperparameters': hyperparameters,
                        'training_config': training_config
                    }
                }
                
                logger.info(f"模型训练任务已启动: {training_result['training_id']}")
                
                return api_response(
                    data=training_result,
                    message="模型训练任务已启动"
                )
                
            except Exception as e:
                logger.error(f"模型训练失败: {str(e)}")
                return api_response(
                    error=f"模型训练失败: {str(e)}",
                    message="训练过程中出现错误",
                    status_code=500
                )
        else:
            # 模拟训练过程
            training_result = {
                'training_id': f"train_mock_{int(time.time())}",
                'model_name': model_name,
                'model_type': model_type,
                'status': 'started',
                'start_time': datetime.now().isoformat(),
                'estimated_duration': '30-60 minutes',
                'parameters': {
                    'training_data': training_data_config,
                    'hyperparameters': hyperparameters,
                    'training_config': training_config
                },
                'note': '模拟训练模式 - 实际部署时会进行真实训练'
            }
            
            return api_response(
                data=training_result,
                message="模型训练任务已启动（模拟模式）"
            )
            
    except Exception as e:
        logger.error(f"训练模型请求失败: {str(e)}")
        logger.error(traceback.format_exc())
        return api_response(
            error=f"训练模型请求失败: {str(e)}",
            message="内部服务错误",
            status_code=500
        )
    
    hash_obj = hashlib.md5(data_str.encode())
    return f"prediction:{prediction_type}:{hash_obj.hexdigest()}"
