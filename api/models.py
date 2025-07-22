"""
模型管理API模块
提供模型信息、训练状态查询等接口
"""

from flask import Blueprint, jsonify, request, current_app
from datetime import datetime
import os
import torch
from utils.database import execute_query
from utils.redis_client import CacheManager

models_bp = Blueprint('models', __name__)

@models_bp.route('/models', methods=['GET'])
def list_models():
    """获取模型列表"""
    
    try:
        # 从配置获取模型路径
        model_path = current_app.config.get('MODEL_PATH', 'models/saved_models')
        
        models = []
        
        # 扫描模型目录
        if os.path.exists(model_path):
            for filename in os.listdir(model_path):
                if filename.endswith(('.pth', '.pt', '.pkl')):
                    file_path = os.path.join(model_path, filename)
                    file_stat = os.stat(file_path)
                    
                    model_info = {
                        'name': filename,
                        'path': file_path,
                        'size': file_stat.st_size,
                        'created_at': datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                        'modified_at': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                        'type': 'pytorch' if filename.endswith(('.pth', '.pt')) else 'sklearn'
                    }
                    
                    # 尝试加载模型获取更多信息
                    try:
                        if filename.endswith(('.pth', '.pt')):
                            checkpoint = torch.load(file_path, map_location='cpu')
                            if isinstance(checkpoint, dict):
                                model_info.update({
                                    'version': checkpoint.get('version', 'unknown'),
                                    'epoch': checkpoint.get('epoch', 'unknown'),
                                    'accuracy': checkpoint.get('accuracy', 'unknown'),
                                    'loss': checkpoint.get('loss', 'unknown')
                                })
                    except Exception as e:
                        current_app.logger.warning(f"无法加载模型信息 {filename}: {str(e)}")
                    
                    models.append(model_info)
        
        # 添加默认模型信息
        default_models = [
            {
                'name': 'health_lstm_v1.0',
                'type': 'pytorch',
                'status': 'available',
                'description': '健康趋势预测LSTM模型',
                'input_features': ['age', 'gender', 'systolic_bp', 'diastolic_bp', 'heart_rate'],
                'output_type': 'time_series_prediction',
                'accuracy': 0.85,
                'version': '1.0'
            },
            {
                'name': 'risk_lgb_v1.0',
                'type': 'lightgbm',
                'status': 'available',
                'description': '健康风险评估LightGBM模型',
                'input_features': ['age', 'gender', 'systolic_bp', 'diastolic_bp', 'bmi'],
                'output_type': 'classification',
                'accuracy': 0.89,
                'version': '1.0'
            },
            {
                'name': 'anomaly_detection_v1.0',
                'type': 'pytorch',
                'status': 'training',
                'description': '异常检测自编码器模型',
                'input_features': ['multiple_health_metrics'],
                'output_type': 'anomaly_score',
                'accuracy': 0.82,
                'version': '1.0'
            }
        ]
        
        return jsonify({
            'models': models + default_models,
            'total_count': len(models) + len(default_models),
            'model_path': model_path,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"获取模型列表失败: {str(e)}")
        return jsonify({
            'error': '无法获取模型列表',
            'details': str(e) if current_app.debug else None,
            'status_code': 500
        }), 500

@models_bp.route('/models/<model_name>', methods=['GET'])
def get_model_info(model_name):
    """获取指定模型的详细信息"""
    
    try:
        # 缓存键
        cache_key = f"model_info:{model_name}"
        
        # 检查缓存
        cached_info = CacheManager.get(cache_key)
        if cached_info:
            return jsonify(cached_info)
        
        # 模拟模型信息
        model_details = {
            'health_lstm_v1.0': {
                'name': 'health_lstm_v1.0',
                'type': 'pytorch',
                'architecture': 'LSTM',
                'description': '用于预测健康指标时间序列的LSTM模型',
                'input_shape': [None, 10, 6],  # [batch_size, sequence_length, features]
                'output_shape': [None, 3],     # [batch_size, predictions]
                'parameters': {
                    'hidden_size': 64,
                    'num_layers': 2,
                    'dropout': 0.2,
                    'learning_rate': 0.001
                },
                'training_info': {
                    'dataset_size': 10000,
                    'epochs': 100,
                    'batch_size': 32,
                    'validation_split': 0.2,
                    'early_stopping': True
                },
                'performance': {
                    'train_accuracy': 0.87,
                    'val_accuracy': 0.85,
                    'test_accuracy': 0.84,
                    'mse': 0.023,
                    'mae': 0.12
                },
                'features': [
                    'age', 'gender', 'systolic_bp', 'diastolic_bp', 
                    'heart_rate', 'temperature'
                ],
                'target_variables': ['systolic_bp_7d', 'diastolic_bp_7d', 'heart_rate_7d'],
                'last_trained': '2025-07-15T10:30:00Z',
                'status': 'available'
            },
            'risk_lgb_v1.0': {
                'name': 'risk_lgb_v1.0',
                'type': 'lightgbm',
                'description': '基于LightGBM的健康风险分类模型',
                'parameters': {
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': 0
                },
                'training_info': {
                    'dataset_size': 50000,
                    'num_boost_round': 1000,
                    'early_stopping_rounds': 100,
                    'validation_split': 0.2
                },
                'performance': {
                    'train_accuracy': 0.92,
                    'val_accuracy': 0.89,
                    'test_accuracy': 0.88,
                    'precision': 0.87,
                    'recall': 0.89,
                    'f1_score': 0.88,
                    'auc': 0.93
                },
                'features': [
                    'age', 'gender', 'bmi', 'systolic_bp', 'diastolic_bp',
                    'heart_rate', 'blood_sugar', 'cholesterol'
                ],
                'target_classes': ['low_risk', 'medium_risk', 'high_risk', 'critical_risk'],
                'feature_importance': {
                    'age': 0.25,
                    'systolic_bp': 0.22,
                    'bmi': 0.18,
                    'blood_sugar': 0.15,
                    'diastolic_bp': 0.12,
                    'cholesterol': 0.08
                },
                'last_trained': '2025-07-16T14:20:00Z',
                'status': 'available'
            }
        }
        
        if model_name not in model_details:
            return jsonify({
                'error': f'模型 {model_name} 不存在',
                'status_code': 404
            }), 404
        
        model_info = model_details[model_name]
        model_info['timestamp'] = datetime.utcnow().isoformat()
        
        # 缓存模型信息
        CacheManager.set(cache_key, model_info, timeout=1800)  # 30分钟
        
        return jsonify(model_info)
        
    except Exception as e:
        current_app.logger.error(f"获取模型信息失败: {str(e)}")
        return jsonify({
            'error': '无法获取模型信息',
            'details': str(e) if current_app.debug else None,
            'status_code': 500
        }), 500

@models_bp.route('/models/train', methods=['POST'])
def train_model():
    """启动模型训练任务"""
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'error': '请求数据不能为空',
                'status_code': 400
            }), 400
        
        model_type = data.get('model_type', 'lstm')
        model_name = data.get('model_name', f'{model_type}_model_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}')
        
        # 验证参数
        required_params = ['dataset_path', 'target_column']
        for param in required_params:
            if param not in data:
                return jsonify({
                    'error': f'缺少必需参数: {param}',
                    'status_code': 400
                }), 400
        
        # 创建训练任务
        training_task = {
            'task_id': f'train_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}',
            'model_name': model_name,
            'model_type': model_type,
            'status': 'queued',
            'dataset_path': data['dataset_path'],
            'target_column': data['target_column'],
            'parameters': data.get('parameters', {}),
            'created_at': datetime.utcnow().isoformat(),
            'estimated_duration': '30-60 minutes',
            'progress': 0
        }
        
        # 保存训练任务到缓存
        CacheManager.set(f"training_task:{training_task['task_id']}", training_task, timeout=7200)
        
        # TODO: 这里应该启动异步训练任务
        # 可以使用Celery或其他任务队列
        current_app.logger.info(f"模型训练任务已创建: {training_task['task_id']}")
        
        return jsonify({
            'message': '训练任务已创建',
            'task_id': training_task['task_id'],
            'status': 'queued',
            'estimated_duration': training_task['estimated_duration'],
            'check_status_url': f"/api/models/train/{training_task['task_id']}"
        })
        
    except Exception as e:
        current_app.logger.error(f"创建训练任务失败: {str(e)}")
        return jsonify({
            'error': '无法创建训练任务',
            'details': str(e) if current_app.debug else None,
            'status_code': 500
        }), 500

@models_bp.route('/models/train/<task_id>', methods=['GET'])
def get_training_status(task_id):
    """获取训练任务状态"""
    
    try:
        # 从缓存获取任务信息
        task_info = CacheManager.get(f"training_task:{task_id}")
        
        if not task_info:
            return jsonify({
                'error': f'训练任务 {task_id} 不存在',
                'status_code': 404
            }), 404
        
        # 模拟训练进度更新
        if task_info['status'] == 'queued':
            task_info['status'] = 'running'
            task_info['progress'] = 15
            task_info['current_epoch'] = 5
            task_info['total_epochs'] = 100
            task_info['current_loss'] = 0.045
        elif task_info['status'] == 'running' and task_info['progress'] < 100:
            task_info['progress'] = min(100, task_info['progress'] + 10)
            task_info['current_epoch'] = min(100, task_info.get('current_epoch', 0) + 5)
            task_info['current_loss'] = max(0.001, task_info.get('current_loss', 0.05) * 0.95)
            
            if task_info['progress'] >= 100:
                task_info['status'] = 'completed'
                task_info['model_path'] = f"models/saved_models/{task_info['model_name']}.pth"
                task_info['final_accuracy'] = 0.87
        
        # 更新缓存
        CacheManager.set(f"training_task:{task_id}", task_info, timeout=7200)
        
        return jsonify(task_info)
        
    except Exception as e:
        current_app.logger.error(f"获取训练状态失败: {str(e)}")
        return jsonify({
            'error': '无法获取训练状态',
            'details': str(e) if current_app.debug else None,
            'status_code': 500
        }), 500

@models_bp.route('/models/performance', methods=['GET'])
def get_model_performance():
    """获取模型性能统计"""
    
    try:
        # 模拟性能数据
        performance_data = {
            'overall_stats': {
                'total_models': 3,
                'active_models': 2,
                'training_models': 1,
                'total_predictions': 15420,
                'average_accuracy': 0.86
            },
            'model_performance': [
                {
                    'model_name': 'health_lstm_v1.0',
                    'accuracy': 0.85,
                    'predictions_count': 8500,
                    'avg_response_time': 0.12,
                    'last_prediction': '2025-07-18T10:30:00Z'
                },
                {
                    'model_name': 'risk_lgb_v1.0',
                    'accuracy': 0.89,
                    'predictions_count': 6920,
                    'avg_response_time': 0.08,
                    'last_prediction': '2025-07-18T10:28:00Z'
                }
            ],
            'daily_stats': {
                'predictions_today': 342,
                'average_accuracy_today': 0.87,
                'peak_hour': '14:00-15:00',
                'peak_predictions': 45
            },
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify(performance_data)
        
    except Exception as e:
        current_app.logger.error(f"获取性能统计失败: {str(e)}")
        return jsonify({
            'error': '无法获取性能统计',
            'details': str(e) if current_app.debug else None,
            'status_code': 500
        }), 500
