# 数据管理API
# 处理健康数据的增删改查、批量导入导出等功能
# 完全兼容现有SHEC-PSIMS数据库结构

from flask import Blueprint, request, jsonify, current_app, send_file
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import io
import json
import traceback

from utils.database_adapter import db_adapter
from utils.validators import HealthDataValidator
from utils.logger import get_logger
from utils.redis_manager import RedisManager
from models.data_processor import HealthDataPipeline
from config.settings import Config

# 创建数据管理蓝图
data_bp = Blueprint('data', __name__, url_prefix='/api/data')

# 初始化组件
logger = get_logger(__name__)
validator = HealthDataValidator()
redis_manager = RedisManager()

@data_bp.route('/health', methods=['GET'])
def health_check():
    """数据API健康检查"""
    try:
        # 检查数据库连接
        db_status = db_adapter.db.test_connection()
        
        # 检查Redis连接
        redis_status = redis_manager.ping()
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'database': 'connected' if db_status else 'disconnected',
            'redis': 'connected' if redis_status else 'disconnected',
            'service': 'data_management'
        })
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@data_bp.route('/patients', methods=['GET'])
def get_patients():
    """获取患者列表（兼容现有数据库）"""
    try:
        # 分页参数
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 20, type=int), 100)
        
        # 搜索参数
        search = request.args.get('search', '')
        gender = request.args.get('gender', '')
        age_min = request.args.get('age_min', type=int)
        age_max = request.args.get('age_max', type=int)
        
        # 构建查询 - 使用现有的user表
        query = """
        SELECT 
            u.user_id as patient_id, u.username, u.real_name, u.age, u.gender,
            u.phone, u.email, u.created_at,
            COUNT(hr.health_record_id) as metrics_count,
            MAX(hr.measurement_time) as last_measurement
        FROM user u
        LEFT JOIN health_record hr ON u.user_id = hr.patient_id
        WHERE u.user_type = 'patient' AND u.status = 'active'
        """
        params = []
        
        # 添加过滤条件
        if search:
            query += " AND (u.username LIKE %s OR u.real_name LIKE %s OR u.phone LIKE %s)"
            search_pattern = f"%{search}%"
            params.extend([search_pattern, search_pattern, search_pattern])
        
        if gender:
            query += " AND u.gender = %s"
            params.append(gender)
        
        if age_min is not None:
            query += " AND u.age >= %s"
            params.append(age_min)
        
        if age_max is not None:
            query += " AND u.age <= %s"
            params.append(age_max)
        
        query += " GROUP BY u.user_id ORDER BY u.created_at DESC"
        
        # 执行查询
        offset = (page - 1) * per_page
        paginated_query = f"{query} LIMIT %s OFFSET %s"
        params.extend([per_page, offset])
        
        patients = db_adapter.db.fetch_all(paginated_query, params)
        
        # 获取总数
        count_query = """
        SELECT COUNT(DISTINCT u.user_id)
        FROM user u
        WHERE u.user_type = 'patient' AND u.status = 'active'
        """
        count_params = []
        
        if search:
            count_query += " AND (u.username LIKE %s OR u.real_name LIKE %s OR u.phone LIKE %s)"
            search_pattern = f"%{search}%"
            count_params.extend([search_pattern, search_pattern, search_pattern])
        
        if gender:
            count_query += " AND u.gender = %s"
            count_params.append(gender)
        
        if age_min is not None:
            count_query += " AND u.age >= %s"
            count_params.append(age_min)
        
        if age_max is not None:
            count_query += " AND u.age <= %s"
            count_params.append(age_max)
        
        total_result = db_adapter.db.fetch_one(count_query, count_params)
        total = total_result[0] if total_result else 0
        
        # 转换为字典
        result = [dict(row) for row in patients] if patients else []
        
        return jsonify({
            'success': True,
            'data': result,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total,
                'pages': (total + per_page - 1) // per_page
            }
        })
        
    except Exception as e:
        logger.error(f"获取患者列表失败: {str(e)}")
        return jsonify({
            'success': False,
            'message': '获取患者列表失败',
            'error': str(e)
        }), 500

@data_bp.route('/patients/<int:patient_id>', methods=['GET'])
def get_patient_detail(patient_id):
    """获取患者详细信息"""
    try:
        # 使用适配器获取患者信息
        patient_info = db_adapter.get_patient_info(patient_id)
        
        if not patient_info:
            return jsonify({
                'success': False,
                'message': '患者不存在'
            }), 404
        
        # 获取健康记录
        health_records = db_adapter.get_health_records(patient_id, limit=10)
        
        # 获取医疗记录
        medical_history = db_adapter.get_medical_history(patient_id)
        
        # 获取AI预测记录
        ai_predictions = db_adapter.get_ai_predictions(patient_id)
        
        return jsonify({
            'success': True,
            'data': {
                'patient_info': patient_info,
                'health_records': health_records,
                'medical_history': medical_history,
                'ai_predictions': ai_predictions
            }
        })
        
    except Exception as e:
        logger.error(f"获取患者详细信息失败: {str(e)}")
        return jsonify({
            'success': False,
            'message': '获取患者详细信息失败',
            'error': str(e)
        }), 500

@data_bp.route('/health-records', methods=['GET'])
def get_health_records():
    """获取健康记录列表"""
    try:
        # 分页参数
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 20, type=int), 100)
        
        # 过滤参数
        patient_id = request.args.get('patient_id', type=int)
        date_from = request.args.get('date_from')
        date_to = request.args.get('date_to')
        
        # 构建查询
        query = """
        SELECT 
            hr.health_record_id as metric_id,
            hr.patient_id,
            u.username,
            u.real_name,
            hr.systolic_pressure,
            hr.diastolic_pressure,
            hr.blood_sugar,
            hr.blood_sugar_type,
            hr.bmi,
            hr.weight,
            hr.height,
            hr.heart_rate,
            hr.body_temperature,
            hr.exercise_frequency,
            hr.smoking_status,
            hr.drinking_status,
            hr.measurement_time,
            hr.created_at
        FROM health_record hr
        JOIN user u ON hr.patient_id = u.user_id
        WHERE 1=1
        """
        params = []
        
        if patient_id:
            query += " AND hr.patient_id = %s"
            params.append(patient_id)
        
        if date_from:
            query += " AND hr.measurement_time >= %s"
            params.append(date_from)
        
        if date_to:
            query += " AND hr.measurement_time <= %s"
            params.append(date_to)
        
        query += " ORDER BY hr.measurement_time DESC"
        
        # 执行分页查询
        offset = (page - 1) * per_page
        paginated_query = f"{query} LIMIT %s OFFSET %s"
        params.extend([per_page, offset])
        
        metrics = db_adapter.db.fetch_all(paginated_query, params)
        
        # 获取总数
        count_query = query.replace(
            """SELECT 
            hr.health_record_id as metric_id,
            hr.patient_id,
            u.username,
            u.real_name,
            hr.systolic_pressure,
            hr.diastolic_pressure,
            hr.blood_sugar,
            hr.blood_sugar_type,
            hr.bmi,
            hr.weight,
            hr.height,
            hr.heart_rate,
            hr.body_temperature,
            hr.exercise_frequency,
            hr.smoking_status,
            hr.drinking_status,
            hr.measurement_time,
            hr.created_at""",
            "SELECT COUNT(*)"
        ).replace("ORDER BY hr.measurement_time DESC", "")
        
        count_params = params[:-2]
        total_result = db_adapter.db.fetch_one(count_query, count_params)
        total = total_result[0] if total_result else 0
        
        # 转换为字典
        result = [dict(row) for row in metrics] if metrics else []
        
        return jsonify({
            'success': True,
            'data': result,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total,
                'pages': (total + per_page - 1) // per_page
            }
        })
        
    except Exception as e:
        logger.error(f"获取健康记录列表失败: {str(e)}")
        return jsonify({
            'success': False,
            'message': '获取健康记录列表失败',
            'error': str(e)
        }), 500

@data_bp.route('/health-records', methods=['POST'])
def create_health_record():
    """创建新的健康记录"""
    try:
        data = request.get_json()
        
        # 基本验证
        if not data or 'patient_id' not in data:
            return jsonify({
                'success': False,
                'message': '患者ID不能为空'
            }), 400
        
        # 验证患者存在
        patient_info = db_adapter.get_patient_info(data['patient_id'])
        if not patient_info:
            return jsonify({
                'success': False,
                'message': '患者不存在'
            }), 404
        
        # 准备插入数据
        insert_data = {
            'patient_id': data['patient_id'],
            'systolic_pressure': data.get('systolic_pressure'),
            'diastolic_pressure': data.get('diastolic_pressure'),
            'blood_sugar': data.get('blood_sugar'),
            'blood_sugar_type': data.get('blood_sugar_type', 'fasting'),
            'bmi': data.get('bmi'),
            'weight': data.get('weight'),
            'height': data.get('height'),
            'heart_rate': data.get('heart_rate'),
            'body_temperature': data.get('body_temperature'),
            'exercise_frequency': data.get('exercise_frequency'),
            'smoking_status': data.get('smoking_status', 'never'),
            'drinking_status': data.get('drinking_status', 'never'),
            'medication_usage': json.dumps(data.get('medications', [])),
            'measurement_time': data.get('measurement_time', datetime.now().isoformat())
        }
        
        # 插入数据
        query = """
        INSERT INTO health_record (
            patient_id, systolic_pressure, diastolic_pressure, blood_sugar,
            blood_sugar_type, bmi, weight, height, heart_rate, body_temperature,
            exercise_frequency, smoking_status, drinking_status, medication_usage,
            measurement_time
        ) VALUES (
            %(patient_id)s, %(systolic_pressure)s, %(diastolic_pressure)s, %(blood_sugar)s,
            %(blood_sugar_type)s, %(bmi)s, %(weight)s, %(height)s, %(heart_rate)s, %(body_temperature)s,
            %(exercise_frequency)s, %(smoking_status)s, %(drinking_status)s, %(medication_usage)s,
            %(measurement_time)s
        )
        """
        
        record_id = db_adapter.db.execute_query(query, insert_data)
        
        logger.info(f"创建健康记录成功，ID: {record_id}, 患者: {data['patient_id']}")
        
        return jsonify({
            'success': True,
            'message': '健康记录创建成功',
            'record_id': record_id
        }), 201
        
    except Exception as e:
        logger.error(f"创建健康记录失败: {str(e)}")
        return jsonify({
            'success': False,
            'message': '创建健康记录失败',
            'error': str(e)
        }), 500

@data_bp.route('/ai-predictions', methods=['GET'])
def get_ai_predictions():
    """获取AI预测记录"""
    try:
        # 分页参数
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 20, type=int), 100)
        
        # 过滤参数
        patient_id = request.args.get('patient_id', type=int)
        prediction_type = request.args.get('prediction_type')
        risk_level = request.args.get('risk_level')
        
        # 构建查询
        query = """
        SELECT 
            apr.id as result_id,
            apr.patient_id,
            u.username,
            u.real_name,
            apr.prediction_type,
            apr.model_name,
            apr.model_version,
            apr.prediction_result,
            apr.confidence_score,
            apr.risk_level,
            apr.recommendations,
            apr.created_at
        FROM ai_prediction_results apr
        JOIN user u ON apr.patient_id = u.user_id
        WHERE 1=1
        """
        params = []
        
        if patient_id:
            query += " AND apr.patient_id = %s"
            params.append(patient_id)
        
        if prediction_type:
            query += " AND apr.prediction_type = %s"
            params.append(prediction_type)
        
        if risk_level:
            query += " AND apr.risk_level = %s"
            params.append(risk_level)
        
        query += " ORDER BY apr.created_at DESC"
        
        # 执行分页查询
        offset = (page - 1) * per_page
        paginated_query = f"{query} LIMIT %s OFFSET %s"
        params.extend([per_page, offset])
        
        predictions = db_adapter.db.fetch_all(paginated_query, params)
        
        # 获取总数
        count_query = query.replace(
            """SELECT 
            apr.id as result_id,
            apr.patient_id,
            u.username,
            u.real_name,
            apr.prediction_type,
            apr.model_name,
            apr.model_version,
            apr.prediction_result,
            apr.confidence_score,
            apr.risk_level,
            apr.recommendations,
            apr.created_at""",
            "SELECT COUNT(*)"
        ).replace("ORDER BY apr.created_at DESC", "")
        
        count_params = params[:-2]
        total_result = db_adapter.db.fetch_one(count_query, count_params)
        total = total_result[0] if total_result else 0
        
        # 转换为字典并处理JSON字段
        result = []
        if predictions:
            for row in predictions:
                row_dict = dict(row)
                # 解析JSON字段
                try:
                    if row_dict.get('prediction_result'):
                        row_dict['prediction_result'] = json.loads(row_dict['prediction_result'])
                    if row_dict.get('recommendations'):
                        row_dict['recommendations'] = json.loads(row_dict['recommendations'])
                except json.JSONDecodeError:
                    pass
                result.append(row_dict)
        
        return jsonify({
            'success': True,
            'data': result,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total,
                'pages': (total + per_page - 1) // per_page
            }
        })
        
    except Exception as e:
        logger.error(f"获取AI预测记录失败: {str(e)}")
        return jsonify({
            'success': False,
            'message': '获取AI预测记录失败',
            'error': str(e)
        }), 500

@data_bp.route('/ai-predictions', methods=['POST'])
def create_ai_prediction():
    """保存AI预测结果"""
    try:
        data = request.get_json()
        
        # 基本验证
        required_fields = ['patient_id', 'prediction_type', 'model_name', 'prediction_result']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'message': f'字段 {field} 是必需的'
                }), 400
        
        # 验证患者存在
        patient_info = db_adapter.get_patient_info(data['patient_id'])
        if not patient_info:
            return jsonify({
                'success': False,
                'message': '患者不存在'
            }), 404
        
        # 准备保存数据
        prediction_data = {
            'patient_id': data['patient_id'],
            'prediction_type': data['prediction_type'],
            'model_name': data['model_name'],
            'model_version': data.get('model_version', '1.0'),
            'input_data': json.dumps(data.get('input_data', {})),
            'prediction_result': json.dumps(data['prediction_result']),
            'confidence_score': data.get('confidence_score', 0.0),
            'risk_level': data.get('risk_level', '中'),
            'recommendations': json.dumps(data.get('recommendations', [])),
            'expires_at': data.get('expires_at')
        }
        
        # 保存预测结果
        result_id = db_adapter.save_ai_prediction(data['patient_id'], prediction_data)
        
        logger.info(f"保存AI预测结果成功，ID: {result_id}, 患者: {data['patient_id']}")
        
        return jsonify({
            'success': True,
            'message': 'AI预测结果保存成功',
            'result_id': result_id
        }), 201
        
    except Exception as e:
        logger.error(f"保存AI预测结果失败: {str(e)}")
        return jsonify({
            'success': False,
            'message': '保存AI预测结果失败',
            'error': str(e)
        }), 500

@data_bp.route('/statistics', methods=['GET'])
def get_statistics():
    """获取数据统计信息"""
    try:
        # 患者统计
        patient_stats_query = """
        SELECT 
            COUNT(*) as total_patients,
            COUNT(CASE WHEN gender = 'male' THEN 1 END) as male_count,
            COUNT(CASE WHEN gender = 'female' THEN 1 END) as female_count,
            AVG(age) as avg_age,
            COUNT(CASE WHEN created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY) THEN 1 END) as new_patients_30d
        FROM user 
        WHERE user_type = 'patient' AND status = 'active'
        """
        
        patient_stats = db_adapter.db.fetch_one(patient_stats_query)
        
        # 健康记录统计
        metrics_stats_query = """
        SELECT 
            COUNT(*) as total_records,
            COUNT(CASE WHEN created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY) THEN 1 END) as records_7d,
            COUNT(CASE WHEN created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY) THEN 1 END) as records_30d,
            AVG(systolic_pressure) as avg_systolic,
            AVG(diastolic_pressure) as avg_diastolic,
            AVG(blood_sugar) as avg_blood_sugar
        FROM health_record
        WHERE created_at >= DATE_SUB(NOW(), INTERVAL 1 YEAR)
        """
        
        metrics_stats = db_adapter.db.fetch_one(metrics_stats_query)
        
        # AI预测统计
        prediction_stats_query = """
        SELECT 
            COUNT(*) as total_predictions,
            COUNT(CASE WHEN created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY) THEN 1 END) as predictions_7d,
            COUNT(CASE WHEN risk_level = '高' THEN 1 END) as high_risk_count,
            COUNT(CASE WHEN risk_level = '中' THEN 1 END) as medium_risk_count,
            COUNT(CASE WHEN risk_level = '低' THEN 1 END) as low_risk_count,
            AVG(confidence_score) as avg_confidence
        FROM ai_prediction_results
        WHERE created_at >= DATE_SUB(NOW(), INTERVAL 1 YEAR)
        """
        
        prediction_stats = db_adapter.db.fetch_one(prediction_stats_query)
        
        # 组装结果
        stats = {
            'patients': dict(patient_stats) if patient_stats else {},
            'health_records': dict(metrics_stats) if metrics_stats else {},
            'ai_predictions': dict(prediction_stats) if prediction_stats else {},
            'updated_at': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'data': stats
        })
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {str(e)}")
        return jsonify({
            'success': False,
            'message': '获取统计信息失败',
            'error': str(e)
        }), 500

@data_bp.route('/data-pipeline/patients', methods=['GET'])
def get_patients_for_prediction():
    """获取需要进行AI预测的患者列表"""
    try:
        risk_threshold = request.args.get('risk_threshold', '中')
        limit = min(request.args.get('limit', 50, type=int), 200)
        
        # 使用适配器获取需要预测的患者
        patients = db_adapter.get_patients_for_prediction(risk_threshold)[:limit]
        
        return jsonify({
            'success': True,
            'data': patients,
            'count': len(patients)
        })
        
    except Exception as e:
        logger.error(f"获取预测候选患者列表失败: {str(e)}")
        return jsonify({
            'success': False,
            'message': '获取预测候选患者列表失败',
            'error': str(e)
        }), 500

# 错误处理器
@data_bp.errorhandler(Exception)
def handle_exception(e):
    """统一异常处理"""
    logger.error(f"数据API异常: {str(e)}", exc_info=True)
    return jsonify({
        'success': False,
        'message': '服务器内部错误',
        'error': str(e) if current_app.debug else None
    }), 500
