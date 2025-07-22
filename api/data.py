# 数据管理API
# 处理健康数据的增删改查、批量导入导出等功能
# 兼容现有SHEC-PSIMS数据库结构

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
    """获取患者列表（使用现有数据库结构）"""
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
        WHERE u.user_type = 'patient'
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
        
        patients = db_manager.execute_query(paginated_query, params)
        
        # 获取总数
        count_query = f"SELECT COUNT(DISTINCT p.patient_id) as total FROM patients p WHERE 1=1"
        count_params = []
        
        if search:
            count_query += " AND (p.username LIKE %s OR p.real_name LIKE %s OR p.phone LIKE %s)"
            count_params.extend([search_pattern, search_pattern, search_pattern])
        
        if gender:
            count_query += " AND p.gender = %s"
            count_params.append(gender)
        
        if age_min is not None:
            count_query += " AND p.age >= %s"
            count_params.append(age_min)
        
        if age_max is not None:
            count_query += " AND p.age <= %s"
            count_params.append(age_max)
        
        total_result = db_manager.execute_query(count_query, count_params)
        total = total_result[0]['total'] if total_result else 0
        
        return jsonify({
            'data': patients,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total,
                'pages': (total + per_page - 1) // per_page
            }
        })
        
    except Exception as e:
        logger.error(f"获取患者列表失败: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': '获取患者列表失败'}), 500

@data_bp.route('/patients/<int:patient_id>/metrics', methods=['GET'])
def get_patient_metrics(patient_id: int):
    """获取患者的健康指标历史"""
    try:
        # 时间范围参数
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        metric_type = request.args.get('metric_type', 'all')
        limit = min(request.args.get('limit', 100, type=int), 500)
        
        # 检查缓存
        cache_key = f"patient_metrics:{patient_id}:{start_date}:{end_date}:{metric_type}:{limit}"
        cached_result = redis_manager.get_json(cache_key)
        if cached_result:
            return jsonify(cached_result)
        
        # 构建查询
        query = """
        SELECT * FROM health_metrics 
        WHERE patient_id = %s
        """
        params = [patient_id]
        
        if start_date:
            query += " AND measurement_time >= %s"
            params.append(start_date)
        
        if end_date:
            query += " AND measurement_time <= %s"
            params.append(end_date)
        
        query += " ORDER BY measurement_time DESC LIMIT %s"
        params.append(limit)
        
        metrics = db_manager.execute_query(query, params)
        
        # 数据处理和统计
        if metrics:
            # 基本统计
            stats = {
                'total_records': len(metrics),
                'date_range': {
                    'start': min(m['measurement_time'].isoformat() if m['measurement_time'] else None for m in metrics if m['measurement_time']),
                    'end': max(m['measurement_time'].isoformat() if m['measurement_time'] else None for m in metrics if m['measurement_time'])
                },
                'latest_measurement': metrics[0]['measurement_time'].isoformat() if metrics[0]['measurement_time'] else None
            }
            
            # 计算趋势（简单的平均值变化）
            if len(metrics) >= 2:
                recent_avg = {}
                older_avg = {}
                recent_count = min(5, len(metrics) // 2)
                
                numeric_fields = ['systolic_pressure', 'diastolic_pressure', 'blood_sugar', 'bmi', 'heart_rate']
                
                for field in numeric_fields:
                    recent_values = [float(m[field]) for m in metrics[:recent_count] if m[field] is not None]
                    older_values = [float(m[field]) for m in metrics[recent_count:] if m[field] is not None]
                    
                    if recent_values and older_values:
                        recent_avg[field] = sum(recent_values) / len(recent_values)
                        older_avg[field] = sum(older_values) / len(older_values)
                
                stats['trends'] = {}
                for field in recent_avg:
                    if field in older_avg:
                        change = recent_avg[field] - older_avg[field]
                        change_percent = (change / older_avg[field]) * 100 if older_avg[field] != 0 else 0
                        stats['trends'][field] = {
                            'change': round(change, 2),
                            'change_percent': round(change_percent, 2),
                            'direction': 'up' if change > 0 else 'down' if change < 0 else 'stable'
                        }
        else:
            stats = {'total_records': 0}
        
        result = {
            'patient_id': patient_id,
            'metrics': metrics,
            'statistics': stats
        }
        
        # 缓存结果
        redis_manager.set_json(cache_key, result, ttl=300)  # 缓存5分钟
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"获取患者健康指标失败: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': '获取健康指标失败'}), 500

@data_bp.route('/metrics', methods=['POST'])
def create_health_metric():
    """创建新的健康指标记录"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '请提供有效的数据'}), 400
        
        # 数据验证
        is_valid, validated_data, errors = validator.validate_single_record(data, 'health_metrics')
        if not is_valid:
            return jsonify({'error': '数据验证失败', 'details': errors}), 400
        
        # 添加时间戳
        if 'measurement_time' not in validated_data:
            validated_data['measurement_time'] = datetime.utcnow()
        
        if 'created_at' not in validated_data:
            validated_data['created_at'] = datetime.utcnow()
        
        # 插入数据库
        columns = list(validated_data.keys())
        placeholders = ', '.join(['%s'] * len(columns))
        query = f"INSERT INTO health_metrics ({', '.join(columns)}) VALUES ({placeholders})"
        
        result = db_manager.execute_query(
            query, 
            list(validated_data.values()),
            return_lastrowid=True
        )
        
        if result:
            # 清除相关缓存
            patient_id = validated_data['patient_id']
            redis_manager.delete_pattern(f"patient_metrics:{patient_id}:*")
            
            logger.info(f"创建健康指标记录成功: {result}")
            return jsonify({
                'message': '健康指标记录创建成功',
                'metric_id': result,
                'data': validated_data
            }), 201
        else:
            return jsonify({'error': '创建记录失败'}), 500
            
    except Exception as e:
        logger.error(f"创建健康指标失败: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': '创建健康指标失败'}), 500

@data_bp.route('/metrics/batch', methods=['POST'])
def batch_import_metrics():
    """批量导入健康指标"""
    try:
        data = request.get_json()
        if not data or 'records' not in data:
            return jsonify({'error': '请提供有效的批量数据'}), 400
        
        records = data['records']
        if not isinstance(records, list) or len(records) == 0:
            return jsonify({'error': '记录列表不能为空'}), 400
        
        # 批量验证
        validation_result = validator.validate_batch_records(records, 'health_metrics')
        
        if validation_result.valid_count == 0:
            return jsonify({
                'error': '没有有效记录',
                'validation_errors': validation_result.validation_errors
            }), 400
        
        # 批量插入有效记录
        inserted_count = 0
        failed_records = []
        
        for record in validation_result.valid_data:
            try:
                # 添加时间戳
                if 'measurement_time' not in record:
                    record['measurement_time'] = datetime.utcnow()
                if 'created_at' not in record:
                    record['created_at'] = datetime.utcnow()
                
                # 插入数据库
                columns = list(record.keys())
                placeholders = ', '.join(['%s'] * len(columns))
                query = f"INSERT INTO health_metrics ({', '.join(columns)}) VALUES ({placeholders})"
                
                result = db_manager.execute_query(
                    query, 
                    list(record.values()),
                    return_lastrowid=True
                )
                
                if result:
                    inserted_count += 1
                else:
                    failed_records.append({'record': record, 'error': '插入失败'})
                    
            except Exception as e:
                failed_records.append({'record': record, 'error': str(e)})
        
        # 清除缓存
        unique_patients = set(record['patient_id'] for record in validation_result.valid_data)
        for patient_id in unique_patients:
            redis_manager.delete_pattern(f"patient_metrics:{patient_id}:*")
        
        result_data = {
            'message': '批量导入完成',
            'summary': {
                'total_records': len(records),
                'validation_passed': validation_result.valid_count,
                'validation_failed': validation_result.invalid_count,
                'inserted_successfully': inserted_count,
                'insertion_failed': len(failed_records)
            }
        }
        
        # 包含错误详情（如果有的话）
        if validation_result.validation_errors:
            result_data['validation_errors'] = validation_result.validation_errors
        
        if failed_records:
            result_data['insertion_errors'] = failed_records
        
        status_code = 200 if inserted_count > 0 else 400
        return jsonify(result_data), status_code
        
    except Exception as e:
        logger.error(f"批量导入失败: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': '批量导入失败'}), 500

@data_bp.route('/export/patient/<int:patient_id>', methods=['GET'])
def export_patient_data(patient_id: int):
    """导出患者数据为Excel"""
    try:
        # 获取导出格式
        export_format = request.args.get('format', 'excel')  # excel, csv
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # 查询患者基本信息
        patient_query = "SELECT * FROM patients WHERE patient_id = %s"
        patient_data = db_manager.execute_query(patient_query, [patient_id])
        
        if not patient_data:
            return jsonify({'error': '患者不存在'}), 404
        
        patient_info = patient_data[0]
        
        # 查询健康指标
        metrics_query = """
        SELECT * FROM health_metrics 
        WHERE patient_id = %s
        """
        params = [patient_id]
        
        if start_date:
            metrics_query += " AND measurement_time >= %s"
            params.append(start_date)
        
        if end_date:
            metrics_query += " AND measurement_time <= %s"
            params.append(end_date)
        
        metrics_query += " ORDER BY measurement_time DESC"
        
        metrics_data = db_manager.execute_query(metrics_query, params)
        
        # 查询预测结果
        predictions_query = """
        SELECT * FROM ai_prediction_results 
        WHERE patient_id = %s
        ORDER BY created_at DESC
        LIMIT 50
        """
        predictions_data = db_manager.execute_query(predictions_query, [patient_id])
        
        # 创建Excel文件
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # 患者基本信息
            patient_df = pd.DataFrame([patient_info])
            patient_df.to_excel(writer, sheet_name='患者信息', index=False)
            
            # 健康指标
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                # 处理日期时间列
                if 'measurement_time' in metrics_df.columns:
                    metrics_df['measurement_time'] = pd.to_datetime(metrics_df['measurement_time'])
                metrics_df.to_excel(writer, sheet_name='健康指标', index=False)
            
            # AI预测结果
            if predictions_data:
                predictions_df = pd.DataFrame(predictions_data)
                # 处理JSON列
                for col in ['input_data', 'prediction_result', 'recommendations']:
                    if col in predictions_df.columns:
                        predictions_df[col] = predictions_df[col].astype(str)
                predictions_df.to_excel(writer, sheet_name='AI预测结果', index=False)
        
        output.seek(0)
        
        # 生成文件名
        filename = f"患者_{patient_info['username']}_健康数据_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        logger.error(f"导出患者数据失败: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': '导出数据失败'}), 500

@data_bp.route('/preprocessing/test', methods=['POST'])
def test_data_preprocessing():
    """测试数据预处理功能"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '请提供测试数据'}), 400
        
        # 初始化数据处理管道
        pipeline = HealthDataPipeline()
        
        # 获取测试类型
        test_type = data.get('test_type', 'single')
        
        if test_type == 'single':
            # 单条记录测试
            test_record = data.get('record', {})
            if not test_record:
                return jsonify({'error': '请提供测试记录'}), 400
            
            # 数据预处理
            processed_data = pipeline.preprocess_single_record(test_record)
            
            return jsonify({
                'test_type': 'single',
                'original_data': test_record,
                'processed_data': processed_data,
                'preprocessing_steps': pipeline.get_last_processing_steps()
            })
        
        elif test_type == 'batch':
            # 批量数据测试
            test_records = data.get('records', [])
            if not test_records:
                return jsonify({'error': '请提供测试记录列表'}), 400
            
            # 批量预处理
            processed_data = pipeline.preprocess_batch_records(test_records)
            
            return jsonify({
                'test_type': 'batch',
                'original_count': len(test_records),
                'processed_count': len(processed_data['processed_data']),
                'processing_summary': processed_data['summary'],
                'sample_processed_data': processed_data['processed_data'][:5]  # 只返回前5条
            })
        
        elif test_type == 'patient':
            # 患者数据预处理测试
            patient_id = data.get('patient_id')
            if not patient_id:
                return jsonify({'error': '请提供患者ID'}), 400
            
            # 获取患者历史数据
            metrics_query = """
            SELECT * FROM health_metrics 
            WHERE patient_id = %s 
            ORDER BY measurement_time DESC 
            LIMIT 20
            """
            patient_metrics = db_manager.execute_query(metrics_query, [patient_id])
            
            if not patient_metrics:
                return jsonify({'error': '未找到患者数据'}), 404
            
            # 预处理患者数据
            processed_data = pipeline.prepare_training_data(patient_metrics)
            
            return jsonify({
                'test_type': 'patient',
                'patient_id': patient_id,
                'original_records': len(patient_metrics),
                'processed_features': processed_data['features'].shape if hasattr(processed_data['features'], 'shape') else 'N/A',
                'feature_names': processed_data.get('feature_names', []),
                'processing_summary': processed_data.get('summary', {})
            })
        
        else:
            return jsonify({'error': '不支持的测试类型'}), 400
            
    except Exception as e:
        logger.error(f"数据预处理测试失败: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'预处理测试失败: {str(e)}'}), 500

@data_bp.route('/statistics', methods=['GET'])
def get_data_statistics():
    """获取数据统计信息"""
    try:
        # 检查缓存
        cache_key = "data_statistics"
        cached_result = redis_manager.get_json(cache_key)
        if cached_result:
            return jsonify(cached_result)
        
        stats = {}
        
        # 患者统计
        patient_stats_query = """
        SELECT 
            COUNT(*) as total_patients,
            COUNT(CASE WHEN gender = 'M' THEN 1 END) as male_count,
            COUNT(CASE WHEN gender = 'F' THEN 1 END) as female_count,
            AVG(age) as avg_age,
            MIN(age) as min_age,
            MAX(age) as max_age
        FROM patients
        """
        patient_stats = db_manager.execute_query(patient_stats_query)
        stats['patients'] = patient_stats[0] if patient_stats else {}
        
        # 健康指标统计
        metrics_stats_query = """
        SELECT 
            COUNT(*) as total_metrics,
            COUNT(DISTINCT patient_id) as patients_with_metrics,
            AVG(systolic_pressure) as avg_systolic_pressure,
            AVG(diastolic_pressure) as avg_diastolic_pressure,
            AVG(blood_sugar) as avg_blood_sugar,
            AVG(bmi) as avg_bmi,
            COUNT(CASE WHEN measurement_time >= DATE_SUB(NOW(), INTERVAL 30 DAY) THEN 1 END) as recent_measurements
        FROM health_metrics
        WHERE systolic_pressure IS NOT NULL OR diastolic_pressure IS NOT NULL
        """
        metrics_stats = db_manager.execute_query(metrics_stats_query)
        stats['metrics'] = metrics_stats[0] if metrics_stats else {}
        
        # AI预测统计
        prediction_stats_query = """
        SELECT 
            COUNT(*) as total_predictions,
            COUNT(DISTINCT patient_id) as patients_with_predictions,
            AVG(confidence_score) as avg_confidence,
            COUNT(CASE WHEN created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY) THEN 1 END) as recent_predictions,
            prediction_type,
            COUNT(*) as type_count
        FROM ai_prediction_results
        GROUP BY prediction_type
        """
        prediction_stats = db_manager.execute_query(prediction_stats_query)
        
        # 处理预测类型统计
        prediction_summary = {
            'total_predictions': sum(row['type_count'] for row in prediction_stats),
            'patients_with_predictions': len(set(row.get('patient_id', 0) for row in prediction_stats)),
            'by_type': {row['prediction_type']: row['type_count'] for row in prediction_stats}
        }
        stats['predictions'] = prediction_summary
        
        # 数据质量统计
        data_quality_query = """
        SELECT 
            COUNT(*) as total_records,
            COUNT(CASE WHEN systolic_pressure IS NOT NULL AND diastolic_pressure IS NOT NULL THEN 1 END) as complete_bp,
            COUNT(CASE WHEN blood_sugar IS NOT NULL THEN 1 END) as has_blood_sugar,
            COUNT(CASE WHEN bmi IS NOT NULL THEN 1 END) as has_bmi,
            COUNT(CASE WHEN heart_rate IS NOT NULL THEN 1 END) as has_heart_rate
        FROM health_metrics
        """
        quality_stats = db_manager.execute_query(data_quality_query)
        if quality_stats:
            quality_data = quality_stats[0]
            total = quality_data['total_records']
            stats['data_quality'] = {
                'total_records': total,
                'completeness': {
                    'blood_pressure': (quality_data['complete_bp'] / total * 100) if total > 0 else 0,
                    'blood_sugar': (quality_data['has_blood_sugar'] / total * 100) if total > 0 else 0,
                    'bmi': (quality_data['has_bmi'] / total * 100) if total > 0 else 0,
                    'heart_rate': (quality_data['has_heart_rate'] / total * 100) if total > 0 else 0
                }
            }
        
        # 缓存结果
        redis_manager.set_json(cache_key, stats, ttl=600)  # 缓存10分钟
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"获取数据统计失败: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': '获取统计数据失败'}), 500

# 错误处理
@data_bp.errorhandler(404)
def not_found(error):
    return jsonify({'error': '资源未找到'}), 404

@data_bp.errorhandler(500)
def internal_error(error):
    logger.error(f"数据API内部错误: {str(error)}")
    return jsonify({'error': '服务器内部错误'}), 500

# 请求前处理
@data_bp.before_request
def before_request():
    """请求前处理"""
    if request.endpoint and 'health' not in request.endpoint:
        logger.debug(f"数据API请求: {request.method} {request.path}")

# 请求后处理
@data_bp.after_request
def after_request(response):
    """请求后处理"""
    logger.debug(f"数据API响应: {response.status_code}")
    return response
