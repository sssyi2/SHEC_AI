"""
AI预测服务API - 兼容现有SHEC-PSIMS数据库结构
提供健康风险预测、疾病预测等AI服务
"""

import json
import traceback
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify, current_app
from typing import Dict, List, Any, Optional

from utils.database_adapter import db_adapter
from utils.logger import get_logger
from utils.redis_manager import RedisManager
from models.data_processor import HealthDataPipeline
from models.risk_assessment import HealthRiskAssessment
from config.settings import Config

# 创建预测服务蓝图
predict_bp = Blueprint('predict', __name__, url_prefix='/api/predict')

# 初始化组件
logger = get_logger(__name__)
redis_manager = RedisManager()
data_pipeline = HealthDataPipeline()
risk_assessment = HealthRiskAssessment()

@predict_bp.route('/health', methods=['GET'])
def health_check():
    """预测服务健康检查"""
    try:
        return jsonify({
            'status': 'healthy',
            'service': 'prediction_service',
            'timestamp': datetime.utcnow().isoformat(),
            'models_loaded': True,
            'cache_status': 'connected' if redis_manager.ping() else 'disconnected'
        })
    except Exception as e:
        logger.error(f"预测服务健康检查失败: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@predict_bp.route('/risk/<int:patient_id>', methods=['POST'])
def predict_health_risk(patient_id):
    """
    预测患者健康风险
    兼容现有数据库结构
    """
    try:
        # 验证患者存在
        patient_info = db_adapter.get_patient_info(patient_id)
        if not patient_info:
            return jsonify({
                'success': False,
                'message': '患者不存在'
            }), 404
        
        # 检查缓存
        cache_key = f"risk_prediction:{patient_id}"
        cached_result = redis_manager.get(cache_key)
        
        if cached_result:
            logger.info(f"返回缓存的风险预测结果: 患者 {patient_id}")
            return jsonify({
                'success': True,
                'data': cached_result,
                'from_cache': True
            })
        
        # 获取患者最近的健康数据
        health_records = db_adapter.get_health_records(patient_id, limit=20)
        
        if not health_records:
            return jsonify({
                'success': False,
                'message': '患者没有足够的健康数据进行预测'
            }), 400
        
        # 获取医疗记录
        medical_history = db_adapter.get_medical_history(patient_id)
        
        # 准备预测数据
        prediction_input = {
            'patient_info': patient_info,
            'health_records': health_records,
            'medical_history': medical_history,
            'prediction_date': datetime.now().isoformat()
        }
        
        # 使用风险评估模型进行预测
        try:
            risk_result = risk_assessment.predict_health_risk(
                patient_info=patient_info,
                health_data=health_records,
                medical_history=medical_history
            )
        except Exception as model_error:
            logger.error(f"风险评估模型预测失败: {str(model_error)}")
            # 降级为基础规则评估
            risk_result = _basic_risk_assessment(patient_info, health_records)
        
        # 生成建议
        recommendations = _generate_health_recommendations(
            patient_info, health_records, risk_result
        )
        
        # 准备保存的预测结果
        prediction_result = {
            'risk_score': risk_result.get('risk_score', 0.5),
            'risk_level': risk_result.get('risk_level', '中'),
            'risk_factors': risk_result.get('risk_factors', []),
            'confidence': risk_result.get('confidence', 0.7),
            'model_version': risk_result.get('model_version', 'basic_rules_v1.0')
        }
        
        # 保存预测结果到数据库
        prediction_data = {
            'patient_id': patient_id,
            'prediction_type': 'health_risk',
            'model_name': 'HealthRiskAssessment',
            'model_version': prediction_result['model_version'],
            'input_data': json.dumps(prediction_input),
            'prediction_result': json.dumps(prediction_result),
            'confidence_score': prediction_result['confidence'],
            'risk_level': prediction_result['risk_level'],
            'recommendations': json.dumps(recommendations),
            'expires_at': (datetime.now() + timedelta(days=7)).isoformat()
        }
        
        try:
            result_id = db_adapter.save_ai_prediction(patient_id, prediction_data)
            logger.info(f"保存风险预测结果成功: ID {result_id}, 患者 {patient_id}")
        except Exception as save_error:
            logger.error(f"保存风险预测结果失败: {str(save_error)}")
            # 不影响返回结果，只记录错误
        
        # 组装返回结果
        response_data = {
            'patient_id': patient_id,
            'patient_name': patient_info.get('real_name', patient_info.get('username')),
            'prediction': prediction_result,
            'recommendations': recommendations,
            'data_points_used': len(health_records),
            'prediction_timestamp': datetime.now().isoformat()
        }
        
        # 缓存结果（1小时过期）
        redis_manager.set(cache_key, response_data, expire_seconds=3600)
        
        return jsonify({
            'success': True,
            'data': response_data,
            'from_cache': False
        })
        
    except Exception as e:
        logger.error(f"健康风险预测失败: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'message': '健康风险预测失败',
            'error': str(e) if current_app.debug else None
        }), 500

@predict_bp.route('/batch-risk', methods=['POST'])
def predict_batch_risk():
    """批量预测多个患者的健康风险"""
    try:
        data = request.get_json()
        patient_ids = data.get('patient_ids', [])
        
        if not patient_ids:
            return jsonify({
                'success': False,
                'message': '患者ID列表不能为空'
            }), 400
        
        if len(patient_ids) > 50:
            return jsonify({
                'success': False,
                'message': '单次批量预测最多支持50个患者'
            }), 400
        
        results = []
        failed_predictions = []
        
        for patient_id in patient_ids:
            try:
                # 验证患者存在
                patient_info = db_adapter.get_patient_info(patient_id)
                if not patient_info:
                    failed_predictions.append({
                        'patient_id': patient_id,
                        'reason': '患者不存在'
                    })
                    continue
                
                # 获取健康数据
                health_records = db_adapter.get_health_records(patient_id, limit=10)
                if not health_records:
                    failed_predictions.append({
                        'patient_id': patient_id,
                        'reason': '缺少健康数据'
                    })
                    continue
                
                # 进行预测
                try:
                    risk_result = risk_assessment.predict_health_risk(
                        patient_info=patient_info,
                        health_data=health_records,
                        medical_history=[]
                    )
                except Exception:
                    # 降级为基础评估
                    risk_result = _basic_risk_assessment(patient_info, health_records)
                
                # 组装结果
                result = {
                    'patient_id': patient_id,
                    'patient_name': patient_info.get('real_name', patient_info.get('username')),
                    'risk_score': risk_result.get('risk_score', 0.5),
                    'risk_level': risk_result.get('risk_level', '中'),
                    'confidence': risk_result.get('confidence', 0.7),
                    'data_points': len(health_records)
                }
                
                results.append(result)
                
                # 保存预测结果
                prediction_data = {
                    'patient_id': patient_id,
                    'prediction_type': 'health_risk_batch',
                    'model_name': 'HealthRiskAssessment',
                    'model_version': risk_result.get('model_version', 'basic_rules_v1.0'),
                    'input_data': json.dumps({'batch': True}),
                    'prediction_result': json.dumps(risk_result),
                    'confidence_score': risk_result.get('confidence', 0.7),
                    'risk_level': risk_result.get('risk_level', '中'),
                    'recommendations': json.dumps([]),
                    'expires_at': (datetime.now() + timedelta(days=3)).isoformat()
                }
                
                try:
                    db_adapter.save_ai_prediction(patient_id, prediction_data)
                except Exception as save_error:
                    logger.error(f"保存批量预测结果失败 - 患者 {patient_id}: {str(save_error)}")
                
            except Exception as e:
                logger.error(f"批量预测单个患者失败 - 患者 {patient_id}: {str(e)}")
                failed_predictions.append({
                    'patient_id': patient_id,
                    'reason': f'预测过程出错: {str(e)}'
                })
        
        return jsonify({
            'success': True,
            'data': {
                'successful_predictions': len(results),
                'failed_predictions': len(failed_predictions),
                'results': results,
                'failures': failed_predictions,
                'processed_at': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"批量风险预测失败: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'message': '批量风险预测失败',
            'error': str(e) if current_app.debug else None
        }), 500

@predict_bp.route('/disease/<int:patient_id>', methods=['POST'])
def predict_disease_risk(patient_id):
    """预测疾病风险（糖尿病、高血压等）"""
    try:
        data = request.get_json()
        disease_type = data.get('disease_type', 'diabetes')
        
        # 验证患者存在
        patient_info = db_adapter.get_patient_info(patient_id)
        if not patient_info:
            return jsonify({
                'success': False,
                'message': '患者不存在'
            }), 404
        
        # 获取健康数据
        health_records = db_adapter.get_health_records(patient_id, limit=20)
        medical_history = db_adapter.get_medical_history(patient_id)
        
        if not health_records:
            return jsonify({
                'success': False,
                'message': '患者没有足够的健康数据进行疾病风险预测'
            }), 400
        
        # 根据疾病类型进行预测
        if disease_type == 'diabetes':
            prediction_result = _predict_diabetes_risk(patient_info, health_records, medical_history)
        elif disease_type == 'hypertension':
            prediction_result = _predict_hypertension_risk(patient_info, health_records, medical_history)
        elif disease_type == 'cardiovascular':
            prediction_result = _predict_cardiovascular_risk(patient_info, health_records, medical_history)
        else:
            return jsonify({
                'success': False,
                'message': f'不支持的疾病类型: {disease_type}'
            }), 400
        
        # 生成针对性建议
        recommendations = _generate_disease_recommendations(
            disease_type, patient_info, health_records, prediction_result
        )
        
        # 保存预测结果
        prediction_data = {
            'patient_id': patient_id,
            'prediction_type': f'{disease_type}_risk',
            'model_name': f'{disease_type.capitalize()}RiskModel',
            'model_version': '1.0',
            'input_data': json.dumps({
                'disease_type': disease_type,
                'data_points': len(health_records)
            }),
            'prediction_result': json.dumps(prediction_result),
            'confidence_score': prediction_result.get('confidence', 0.7),
            'risk_level': prediction_result.get('risk_level', '中'),
            'recommendations': json.dumps(recommendations),
            'expires_at': (datetime.now() + timedelta(days=30)).isoformat()
        }
        
        try:
            result_id = db_adapter.save_ai_prediction(patient_id, prediction_data)
            logger.info(f"保存疾病风险预测结果成功: ID {result_id}, 患者 {patient_id}, 疾病 {disease_type}")
        except Exception as save_error:
            logger.error(f"保存疾病风险预测结果失败: {str(save_error)}")
        
        # 组装返回结果
        response_data = {
            'patient_id': patient_id,
            'patient_name': patient_info.get('real_name', patient_info.get('username')),
            'disease_type': disease_type,
            'prediction': prediction_result,
            'recommendations': recommendations,
            'prediction_timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'data': response_data
        })
        
    except Exception as e:
        logger.error(f"疾病风险预测失败: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'message': '疾病风险预测失败',
            'error': str(e) if current_app.debug else None
        }), 500

@predict_bp.route('/models', methods=['GET'])
def get_available_models():
    """获取可用的AI模型列表"""
    try:
        model_type = request.args.get('model_type')
        
        # 使用适配器获取模型列表
        models = db_adapter.get_active_ai_models(model_type)
        
        return jsonify({
            'success': True,
            'data': models,
            'count': len(models)
        })
        
    except Exception as e:
        logger.error(f"获取AI模型列表失败: {str(e)}")
        return jsonify({
            'success': False,
            'message': '获取AI模型列表失败',
            'error': str(e)
        }), 500

# 辅助函数

def _basic_risk_assessment(patient_info: Dict, health_records: List[Dict]) -> Dict:
    """基础风险评估（规则引擎）"""
    risk_score = 0.5
    risk_factors = []
    
    # 年龄因素
    age = patient_info.get('age', 0)
    if age > 65:
        risk_score += 0.2
        risk_factors.append('年龄超过65岁')
    elif age > 50:
        risk_score += 0.1
        risk_factors.append('年龄超过50岁')
    
    # 最近的健康指标
    if health_records:
        latest_record = health_records[0]
        
        # 血压
        systolic = latest_record.get('systolic_pressure', 0)
        diastolic = latest_record.get('diastolic_pressure', 0)
        
        if systolic > 160 or diastolic > 100:
            risk_score += 0.3
            risk_factors.append('血压严重偏高')
        elif systolic > 140 or diastolic > 90:
            risk_score += 0.2
            risk_factors.append('血压偏高')
        
        # 血糖
        blood_sugar = latest_record.get('blood_sugar', 0)
        if blood_sugar > 11.1:
            risk_score += 0.3
            risk_factors.append('血糖严重偏高')
        elif blood_sugar > 7.0:
            risk_score += 0.2
            risk_factors.append('血糖偏高')
        
        # BMI
        bmi = latest_record.get('bmi', 0)
        if bmi > 30:
            risk_score += 0.15
            risk_factors.append('肥胖')
        elif bmi > 25:
            risk_score += 0.1
            risk_factors.append('超重')
        
        # 生活方式
        if latest_record.get('smoking_status') in ['current', 'daily']:
            risk_score += 0.15
            risk_factors.append('吸烟')
        
        if latest_record.get('drinking_status') == 'heavy':
            risk_score += 0.1
            risk_factors.append('过度饮酒')
        
        if latest_record.get('exercise_frequency') in ['never', 'rarely']:
            risk_score += 0.1
            risk_factors.append('缺乏运动')
    
    # 确保风险评分在合理范围内
    risk_score = max(0.0, min(1.0, risk_score))
    
    # 确定风险等级
    if risk_score >= 0.8:
        risk_level = '极高'
    elif risk_score >= 0.6:
        risk_level = '高'
    elif risk_score >= 0.4:
        risk_level = '中'
    else:
        risk_level = '低'
    
    return {
        'risk_score': round(risk_score, 3),
        'risk_level': risk_level,
        'risk_factors': risk_factors,
        'confidence': 0.7,
        'model_version': 'basic_rules_v1.0'
    }

def _generate_health_recommendations(patient_info: Dict, health_records: List[Dict], risk_result: Dict) -> List[Dict]:
    """生成健康建议"""
    recommendations = []
    risk_factors = risk_result.get('risk_factors', [])
    
    # 基于风险因素生成建议
    if '血压严重偏高' in risk_factors or '血压偏高' in risk_factors:
        recommendations.append({
            'category': '血压管理',
            'priority': 'high',
            'title': '监测和控制血压',
            'content': '建议定期监测血压，控制盐分摄入，适量运动，必要时遵医嘱用药',
            'actionable': True
        })
    
    if '血糖严重偏高' in risk_factors or '血糖偏高' in risk_factors:
        recommendations.append({
            'category': '血糖管理',
            'priority': 'high',
            'title': '控制血糖水平',
            'content': '建议控制碳水化合物摄入，规律进餐，监测血糖变化',
            'actionable': True
        })
    
    if '肥胖' in risk_factors or '超重' in risk_factors:
        recommendations.append({
            'category': '体重管理',
            'priority': 'medium',
            'title': '控制体重',
            'content': '建议制定合理的饮食计划，增加有氧运动，逐步减重',
            'actionable': True
        })
    
    if '吸烟' in risk_factors:
        recommendations.append({
            'category': '生活方式',
            'priority': 'high',
            'title': '戒烟',
            'content': '强烈建议戒烟，可寻求专业戒烟指导',
            'actionable': True
        })
    
    if '缺乏运动' in risk_factors:
        recommendations.append({
            'category': '运动锻炼',
            'priority': 'medium',
            'title': '增加运动量',
            'content': '建议每周至少150分钟中等强度有氧运动',
            'actionable': True
        })
    
    # 通用建议
    recommendations.append({
        'category': '定期检查',
        'priority': 'medium',
        'title': '定期体检',
        'content': '建议定期进行健康检查，及时发现和预防健康问题',
        'actionable': True
    })
    
    return recommendations

def _predict_diabetes_risk(patient_info: Dict, health_records: List[Dict], medical_history: List[Dict]) -> Dict:
    """预测糖尿病风险"""
    risk_score = 0.1
    risk_factors = []
    
    # 年龄因素
    age = patient_info.get('age', 0)
    if age > 45:
        risk_score += 0.2
        risk_factors.append('年龄超过45岁')
    
    # 血糖历史
    if health_records:
        high_glucose_count = sum(1 for record in health_records[:10] 
                               if record.get('blood_sugar', 0) > 7.0)
        
        if high_glucose_count >= 5:
            risk_score += 0.4
            risk_factors.append('血糖长期偏高')
        elif high_glucose_count >= 2:
            risk_score += 0.2
            risk_factors.append('血糖偶尔偏高')
        
        # BMI
        latest_record = health_records[0]
        bmi = latest_record.get('bmi', 0)
        if bmi > 30:
            risk_score += 0.25
            risk_factors.append('肥胖（BMI>30）')
        elif bmi > 25:
            risk_score += 0.15
            risk_factors.append('超重（BMI>25）')
    
    # 家族史（从医疗记录中查找）
    for record in medical_history:
        diagnosis = record.get('diagnosis', '').lower()
        if 'diabetes' in diagnosis or '糖尿病' in diagnosis:
            risk_score += 0.3
            risk_factors.append('糖尿病病史')
            break
    
    risk_score = max(0.0, min(1.0, risk_score))
    
    if risk_score >= 0.7:
        risk_level = '高'
    elif risk_score >= 0.4:
        risk_level = '中'
    else:
        risk_level = '低'
    
    return {
        'risk_score': round(risk_score, 3),
        'risk_level': risk_level,
        'risk_factors': risk_factors,
        'confidence': 0.75,
        'disease_type': 'diabetes'
    }

def _predict_hypertension_risk(patient_info: Dict, health_records: List[Dict], medical_history: List[Dict]) -> Dict:
    """预测高血压风险"""
    risk_score = 0.1
    risk_factors = []
    
    # 年龄和性别
    age = patient_info.get('age', 0)
    gender = patient_info.get('gender', '')
    
    if age > 60:
        risk_score += 0.25
        risk_factors.append('年龄超过60岁')
    elif age > 40:
        risk_score += 0.15
        risk_factors.append('年龄超过40岁')
    
    # 血压历史
    if health_records:
        high_bp_count = sum(1 for record in health_records[:10] 
                           if (record.get('systolic_pressure', 0) > 140 or 
                               record.get('diastolic_pressure', 0) > 90))
        
        if high_bp_count >= 6:
            risk_score += 0.5
            risk_factors.append('血压持续偏高')
        elif high_bp_count >= 3:
            risk_score += 0.3
            risk_factors.append('血压经常偏高')
        elif high_bp_count >= 1:
            risk_score += 0.15
            risk_factors.append('血压偶尔偏高')
        
        # 其他因素
        latest_record = health_records[0]
        
        # BMI
        bmi = latest_record.get('bmi', 0)
        if bmi > 30:
            risk_score += 0.2
            risk_factors.append('肥胖')
        
        # 生活方式
        if latest_record.get('smoking_status') in ['current', 'daily']:
            risk_score += 0.15
            risk_factors.append('吸烟')
        
        if latest_record.get('exercise_frequency') in ['never', 'rarely']:
            risk_score += 0.1
            risk_factors.append('缺乏运动')
    
    risk_score = max(0.0, min(1.0, risk_score))
    
    if risk_score >= 0.7:
        risk_level = '高'
    elif risk_score >= 0.4:
        risk_level = '中'
    else:
        risk_level = '低'
    
    return {
        'risk_score': round(risk_score, 3),
        'risk_level': risk_level,
        'risk_factors': risk_factors,
        'confidence': 0.8,
        'disease_type': 'hypertension'
    }

def _predict_cardiovascular_risk(patient_info: Dict, health_records: List[Dict], medical_history: List[Dict]) -> Dict:
    """预测心血管疾病风险"""
    risk_score = 0.1
    risk_factors = []
    
    # 年龄和性别
    age = patient_info.get('age', 0)
    gender = patient_info.get('gender', '')
    
    if age > 65:
        risk_score += 0.3
        risk_factors.append('年龄超过65岁')
    elif age > 50:
        risk_score += 0.2
        risk_factors.append('年龄超过50岁')
    
    if gender == 'male' and age > 45:
        risk_score += 0.1
        risk_factors.append('男性且年龄超过45岁')
    elif gender == 'female' and age > 55:
        risk_score += 0.1
        risk_factors.append('女性且年龄超过55岁')
    
    # 综合健康指标
    if health_records:
        latest_record = health_records[0]
        
        # 高血压
        systolic = latest_record.get('systolic_pressure', 0)
        diastolic = latest_record.get('diastolic_pressure', 0)
        if systolic > 140 or diastolic > 90:
            risk_score += 0.25
            risk_factors.append('高血压')
        
        # 血糖
        blood_sugar = latest_record.get('blood_sugar', 0)
        if blood_sugar > 7.0:
            risk_score += 0.2
            risk_factors.append('血糖偏高')
        
        # 肥胖
        bmi = latest_record.get('bmi', 0)
        if bmi > 30:
            risk_score += 0.2
            risk_factors.append('肥胖')
        
        # 生活方式
        if latest_record.get('smoking_status') in ['current', 'daily']:
            risk_score += 0.25
            risk_factors.append('吸烟')
        
        if latest_record.get('drinking_status') == 'heavy':
            risk_score += 0.1
            risk_factors.append('过度饮酒')
        
        if latest_record.get('exercise_frequency') in ['never', 'rarely']:
            risk_score += 0.15
            risk_factors.append('缺乏运动')
    
    risk_score = max(0.0, min(1.0, risk_score))
    
    if risk_score >= 0.7:
        risk_level = '高'
    elif risk_score >= 0.4:
        risk_level = '中'
    else:
        risk_level = '低'
    
    return {
        'risk_score': round(risk_score, 3),
        'risk_level': risk_level,
        'risk_factors': risk_factors,
        'confidence': 0.75,
        'disease_type': 'cardiovascular'
    }

def _generate_disease_recommendations(disease_type: str, patient_info: Dict, 
                                    health_records: List[Dict], prediction_result: Dict) -> List[Dict]:
    """生成疾病特定的建议"""
    recommendations = []
    risk_level = prediction_result.get('risk_level', '中')
    
    if disease_type == 'diabetes':
        if risk_level in ['高', '中']:
            recommendations.extend([
                {
                    'category': '饮食管理',
                    'priority': 'high',
                    'title': '控制饮食',
                    'content': '减少高糖、高淀粉食物摄入，定时定量进餐',
                    'actionable': True
                },
                {
                    'category': '血糖监测',
                    'priority': 'high',
                    'title': '定期监测血糖',
                    'content': '建议每天监测血糖，记录饮食和血糖变化',
                    'actionable': True
                }
            ])
    
    elif disease_type == 'hypertension':
        if risk_level in ['高', '中']:
            recommendations.extend([
                {
                    'category': '饮食控制',
                    'priority': 'high',
                    'title': '低盐饮食',
                    'content': '减少钠盐摄入，每日盐分不超过6克',
                    'actionable': True
                },
                {
                    'category': '血压监测',
                    'priority': 'high',
                    'title': '定期测量血压',
                    'content': '建议每天定时测量血压，记录血压变化趋势',
                    'actionable': True
                }
            ])
    
    elif disease_type == 'cardiovascular':
        if risk_level in ['高', '中']:
            recommendations.extend([
                {
                    'category': '综合管理',
                    'priority': 'high',
                    'title': '心血管风险管理',
                    'content': '控制血压、血脂、血糖，戒烟限酒，规律运动',
                    'actionable': True
                },
                {
                    'category': '定期检查',
                    'priority': 'high',
                    'title': '心血管检查',
                    'content': '建议定期进行心电图、血脂等相关检查',
                    'actionable': True
                }
            ])
    
    return recommendations

# 错误处理器
@predict_bp.errorhandler(Exception)
def handle_exception(e):
    """统一异常处理"""
    logger.error(f"预测服务异常: {str(e)}", exc_info=True)
    return jsonify({
        'success': False,
        'message': '预测服务异常',
        'error': str(e) if current_app.debug else None
    }), 500
