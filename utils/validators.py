# 健康数据验证模块
# 使用 marshmallow 进行数据验证和序列化

from marshmallow import Schema, fields, validates, validates_schema, ValidationError, post_load, validate
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, date
import re
import numpy as np
from decimal import Decimal

from utils.logger import get_logger

logger = get_logger(__name__)

# 自定义验证器
class CustomValidators:
    """自定义验证器集合"""
    
    @staticmethod
    def validate_blood_pressure(systolic: int, diastolic: int) -> bool:
        """验证血压值的合理性"""
        if not (70 <= systolic <= 200) or not (40 <= diastolic <= 130):
            return False
        if systolic <= diastolic:  # 收缩压应该大于舒张压
            return False
        return True
    
    @staticmethod
    def validate_phone(phone: str) -> bool:
        """验证手机号码格式"""
        pattern = r'^1[3-9]\d{9}$'
        return bool(re.match(pattern, phone))
    
    @staticmethod
    def validate_id_card(id_card: str) -> bool:
        """验证身份证号码格式"""
        if len(id_card) != 18:
            return False
        pattern = r'^\d{17}[\dX]$'
        return bool(re.match(pattern, id_card))

# 基础用户信息验证
class UserInfoSchema(Schema):
    """用户基础信息验证模式"""
    
    user_id = fields.Integer(required=False, allow_none=True)
    username = fields.String(required=True, validate=validate.Length(min=2, max=50))
    real_name = fields.String(required=True, validate=validate.Length(min=2, max=20))
    age = fields.Integer(required=True, validate=validate.Range(min=0, max=150))
    gender = fields.String(required=True, validate=validate.OneOf(['M', 'F', '男', '女']))
    phone = fields.String(required=True)
    email = fields.Email(required=False, allow_none=True)
    id_card = fields.String(required=False, allow_none=True)
    created_at = fields.DateTime(required=False, allow_none=True)
    
    @validates('phone')
    def validate_phone(self, value):
        if not CustomValidators.validate_phone(value):
            raise ValidationError('手机号码格式不正确')
    
    @validates('id_card')
    def validate_id_card(self, value):
        if value and not CustomValidators.validate_id_card(value):
            raise ValidationError('身份证号码格式不正确')
    
    @post_load
    def process_gender(self, data, **kwargs):
        """处理性别字段"""
        gender_map = {'男': 'M', '女': 'F'}
        if data.get('gender') in gender_map:
            data['gender'] = gender_map[data['gender']]
        return data

# 健康指标验证
class HealthMetricsSchema(Schema):
    """健康指标验证模式"""
    
    metric_id = fields.Integer(required=False, allow_none=True)
    patient_id = fields.Integer(required=True)
    systolic_pressure = fields.Decimal(
        required=False, 
        allow_none=True,
        validate=validate.Range(min=70, max=200),
        places=1
    )
    diastolic_pressure = fields.Decimal(
        required=False,
        allow_none=True, 
        validate=validate.Range(min=40, max=130),
        places=1
    )
    blood_sugar = fields.Decimal(
        required=False,
        allow_none=True,
        validate=validate.Range(min=3.0, max=30.0),
        places=2
    )
    blood_sugar_type = fields.String(
        required=False,
        allow_none=True,
        validate=validate.OneOf(['空腹', '餐后2小时', '随机'])
    )
    bmi = fields.Decimal(
        required=False,
        allow_none=True,
        validate=validate.Range(min=10.0, max=50.0),
        places=2
    )
    weight = fields.Decimal(
        required=False,
        allow_none=True,
        validate=validate.Range(min=20.0, max=300.0),
        places=1
    )
    height = fields.Decimal(
        required=False,
        allow_none=True,
        validate=validate.Range(min=50.0, max=250.0),
        places=1
    )
    heart_rate = fields.Integer(
        required=False,
        allow_none=True,
        validate=validate.Range(min=30, max=220)
    )
    body_temperature = fields.Decimal(
        required=False,
        allow_none=True,
        validate=validate.Range(min=35.0, max=42.0),
        places=1
    )
    exercise_frequency = fields.Integer(
        required=False,
        allow_none=True,
        validate=validate.Range(min=0, max=7)
    )
    sleep_hours = fields.Integer(
        required=False,
        allow_none=True,
        validate=validate.Range(min=0, max=24)
    )
    smoking_status = fields.String(
        required=False,
        allow_none=True,
        validate=validate.OneOf(['从不', '偶尔', '经常', '已戒'])
    )
    drinking_status = fields.String(
        required=False,
        allow_none=True,
        validate=validate.OneOf(['从不', '偶尔', '经常', '已戒'])
    )
    medication_usage = fields.String(required=False, allow_none=True)
    other_metrics = fields.Raw(required=False, allow_none=True)
    measurement_time = fields.DateTime(required=False, allow_none=True)
    created_at = fields.DateTime(required=False, allow_none=True)
    
    @validates_schema
    def validate_blood_pressure(self, data, **kwargs):
        """验证血压组合的合理性"""
        systolic = data.get('systolic_pressure')
        diastolic = data.get('diastolic_pressure')
        
        if systolic and diastolic:
            if not CustomValidators.validate_blood_pressure(float(systolic), float(diastolic)):
                raise ValidationError('血压值不合理：收缩压应大于舒张压且在正常范围内')
    
    @validates_schema
    def validate_bmi_weight_height(self, data, **kwargs):
        """验证BMI与身高体重的一致性"""
        bmi = data.get('bmi')
        weight = data.get('weight')
        height = data.get('height')
        
        if weight and height and bmi:
            height_m = float(height) / 100  # 转换为米
            calculated_bmi = float(weight) / (height_m ** 2)
            
            # 允许5%的误差
            if abs(calculated_bmi - float(bmi)) > calculated_bmi * 0.05:
                raise ValidationError('BMI与身高体重不匹配')
    
    @post_load
    def calculate_missing_bmi(self, data, **kwargs):
        """自动计算缺失的BMI"""
        if not data.get('bmi') and data.get('weight') and data.get('height'):
            height_m = float(data['height']) / 100
            calculated_bmi = float(data['weight']) / (height_m ** 2)
            data['bmi'] = round(Decimal(str(calculated_bmi)), 2)
        return data

# AI预测结果验证
class PredictionResultSchema(Schema):
    """AI预测结果验证模式"""
    
    result_id = fields.Integer(required=False, allow_none=True)
    patient_id = fields.Integer(required=True)
    prediction_type = fields.String(
        required=True,
        validate=validate.OneOf(['health_trend', 'risk_assessment', 'disease_prediction'])
    )
    model_name = fields.String(required=True, validate=validate.Length(min=1, max=100))
    model_version = fields.String(required=True, validate=validate.Length(min=1, max=50))
    input_data = fields.Raw(required=True)
    prediction_result = fields.Raw(required=True)
    confidence_score = fields.Decimal(
        required=True,
        validate=validate.Range(min=0.0, max=1.0),
        places=4
    )
    risk_level = fields.String(
        required=False,
        allow_none=True,
        validate=validate.OneOf(['低', '中', '高', '极高'])
    )
    recommendations = fields.List(fields.String(), required=False, allow_none=True)
    expires_at = fields.DateTime(required=False, allow_none=True)
    created_at = fields.DateTime(required=False, allow_none=True)
    
    @validates('input_data')
    def validate_input_data(self, value):
        """验证输入数据格式"""
        if not isinstance(value, dict):
            raise ValidationError('输入数据必须是字典格式')
        
        required_fields = ['age', 'gender']
        missing_fields = [field for field in required_fields if field not in value]
        if missing_fields:
            raise ValidationError(f'输入数据缺少必要字段: {missing_fields}')
    
    @validates('prediction_result')
    def validate_prediction_result(self, value):
        """验证预测结果格式"""
        if not isinstance(value, dict):
            raise ValidationError('预测结果必须是字典格式')

# AI模型信息验证
class AIModelSchema(Schema):
    """AI模型信息验证模式"""
    
    model_id = fields.Integer(required=False, allow_none=True)
    model_name = fields.String(required=True, validate=validate.Length(min=1, max=100))
    model_type = fields.String(
        required=True,
        validate=validate.OneOf(['pytorch', 'lightgbm', 'xgboost', 'sklearn'])
    )
    version = fields.String(required=True, validate=validate.Length(min=1, max=50))
    description = fields.String(required=False, allow_none=True)
    model_path = fields.String(required=True, validate=validate.Length(min=1, max=500))
    config_data = fields.Raw(required=False, allow_none=True)
    performance_metrics = fields.Raw(required=False, allow_none=True)
    training_data_info = fields.Raw(required=False, allow_none=True)
    feature_importance = fields.Raw(required=False, allow_none=True)
    is_active = fields.Boolean(required=False, allow_none=True, default=True)
    created_at = fields.DateTime(required=False, allow_none=True)
    updated_at = fields.DateTime(required=False, allow_none=True)

# 预测任务验证
class PredictionTaskSchema(Schema):
    """预测任务验证模式"""
    
    task_id = fields.Integer(required=False, allow_none=True)
    patient_id = fields.Integer(required=True)
    prediction_type = fields.String(
        required=True,
        validate=validate.OneOf(['health_trend', 'risk_assessment', 'disease_prediction'])
    )
    task_status = fields.String(
        required=True,
        validate=validate.OneOf(['pending', 'running', 'completed', 'failed'])
    )
    priority = fields.Integer(
        required=False,
        allow_none=True,
        validate=validate.Range(min=1, max=10),
        default=5
    )
    scheduled_time = fields.DateTime(required=False, allow_none=True)
    started_at = fields.DateTime(required=False, allow_none=True)
    completed_at = fields.DateTime(required=False, allow_none=True)
    error_message = fields.String(required=False, allow_none=True)
    result_id = fields.Integer(required=False, allow_none=True)
    created_at = fields.DateTime(required=False, allow_none=True)

# 批量数据验证
class BatchValidationResult:
    """批量验证结果"""
    
    def __init__(self):
        self.valid_data: List[Dict] = []
        self.invalid_data: List[Dict] = []
        self.validation_errors: List[Dict] = []
        self.total_records = 0
        self.valid_count = 0
        self.invalid_count = 0
    
    def add_valid_record(self, record: Dict):
        """添加有效记录"""
        self.valid_data.append(record)
        self.valid_count += 1
    
    def add_invalid_record(self, record: Dict, errors: Dict):
        """添加无效记录"""
        self.invalid_data.append(record)
        self.validation_errors.append({
            'record': record,
            'errors': errors
        })
        self.invalid_count += 1
    
    def get_summary(self) -> Dict:
        """获取验证摘要"""
        return {
            'total_records': self.total_records,
            'valid_count': self.valid_count,
            'invalid_count': self.invalid_count,
            'success_rate': self.valid_count / self.total_records if self.total_records > 0 else 0,
            'has_errors': len(self.validation_errors) > 0
        }

class HealthDataValidator:
    """健康数据验证器主类"""
    
    def __init__(self):
        self.schemas = {
            'user_info': UserInfoSchema(),
            'health_metrics': HealthMetricsSchema(),
            'prediction_result': PredictionResultSchema(),
            'ai_model': AIModelSchema(),
            'prediction_task': PredictionTaskSchema()
        }
        self.logger = logger
    
    def validate_single_record(self, data: Dict, schema_name: str) -> Tuple[bool, Dict, Dict]:
        """验证单条记录"""
        try:
            schema = self.schemas.get(schema_name)
            if not schema:
                raise ValueError(f"未知的验证模式: {schema_name}")
            
            validated_data = schema.load(data)
            return True, validated_data, {}
            
        except ValidationError as e:
            return False, {}, e.messages
        except Exception as e:
            return False, {}, {"system_error": str(e)}
    
    def validate_batch_records(self, data_list: List[Dict], 
                             schema_name: str) -> BatchValidationResult:
        """批量验证记录"""
        result = BatchValidationResult()
        result.total_records = len(data_list)
        
        for record in data_list:
            is_valid, validated_data, errors = self.validate_single_record(record, schema_name)
            
            if is_valid:
                result.add_valid_record(validated_data)
            else:
                result.add_invalid_record(record, errors)
        
        self.logger.info(f"批量验证完成: {result.get_summary()}")
        return result
    
    def validate_health_data_for_prediction(self, health_data: Dict) -> Tuple[bool, Dict, List[str]]:
        """验证用于预测的健康数据"""
        errors = []
        
        # 基本数据完整性检查
        required_fields = ['patient_id', 'age', 'gender']
        for field in required_fields:
            if field not in health_data or health_data[field] is None:
                errors.append(f"缺少必要字段: {field}")
        
        # 数据类型检查
        if health_data.get('age') and not isinstance(health_data['age'], (int, float)):
            errors.append("年龄必须是数字")
        
        if health_data.get('gender') and health_data['gender'] not in ['M', 'F', '男', '女']:
            errors.append("性别值无效")
        
        # 健康指标检查
        if health_data.get('systolic_pressure') or health_data.get('diastolic_pressure'):
            sys_bp = health_data.get('systolic_pressure', 0)
            dia_bp = health_data.get('diastolic_pressure', 0)
            
            if sys_bp and dia_bp:
                if not CustomValidators.validate_blood_pressure(sys_bp, dia_bp):
                    errors.append("血压值不合理")
        
        # 数据范围检查
        if health_data.get('blood_sugar'):
            if not (3.0 <= float(health_data['blood_sugar']) <= 30.0):
                errors.append("血糖值超出正常范围")
        
        if health_data.get('bmi'):
            if not (10.0 <= float(health_data['bmi']) <= 50.0):
                errors.append("BMI值超出正常范围")
        
        is_valid = len(errors) == 0
        
        if is_valid:
            # 数据标准化
            standardized_data = self._standardize_health_data(health_data)
            return True, standardized_data, []
        else:
            return False, {}, errors
    
    def _standardize_health_data(self, health_data: Dict) -> Dict:
        """标准化健康数据"""
        standardized = health_data.copy()
        
        # 性别标准化
        if 'gender' in standardized:
            gender_map = {'男': 'M', '女': 'F'}
            standardized['gender'] = gender_map.get(standardized['gender'], standardized['gender'])
        
        # 数值类型转换
        numeric_fields = ['age', 'systolic_pressure', 'diastolic_pressure', 
                         'blood_sugar', 'bmi', 'weight', 'height', 'heart_rate']
        
        for field in numeric_fields:
            if field in standardized and standardized[field] is not None:
                try:
                    standardized[field] = float(standardized[field])
                except (ValueError, TypeError):
                    pass
        
        return standardized
    
    def validate_prediction_input(self, prediction_input: Dict) -> Tuple[bool, Dict, List[str]]:
        """验证预测输入参数"""
        errors = []
        
        # 检查预测类型
        prediction_type = prediction_input.get('prediction_type')
        if not prediction_type:
            errors.append("缺少预测类型")
        elif prediction_type not in ['health_trend', 'risk_assessment', 'disease_prediction']:
            errors.append("预测类型无效")
        
        # 检查患者ID
        patient_id = prediction_input.get('patient_id')
        if not patient_id:
            errors.append("缺少患者ID")
        elif not isinstance(patient_id, int) or patient_id <= 0:
            errors.append("患者ID必须是正整数")
        
        # 检查健康数据
        health_data = prediction_input.get('health_data', {})
        if not health_data:
            errors.append("缺少健康数据")
        else:
            is_valid, validated_health, health_errors = self.validate_health_data_for_prediction(health_data)
            if not is_valid:
                errors.extend(health_errors)
            else:
                prediction_input['health_data'] = validated_health
        
        is_valid = len(errors) == 0
        return is_valid, prediction_input if is_valid else {}, errors
    
    def get_validation_statistics(self, data_list: List[Dict], 
                                schema_name: str) -> Dict:
        """获取数据验证统计信息"""
        result = self.validate_batch_records(data_list, schema_name)
        
        # 详细统计
        error_stats = {}
        for error_record in result.validation_errors:
            for field, error_list in error_record['errors'].items():
                if field not in error_stats:
                    error_stats[field] = {}
                
                for error in error_list:
                    if error not in error_stats[field]:
                        error_stats[field][error] = 0
                    error_stats[field][error] += 1
        
        return {
            'summary': result.get_summary(),
            'error_statistics': error_stats,
            'most_common_errors': self._get_most_common_errors(error_stats),
            'data_quality_score': result.valid_count / result.total_records if result.total_records > 0 else 0
        }
    
    def _get_most_common_errors(self, error_stats: Dict, top_n: int = 5) -> List[Tuple[str, str, int]]:
        """获取最常见的错误"""
        all_errors = []
        
        for field, field_errors in error_stats.items():
            for error, count in field_errors.items():
                all_errors.append((field, error, count))
        
        # 按错误次数排序
        all_errors.sort(key=lambda x: x[2], reverse=True)
        return all_errors[:top_n]

# 使用示例
def example_usage():
    """数据验证使用示例"""
    
    validator = HealthDataValidator()
    
    # 1. 单条记录验证
    print("=== 单条记录验证 ===")
    user_data = {
        "username": "test_user",
        "real_name": "测试用户", 
        "age": 35,
        "gender": "M",
        "phone": "13812345678"
    }
    
    is_valid, validated_data, errors = validator.validate_single_record(
        user_data, 'user_info'
    )
    print(f"验证结果: {is_valid}")
    if not is_valid:
        print(f"错误: {errors}")
    
    # 2. 健康数据验证
    print("\n=== 健康数据验证 ===")
    health_data = {
        "patient_id": 1,
        "age": 45,
        "gender": "M",
        "systolic_pressure": 140,
        "diastolic_pressure": 90,
        "blood_sugar": 8.5,
        "bmi": 25.3
    }
    
    is_valid, validated_health, health_errors = validator.validate_health_data_for_prediction(health_data)
    print(f"健康数据验证结果: {is_valid}")
    if not is_valid:
        print(f"错误: {health_errors}")
    
    # 3. 批量验证
    print("\n=== 批量验证 ===")
    batch_data = [user_data, {"username": "invalid", "age": "abc"}]
    result = validator.validate_batch_records(batch_data, 'user_info')
    print(f"批量验证摘要: {result.get_summary()}")

# 预测相关验证模式
class PredictionInputSchema(Schema):
    """预测输入数据验证模式"""
    
    # 时序数据
    sequence_data = fields.List(fields.List(fields.Float()), missing=None)
    
    # 静态特征
    features = fields.Dict(missing=None)
    
    # 用户上下文
    user_context = fields.Dict(missing={})
    
    # 模型名称
    model_name = fields.String(missing='default')
    
    @validates_schema
    def validate_input_type(self, data, **kwargs):
        """验证输入数据类型"""
        sequence_data = data.get('sequence_data')
        features = data.get('features')
        
        if not sequence_data and not features:
            raise ValidationError('必须提供sequence_data或features中的至少一个')

class RiskAssessmentInputSchema(Schema):
    """风险评估输入数据验证模式"""
    
    features = fields.Dict(required=True)
    user_context = fields.Dict(missing={})
    model_name = fields.String(missing='default_risk_assessment')
    
    @validates('features')
    def validate_features(self, value):
        """验证特征数据"""
        if not isinstance(value, dict):
            raise ValidationError('特征数据必须是字典格式')
        
        # 检查必要的特征
        required_features = ['age', 'gender']
        for feature in required_features:
            if feature not in value:
                raise ValidationError(f'缺少必要特征: {feature}')

class BatchPredictionInputSchema(Schema):
    """批量预测输入数据验证模式"""
    
    batch_data = fields.List(fields.Dict(), required=True, validate=lambda x: 1 <= len(x) <= 100)
    prediction_type = fields.String(required=True, validate=lambda x: x in ['health_indicators', 'disease_risk'])
    model_name = fields.String(missing=None)
    
    @validates('batch_data')
    def validate_batch_data(self, value):
        """验证批量数据"""
        if not value:
            raise ValidationError('批量数据不能为空')
        
        for i, item in enumerate(value):
            if not isinstance(item, dict):
                raise ValidationError(f'批量数据第{i+1}项必须是字典格式')

def validate_prediction_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """验证预测输入数据"""
    try:
        schema = PredictionInputSchema()
        result = schema.load(data)
        
        # 额外验证
        if result.get('sequence_data'):
            sequence_data = result['sequence_data']
            if not isinstance(sequence_data, list) or not sequence_data:
                return {'valid': False, 'errors': 'sequence_data必须是非空列表'}
            
            # 检查时序数据格式
            for i, seq in enumerate(sequence_data):
                if not isinstance(seq, list):
                    return {'valid': False, 'errors': f'sequence_data第{i+1}行必须是列表'}
                
                for j, value in enumerate(seq):
                    if not isinstance(value, (int, float)):
                        return {'valid': False, 'errors': f'sequence_data[{i}][{j}]必须是数字'}
                    
                    if np.isnan(value) or np.isinf(value):
                        return {'valid': False, 'errors': f'sequence_data[{i}][{j}]包含无效数值'}
        
        return {'valid': True, 'data': result}
        
    except ValidationError as e:
        logger.warning(f"预测输入数据验证失败: {e.messages}")
        return {'valid': False, 'errors': e.messages}
    except Exception as e:
        logger.error(f"预测输入数据验证异常: {str(e)}")
        return {'valid': False, 'errors': str(e)}

def validate_risk_assessment_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """验证风险评估输入数据"""
    try:
        schema = RiskAssessmentInputSchema()
        result = schema.load(data)
        
        return {'valid': True, 'data': result}
        
    except ValidationError as e:
        logger.warning(f"风险评估输入数据验证失败: {e.messages}")
        return {'valid': False, 'errors': e.messages}
    except Exception as e:
        logger.error(f"风险评估输入数据验证异常: {str(e)}")
        return {'valid': False, 'errors': str(e)}

def validate_batch_prediction_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """验证批量预测输入数据"""
    try:
        schema = BatchPredictionInputSchema()
        result = schema.load(data)
        
        # 验证每个批量数据项
        prediction_type = result['prediction_type']
        batch_data = result['batch_data']
        
        errors = []
        for i, item in enumerate(batch_data):
            if prediction_type == 'health_indicators':
                validation_result = validate_prediction_input(item)
            elif prediction_type == 'disease_risk':
                validation_result = validate_risk_assessment_input(item)
            else:
                return {'valid': False, 'errors': f'不支持的预测类型: {prediction_type}'}
            
            if not validation_result['valid']:
                errors.append(f'第{i+1}项数据验证失败: {validation_result["errors"]}')
        
        if errors:
            return {'valid': False, 'errors': errors}
        
        return {'valid': True, 'data': result}
        
    except ValidationError as e:
        logger.warning(f"批量预测输入数据验证失败: {e.messages}")
        return {'valid': False, 'errors': e.messages}
    except Exception as e:
        logger.error(f"批量预测输入数据验证异常: {str(e)}")
        return {'valid': False, 'errors': str(e)}

def validate_json_request(request):
    """验证JSON请求数据"""
    try:
        if not request.is_json:
            return None
        return request.get_json()
    except Exception:
        return None

if __name__ == "__main__":
    example_usage()
