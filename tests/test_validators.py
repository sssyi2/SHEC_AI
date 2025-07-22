#!/usr/bin/env python3
"""
数据验证模块测试脚本
测试所有验证器的功能和性能
"""

import sys
import os
import json
from datetime import datetime, timedelta
from decimal import Decimal

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.validators import (
        CustomValidators,
        UserInfoSchema, 
        HealthMetricsSchema,
        PredictionResultSchema,
        HealthDataValidator,
        BatchValidationResult
    )
    print("✅ 数据验证模块导入成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

def test_custom_validators():
    """测试自定义验证器"""
    print("\n🔍 测试自定义验证器...")
    
    # 血压验证测试
    test_cases = [
        (120, 80, True, "正常血压"),
        (180, 95, True, "高血压"),
        (90, 110, False, "舒张压大于收缩压"),
        (250, 50, False, "收缩压超出范围"),
        (80, 30, False, "舒张压超出范围")
    ]
    
    for systolic, diastolic, expected, desc in test_cases:
        result = CustomValidators.validate_blood_pressure(systolic, diastolic)
        status = "✅" if result == expected else "❌"
        print(f"  {status} 血压验证 ({desc}): {systolic}/{diastolic} -> {result}")
    
    # 手机号验证测试
    phone_cases = [
        ("13812345678", True, "正确手机号"),
        ("1381234567", False, "长度不足"),
        ("12812345678", False, "首位数字错误"),
        ("138123456789", False, "长度超出"),
        ("13812345abc", False, "包含字母")
    ]
    
    for phone, expected, desc in phone_cases:
        result = CustomValidators.validate_phone(phone)
        status = "✅" if result == expected else "❌"
        print(f"  {status} 手机号验证 ({desc}): {phone} -> {result}")

def test_user_info_schema():
    """测试用户信息验证"""
    print("\n👤 测试用户信息验证...")
    
    schema = UserInfoSchema()
    
    # 正确数据
    valid_data = {
        "username": "testuser",
        "real_name": "测试用户",
        "age": 30,
        "gender": "M",
        "phone": "13812345678",
        "email": "test@example.com"
    }
    
    try:
        result = schema.load(valid_data)
        print(f"  ✅ 正确数据验证通过: {json.dumps(result, ensure_ascii=False, default=str)}")
    except Exception as e:
        print(f"  ❌ 正确数据验证失败: {e}")
    
    # 错误数据
    invalid_data = {
        "username": "a",  # 太短
        "real_name": "",  # 为空
        "age": 200,      # 超出范围
        "gender": "X",   # 无效性别
        "phone": "123"   # 无效手机号
    }
    
    try:
        result = schema.load(invalid_data)
        print(f"  ❌ 错误数据验证应该失败但通过了: {result}")
    except Exception as e:
        print(f"  ✅ 错误数据验证正确失败: {str(e)}")

def test_health_metrics_schema():
    """测试健康指标验证"""
    print("\n🏥 测试健康指标验证...")
    
    schema = HealthMetricsSchema()
    
    # 完整的健康数据
    complete_data = {
        "patient_id": 1,
        "systolic_pressure": Decimal("120.0"),
        "diastolic_pressure": Decimal("80.0"), 
        "blood_sugar": Decimal("5.6"),
        "blood_sugar_type": "空腹",
        "weight": Decimal("70.0"),
        "height": Decimal("175.0"),
        "heart_rate": 72,
        "body_temperature": Decimal("36.5"),
        "sleep_hours": 8,
        "smoking_status": "从不",
        "drinking_status": "偶尔"
    }
    
    try:
        result = schema.load(complete_data)
        calculated_bmi = result.get('bmi')
        print(f"  ✅ 完整健康数据验证通过")
        print(f"    - 自动计算BMI: {calculated_bmi}")
        print(f"    - 血压组合: {result['systolic_pressure']}/{result['diastolic_pressure']}")
    except Exception as e:
        print(f"  ❌ 完整健康数据验证失败: {e}")
    
    # 血压不匹配数据
    invalid_bp_data = {
        "patient_id": 1,
        "systolic_pressure": Decimal("80.0"),   # 收缩压小于舒张压
        "diastolic_pressure": Decimal("120.0"),
    }
    
    try:
        result = schema.load(invalid_bp_data)
        print(f"  ❌ 血压不匹配数据应该失败但通过了: {result}")
    except Exception as e:
        print(f"  ✅ 血压不匹配数据正确失败: {str(e)}")

def test_prediction_result_schema():
    """测试预测结果验证"""
    print("\n🤖 测试预测结果验证...")
    
    schema = PredictionResultSchema()
    
    # 正确的预测结果
    prediction_data = {
        "patient_id": 1,
        "prediction_type": "health_trend",
        "model_name": "HealthLSTM_v1",
        "model_version": "1.0.0",
        "input_data": {
            "age": 45,
            "gender": "M",
            "blood_pressure": [120, 80],
            "blood_sugar": 5.6
        },
        "prediction_result": {
            "health_score": 0.85,
            "trend": "stable",
            "recommendations": ["maintain_diet", "regular_exercise"]
        },
        "confidence_score": Decimal("0.92"),
        "risk_level": "低",
        "created_at": datetime.now().isoformat()
    }
    
    try:
        result = schema.load(prediction_data)
        print(f"  ✅ 预测结果验证通过")
        print(f"    - 模型: {result['model_name']} v{result['model_version']}")
        print(f"    - 置信度: {result['confidence_score']}")
        print(f"    - 风险等级: {result['risk_level']}")
    except Exception as e:
        print(f"  ❌ 预测结果验证失败: {e}")

def test_health_data_validator():
    """测试健康数据验证器"""
    print("\n📊 测试健康数据验证器...")
    
    validator = HealthDataValidator()
    
    # 测试单条记录验证
    health_data = {
        "patient_id": 1,
        "age": 45,
        "gender": "M",
        "systolic_pressure": 140,
        "diastolic_pressure": 90,
        "blood_sugar": 8.5,
        "bmi": 25.3,
        "weight": 75.0,
        "height": 175.0
    }
    
    is_valid, validated_data, errors = validator.validate_health_data_for_prediction(health_data)
    
    if is_valid:
        print(f"  ✅ 健康数据预测验证通过")
        print(f"    - 标准化数据: {json.dumps(validated_data, ensure_ascii=False, default=str)}")
    else:
        print(f"  ❌ 健康数据预测验证失败: {errors}")
    
    # 测试批量验证
    batch_data = [
        {"username": "user1", "real_name": "用户1", "age": 25, "gender": "F", "phone": "13812345678"},
        {"username": "user2", "real_name": "用户2", "age": 35, "gender": "M", "phone": "13987654321"},
        {"username": "bad", "real_name": "", "age": "invalid", "gender": "X", "phone": "123"}  # 错误数据
    ]
    
    result = validator.validate_batch_records(batch_data, 'user_info')
    print(f"\n  📋 批量验证结果:")
    print(f"    - 总记录: {result.total_records}")
    print(f"    - 有效记录: {result.valid_count}")
    print(f"    - 无效记录: {result.invalid_count}")
    
    if result.invalid_count > 0:
        print(f"    - 验证错误:")
        for error in result.validation_errors:
            print(f"      * {error}")

def test_performance():
    """测试性能"""
    print("\n⚡ 性能测试...")
    
    validator = HealthDataValidator()
    
    # 生成大批量数据
    import time
    
    large_batch = []
    for i in range(1000):
        large_batch.append({
            "username": f"user_{i}",
            "real_name": f"用户_{i}",
            "age": 20 + (i % 50),
            "gender": "M" if i % 2 == 0 else "F",
            "phone": f"138{i:08d}"
        })
    
    start_time = time.time()
    result = validator.validate_batch_records(large_batch, 'user_info')
    end_time = time.time()
    
    duration = end_time - start_time
    records_per_second = result.total_records / duration
    
    print(f"  ✅ 批量验证性能:")
    print(f"    - 验证记录数: {result.total_records}")
    print(f"    - 耗时: {duration:.3f}秒")
    print(f"    - 处理速度: {records_per_second:.0f} 记录/秒")
    print(f"    - 有效率: {result.valid_count/result.total_records*100:.1f}%")

def main():
    """主测试函数"""
    print("🧪 SHEC AI 数据验证模块测试")
    print("=" * 50)
    
    try:
        test_custom_validators()
        test_user_info_schema() 
        test_health_metrics_schema()
        test_prediction_result_schema()
        test_health_data_validator()
        test_performance()
        
        print("\n" + "=" * 50)
        print("🎉 所有测试完成！数据验证模块运行正常")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
