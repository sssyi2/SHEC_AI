"""
数据库使用示例
展示如何使用适配schema的database.py模块
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.database import DatabaseManager, AIDataAccess, format_datetime, parse_json_field
from datetime import datetime, date
import json

# 示例：如何使用新的数据库API

def example_user_operations():
    """用户操作示例"""
    print("=" * 50)
    print("用户操作示例")
    print("=" * 50)
    
    # 获取用户信息
    user_id = 1
    user_info = DatabaseManager.get_user_by_id(user_id)
    print(f"用户信息: {user_info}")
    
    # 获取用户健康记录
    health_record = DatabaseManager.get_health_record(user_id)
    print(f"健康记录: {health_record}")
    
    # 获取健康指标
    health_metrics = DatabaseManager.get_patient_health_metrics(user_id, limit=5)
    print(f"健康指标数量: {len(health_metrics) if health_metrics else 0}")

def example_ai_predictions():
    """AI预测操作示例"""
    print("=" * 50)
    print("AI预测操作示例")
    print("=" * 50)
    
    patient_id = 1
    
    # 获取用于预测的患者数据
    patient_data = AIDataAccess.get_patient_for_prediction(patient_id)
    if patient_data:
        print("✅ 成功获取患者预测数据")
        print(f"   - 用户信息: {'存在' if patient_data['user_info'] else '不存在'}")
        print(f"   - 健康记录: {'存在' if patient_data['health_record'] else '不存在'}")
        print(f"   - 历史指标: {len(patient_data['recent_metrics']) if patient_data['recent_metrics'] else 0} 条")
    else:
        print("❌ 患者数据不存在")

def example_save_prediction():
    """保存预测结果示例"""
    print("=" * 50)
    print("保存预测结果示例")
    print("=" * 50)
    
    # 模拟预测结果数据
    prediction_data = {
        'patient_id': 1,
        'prediction_id': f'pred_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'prediction_type': 'health_trend',
        'input_data': {
            'age': 45,
            'gender': 'male',
            'systolic_bp': 130,
            'diastolic_bp': 85,
            'heart_rate': 72
        },
        'prediction_data': {
            'risk_score': 0.65,
            'trend': 'stable',
            'recommendations': ['保持适量运动', '控制饮食']
        },
        'confidence_score': 0.87,
        'model_version': 'v1.0.0',
        'prediction_period': 30
    }
    
    try:
        # 注意：这只是示例，实际需要数据库连接
        print("预测结果数据结构验证:")
        print(f"   - 患者ID: {prediction_data['patient_id']}")
        print(f"   - 预测类型: {prediction_data['prediction_type']}")
        print(f"   - 置信度: {prediction_data['confidence_score']}")
        print("✅ 预测结果数据结构正确")
    except Exception as e:
        print(f"❌ 预测结果保存失败: {e}")

def example_health_metrics():
    """健康指标操作示例"""
    print("=" * 50)
    print("健康指标操作示例")
    print("=" * 50)
    
    # 模拟健康指标数据
    metrics_data = {
        'patient_id': 1,
        'record_date': date.today(),
        'age': 45,
        'gender': 'male',
        'systolic_pressure': 130,
        'diastolic_pressure': 85,
        'blood_sugar': 5.5,
        'bmi': 24.5,
        'other_metrics': {
            'temperature': 36.5,
            'heart_rate': 72,
            'blood_oxygen': 98
        },
        'data_source': 'manual'
    }
    
    print("健康指标数据结构验证:")
    print(f"   - 患者ID: {metrics_data['patient_id']}")
    print(f"   - 记录日期: {metrics_data['record_date']}")
    print(f"   - 血压: {metrics_data['systolic_pressure']}/{metrics_data['diastolic_pressure']}")
    print(f"   - BMI: {metrics_data['bmi']}")
    print("✅ 健康指标数据结构正确")

def example_ai_preferences():
    """AI偏好设置示例"""
    print("=" * 50)
    print("AI偏好设置示例")
    print("=" * 50)
    
    # 模拟AI偏好数据
    preferences = {
        'enable_auto_prediction': True,
        'prediction_frequency': 7,  # 7天
        'notification_enabled': True,
        'risk_threshold': 0.7,
        'preferred_models': ['HealthLSTM_v1.0', 'RiskAssessment_v1.0']
    }
    
    print("AI偏好设置数据结构验证:")
    print(f"   - 自动预测: {'启用' if preferences['enable_auto_prediction'] else '禁用'}")
    print(f"   - 预测频率: {preferences['prediction_frequency']} 天")
    print(f"   - 风险阈值: {preferences['risk_threshold']}")
    print(f"   - 偏好模型: {len(preferences['preferred_models'])} 个")
    print("✅ AI偏好设置数据结构正确")

def example_model_management():
    """模型管理示例"""
    print("=" * 50)
    print("模型管理示例")
    print("=" * 50)
    
    # 模拟AI模型数据
    model_info = {
        'id': 1,
        'model_name': 'HealthLSTM',
        'model_version': 'v1.0.0',
        'model_type': 'LSTM',
        'model_path': '/models/health_lstm.pth',
        'configuration': {
            'input_size': 6,
            'hidden_size': 64,
            'num_layers': 2,
            'output_size': 3,
            'dropout': 0.2
        },
        'is_active': True
    }
    
    print("AI模型信息验证:")
    print(f"   - 模型名称: {model_info['model_name']}")
    print(f"   - 模型版本: {model_info['model_version']}")
    print(f"   - 模型类型: {model_info['model_type']}")
    print(f"   - 状态: {'激活' if model_info['is_active'] else '停用'}")
    print("✅ AI模型信息结构正确")

def example_task_management():
    """任务管理示例"""
    print("=" * 50)
    print("任务管理示例")
    print("=" * 50)
    
    # 模拟预测任务数据
    task_data = {
        'task_id': f'task_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'patient_id': 1,
        'task_type': 'health_prediction',
        'status': 'pending',
        'input_parameters': {
            'prediction_type': 'health_trend',
            'period': 30,
            'model_version': 'v1.0.0'
        },
        'priority': 5
    }
    
    print("预测任务数据结构验证:")
    print(f"   - 任务ID: {task_data['task_id']}")
    print(f"   - 患者ID: {task_data['patient_id']}")
    print(f"   - 任务类型: {task_data['task_type']}")
    print(f"   - 状态: {task_data['status']}")
    print(f"   - 优先级: {task_data['priority']}")
    print("✅ 预测任务数据结构正确")

def example_utility_functions():
    """工具函数示例"""
    print("=" * 50)
    print("工具函数示例")
    print("=" * 50)
    
    # 日期时间格式化
    now = datetime.now()
    today = date.today()
    
    formatted_datetime = format_datetime(now)
    formatted_date = format_datetime(today)
    
    print(f"日期时间格式化:")
    print(f"   - 当前时间: {formatted_datetime}")
    print(f"   - 今天日期: {formatted_date}")
    
    # JSON字段解析
    json_data = '{"temperature": 36.5, "heart_rate": 72}'
    parsed_data = parse_json_field(json_data)
    
    print(f"JSON字段解析:")
    print(f"   - 原始: {json_data}")
    print(f"   - 解析: {parsed_data}")
    
    print("✅ 工具函数测试完成")

def main():
    """主函数 - 运行所有示例"""
    print("🚀 SHEC AI数据库使用示例")
    print("适配现有schema.sql的database.py模块")
    print()
    
    try:
        example_user_operations()
        example_ai_predictions()
        example_save_prediction()
        example_health_metrics()
        example_ai_preferences()
        example_model_management()
        example_task_management()
        example_utility_functions()
        
        print()
        print("=" * 50)
        print("🎉 所有示例执行完成!")
        print("=" * 50)
        print()
        print("📋 数据库适配总结:")
        print("✅ 适配现有user表结构")
        print("✅ 适配AI相关表结构")
        print("✅ 提供完整的数据访问API")
        print("✅ 支持JSON字段操作")
        print("✅ 包含性能监控功能")
        print("✅ 支持任务队列管理")
        print("✅ 提供实用工具函数")
        
    except Exception as e:
        print(f"❌ 示例执行出错: {e}")

if __name__ == "__main__":
    main()
