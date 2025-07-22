# PyTorch模型结构测试脚本
# 验证模型创建、前向传播和基本功能

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from models.model_factory import ModelFactory, model_manager
from models.health_lstm import HealthLSTM, MultiTaskHealthLSTM
from models.risk_assessment import RiskAssessmentNet, MultiDiseaseRiskNet
from utils.logger import get_logger

logger = get_logger(__name__)

def test_health_lstm():
    """测试HealthLSTM模型"""
    print("=" * 60)
    print("测试 HealthLSTM 模型")
    print("=" * 60)
    
    try:
        # 创建模型
        model = ModelFactory.create_health_lstm(
            input_dim=20,
            hidden_dim=64,
            num_layers=2,
            output_dim=3,
            sequence_length=7,
            dropout=0.1
        )
        
        print(f"✓ 模型创建成功: {model.__class__.__name__}")
        print(f"✓ 设备: {model.device}")
        print(f"✓ 模型信息: {model.get_model_info()}")
        
        # 测试前向传播
        batch_size = 4
        sequence_length = 7
        input_dim = 20
        
        # 创建测试数据
        test_input = torch.randn(batch_size, sequence_length, input_dim)
        
        print(f"\n测试输入形状: {test_input.shape}")
        
        # 移动数据到同一设备
        test_input = test_input.to(model.device)
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            output = model(test_input)
            print(f"✓ 前向传播成功，输出形状: {output.shape}")
        
        # 测试预测功能
        test_data = np.random.randn(sequence_length, input_dim)
        prediction = model.predict(test_data)
        print(f"✓ 预测功能测试成功，预测结果形状: {prediction.shape}")
        
        # 测试序列预测
        sequence_pred = model.predict_sequence(test_data, future_steps=3)
        print(f"✓ 序列预测功能测试成功，预测序列形状: {sequence_pred.shape}")
        
        print("✓ HealthLSTM 模型测试通过\n")
        return True
        
    except Exception as e:
        print(f"✗ HealthLSTM 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_risk_assessment():
    """测试RiskAssessmentNet模型"""
    print("=" * 60)
    print("测试 RiskAssessmentNet 模型")
    print("=" * 60)
    
    try:
        # 创建模型
        model = ModelFactory.create_risk_assessment(
            input_dim=25,
            hidden_dims=[128, 64, 32],
            num_classes=3,
            dropout=0.2
        )
        
        print(f"✓ 模型创建成功: {model.__class__.__name__}")
        print(f"✓ 设备: {model.device}")
        print(f"✓ 模型信息: {model.get_model_info()}")
        
        # 测试前向传播
        batch_size = 8
        input_dim = 25
        
        # 创建测试数据
        test_input = torch.randn(batch_size, input_dim)
        
        print(f"\n测试输入形状: {test_input.shape}")
        
        # 移动数据到同一设备
        test_input = test_input.to(model.device)
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            outputs = model(test_input)
            print(f"✓ 前向传播成功")
            for key, value in outputs.items():
                print(f"  - {key}: {value.shape}")
        
        # 测试风险预测功能
        test_data = np.random.randn(input_dim)
        risk_result = model.predict_risk(test_data)
        print(f"✓ 风险预测功能测试成功")
        for key, value in risk_result.items():
            if isinstance(value, np.ndarray):
                print(f"  - {key}: 形状 {value.shape}")
            else:
                print(f"  - {key}: {value}")
        
        # 测试特征重要性
        importance = model.get_feature_importance(test_data)
        print(f"✓ 特征重要性分析成功，形状: {importance.shape}")
        
        print("✓ RiskAssessmentNet 模型测试通过\n")
        return True
        
    except Exception as e:
        print(f"✗ RiskAssessmentNet 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multitask_lstm():
    """测试MultiTaskHealthLSTM模型"""
    print("=" * 60)
    print("测试 MultiTaskHealthLSTM 模型")
    print("=" * 60)
    
    try:
        # 任务配置
        task_configs = {
            'blood_pressure': {'output_dim': 2},  # 收缩压、舒张压
            'blood_sugar': {'output_dim': 1},     # 血糖值
            'heart_rate': {'output_dim': 1},      # 心率
        }
        
        # 创建模型
        model = ModelFactory.create_multitask_lstm(
            input_dim=30,
            task_configs=task_configs,
            shared_hidden_dim=128
        )
        
        print(f"✓ 模型创建成功: {model.__class__.__name__}")
        print(f"✓ 设备: {model.device}")
        print(f"✓ 任务数量: {len(task_configs)}")
        
        # 测试前向传播
        batch_size = 4
        sequence_length = 7
        input_dim = 30
        
        test_input = torch.randn(batch_size, sequence_length, input_dim)
        print(f"\n测试输入形状: {test_input.shape}")
        
        # 移动数据到同一设备
        test_input = test_input.to(model.device)
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            outputs = model(test_input)
            print(f"✓ 前向传播成功")
            for task_name, output in outputs.items():
                print(f"  - {task_name}: {output.shape}")
        
        # 测试多任务预测
        test_data = np.random.randn(sequence_length, input_dim)
        predictions = model.predict_multi_task(test_data)
        print(f"✓ 多任务预测功能测试成功")
        for task_name, pred in predictions.items():
            print(f"  - {task_name}: 形状 {pred.shape}")
        
        print("✓ MultiTaskHealthLSTM 模型测试通过\n")
        return True
        
    except Exception as e:
        print(f"✗ MultiTaskHealthLSTM 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multidisease_risk():
    """测试MultiDiseaseRiskNet模型"""
    print("=" * 60)
    print("测试 MultiDiseaseRiskNet 模型")
    print("=" * 60)
    
    try:
        # 疾病配置
        disease_configs = {
            'diabetes': {'num_classes': 2, 'weight': 1.0},      # 糖尿病：有/无
            'hypertension': {'num_classes': 3, 'weight': 1.2},  # 高血压：正常/轻度/重度
            'heart_disease': {'num_classes': 2, 'weight': 1.5}, # 心脏病：有/无
        }
        
        # 创建模型
        model = ModelFactory.create_model('multidisease_risk', {
            'input_dim': 35,
            'disease_configs': disease_configs,
            'shared_hidden_dims': [256, 128],
            'dropout': 0.3
        })
        
        print(f"✓ 模型创建成功: {model.__class__.__name__}")
        print(f"✓ 设备: {model.device}")
        print(f"✓ 疾病数量: {len(disease_configs)}")
        
        # 测试前向传播
        batch_size = 6
        input_dim = 35
        
        test_input = torch.randn(batch_size, input_dim)
        print(f"\n测试输入形状: {test_input.shape}")
        
        # 移动数据到同一设备
        test_input = test_input.to(model.device)
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            outputs = model(test_input)
            print(f"✓ 前向传播成功")
            for key, value in outputs.items():
                print(f"  - {key}: {value.shape}")
        
        # 测试多疾病风险预测
        test_data = np.random.randn(input_dim)
        risk_results = model.predict_multi_disease_risk(test_data)
        print(f"✓ 多疾病风险预测功能测试成功")
        for disease, result in risk_results.items():
            if disease == 'global_risk_score':
                print(f"  - {disease}: {result}")
            else:
                print(f"  - {disease}:")
                for key, value in result.items():
                    if isinstance(value, np.ndarray):
                        print(f"    * {key}: 形状 {value.shape}")
                    else:
                        print(f"    * {key}: {value}")
        
        print("✓ MultiDiseaseRiskNet 模型测试通过\n")
        return True
        
    except Exception as e:
        print(f"✗ MultiDiseaseRiskNet 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_management():
    """测试模型管理功能"""
    print("=" * 60)
    print("测试模型管理功能")
    print("=" * 60)
    
    try:
        # 创建一个简单的模型
        model = ModelFactory.create_health_lstm(
            input_dim=15,
            hidden_dim=32,
            output_dim=1
        )
        
        print(f"✓ 测试模型创建成功")
        
        # 保存模型
        model_name = "test_health_lstm"
        version = "v1.0.0"
        
        saved_path = model_manager.save_model(
            model=model,
            model_name=model_name,
            version=version,
            metadata={'description': '测试用LSTM模型', 'accuracy': 0.95}
        )
        
        print(f"✓ 模型保存成功: {saved_path}")
        
        # 列出模型
        models_list = model_manager.list_models()
        print(f"✓ 模型列表: {models_list}")
        
        # 加载模型
        loaded_model, metadata = model_manager.load_model(model_name, version)
        print(f"✓ 模型加载成功: {loaded_model.__class__.__name__}")
        print(f"✓ 元数据: {metadata['version']}")
        
        # 测试加载的模型
        test_input = torch.randn(1, 7, 15)
        test_input = test_input.to(loaded_model.device)  # 确保数据在正确设备上
        with torch.no_grad():
            output = loaded_model(test_input)
            print(f"✓ 加载的模型工作正常，输出形状: {output.shape}")
        
        # 获取模型信息
        model_info = model_manager.get_model_info(model_name, version)
        print(f"✓ 模型信息获取成功")
        
        # 清理测试模型
        model_manager.delete_model(model_name)
        print(f"✓ 测试模型删除成功")
        
        print("✓ 模型管理功能测试通过\n")
        return True
        
    except Exception as e:
        print(f"✗ 模型管理功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gpu_compatibility():
    """测试GPU兼容性"""
    print("=" * 60)
    print("测试GPU兼容性")
    print("=" * 60)
    
    try:
        # 检查CUDA可用性
        cuda_available = torch.cuda.is_available()
        print(f"CUDA 可用: {cuda_available}")
        
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            
            print(f"GPU 数量: {gpu_count}")
            print(f"当前设备: {current_device}")
            print(f"设备名称: {device_name}")
            
            # 测试GPU上的模型
            model = ModelFactory.create_risk_assessment(
                input_dim=20,
                device='cuda'
            )
            
            print(f"✓ GPU模型创建成功，设备: {model.device}")
            
            # 在GPU上测试
            test_input = torch.randn(4, 20).cuda()
            with torch.no_grad():
                output = model(test_input)
                print(f"✓ GPU前向传播成功")
            
        else:
            print("GPU不可用，使用CPU模式")
            model = ModelFactory.create_risk_assessment(
                input_dim=20,
                device='cpu'
            )
            print(f"✓ CPU模型创建成功，设备: {model.device}")
        
        print("✓ GPU兼容性测试通过\n")
        return True
        
    except Exception as e:
        print(f"✗ GPU兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("开始PyTorch模型结构测试")
    print("=" * 80)
    
    test_results = []
    
    # 运行各项测试
    tests = [
        ("HealthLSTM模型", test_health_lstm),
        ("RiskAssessmentNet模型", test_risk_assessment),
        ("MultiTaskHealthLSTM模型", test_multitask_lstm),
        ("MultiDiseaseRiskNet模型", test_multidisease_risk),
        ("模型管理功能", test_model_management),
        ("GPU兼容性", test_gpu_compatibility),
    ]
    
    for test_name, test_func in tests:
        print(f"\n开始测试: {test_name}")
        result = test_func()
        test_results.append((test_name, result))
    
    # 总结测试结果
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results:
        status = "通过" if result else "失败"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
        
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n总计: {passed + failed} 个测试")
    print(f"通过: {passed} 个")
    print(f"失败: {failed} 个")
    
    if failed == 0:
        print("\n🎉 所有测试都通过了！PyTorch模型结构开发完成。")
        return True
    else:
        print(f"\n⚠️ 有 {failed} 个测试失败，请检查相关代码。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
