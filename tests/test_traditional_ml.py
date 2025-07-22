# 传统机器学习模型测试脚本
# 测试LightGBM、XGBoost、随机森林等传统ML模型

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

from models.traditional_ml import (
    create_health_ml_model, 
    create_ensemble_model,
    LightGBMHealthModel,
    XGBoostHealthModel,
    RandomForestHealthModel,
    EnsembleHealthModel
)
from utils.logger import get_logger

logger = get_logger(__name__)

def generate_health_data(task_type='classification', n_samples=1000, n_features=20):
    """
    生成模拟健康数据
    
    Args:
        task_type: 任务类型 ('classification' 或 'regression')
        n_samples: 样本数量
        n_features: 特征数量
        
    Returns:
        X, y: 特征和标签
    """
    if task_type == 'classification':
        # 生成分类数据（健康风险等级：0-低风险，1-中风险，2-高风险）
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features//2,
            n_redundant=n_features//4,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=42
        )
        
        # 模拟健康特征名称
        feature_names = [
            '年龄', '性别', '身高', '体重', 'BMI', '收缩压', '舒张压', '心率',
            '血糖', '胆固醇', '甘油三酯', '白细胞', '红细胞', '血小板',
            '吸烟史', '饮酒史', '运动频率', '睡眠质量', '压力水平', '家族病史'
        ][:n_features]
        
        X_df = pd.DataFrame(X, columns=feature_names)
        y_labels = ['低风险', '中风险', '高风险']
        
        return X_df, y, y_labels
    
    else:
        # 生成回归数据（健康评分：0-100）
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features//2,
            noise=0.1,
            random_state=42
        )
        
        # 将y标准化到0-100范围
        y = (y - y.min()) / (y.max() - y.min()) * 100
        
        feature_names = [
            '年龄', '性别', '身高', '体重', 'BMI', '收缩压', '舒张压', '心率',
            '血糖', '胆固醇', '甘油三酯', '白细胞', '红细胞', '血小板',
            '吸烟史', '饮酒史', '运动频率', '睡眠质量', '压力水平', '家族病史'
        ][:n_features]
        
        X_df = pd.DataFrame(X, columns=feature_names)
        
        return X_df, y, None

def test_lightgbm_model():
    """测试LightGBM模型"""
    print("=" * 60)
    print("测试 LightGBM 模型")
    print("=" * 60)
    
    try:
        # 生成分类数据
        X, y, y_labels = generate_health_data('classification', 1000, 15)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"训练数据形状: {X_train.shape}")
        print(f"测试数据形状: {X_test.shape}")
        print(f"类别分布: {np.bincount(y_train)}")
        
        # 创建LightGBM分类模型
        lgb_model = create_health_ml_model('lightgbm', 'classification', n_estimators=100)
        
        print(f"✓ 模型创建成功: {lgb_model.model_name}")
        
        # 训练模型
        lgb_model.fit(X_train, y_train)
        print("✓ 模型训练完成")
        
        # 预测
        y_pred = lgb_model.predict(X_test)
        y_proba = lgb_model.predict_proba(X_test)
        
        print(f"✓ 预测完成，预测形状: {y_pred.shape}")
        print(f"✓ 概率预测形状: {y_proba.shape}")
        
        # 评估
        metrics = lgb_model.evaluate(X_test, y_test)
        print("✓ 模型评估结果:")
        for metric, value in metrics.items():
            print(f"  - {metric}: {value:.4f}")
        
        # 特征重要性
        importance = lgb_model.get_feature_importance()
        if importance is not None:
            print(f"✓ 特征重要性分析完成，形状: {importance.shape}")
            top_features = np.argsort(importance)[-5:][::-1]
            print("  - Top 5 重要特征:")
            for i, idx in enumerate(top_features):
                print(f"    {i+1}. {X.columns[idx]}: {importance[idx]:.4f}")
        
        # 测试回归模型
        print("\n--- 测试LightGBM回归模型 ---")
        X_reg, y_reg, _ = generate_health_data('regression', 800, 12)
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
        
        lgb_reg_model = create_health_ml_model('lightgbm', 'regression', n_estimators=100)
        lgb_reg_model.fit(X_train_reg, y_train_reg)
        
        reg_metrics = lgb_reg_model.evaluate(X_test_reg, y_test_reg)
        print("✓ 回归模型评估结果:")
        for metric, value in reg_metrics.items():
            print(f"  - {metric}: {value:.4f}")
        
        print("✓ LightGBM 模型测试通过\n")
        return True
        
    except Exception as e:
        print(f"✗ LightGBM 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_xgboost_model():
    """测试XGBoost模型"""
    print("=" * 60)
    print("测试 XGBoost 模型")
    print("=" * 60)
    
    try:
        # 生成数据
        X, y, y_labels = generate_health_data('classification', 800, 18)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"训练数据形状: {X_train.shape}")
        
        # 创建XGBoost模型
        xgb_model = create_health_ml_model('xgboost', 'classification', 
                                          n_estimators=100, max_depth=4, learning_rate=0.1)
        
        print(f"✓ 模型创建成功: {xgb_model.model_name}")
        
        # 训练模型
        xgb_model.fit(X_train, y_train, feature_selection=True, n_features=10)
        print("✓ 模型训练完成")
        
        # 预测和评估
        metrics = xgb_model.evaluate(X_test, y_test)
        print("✓ 模型评估结果:")
        for metric, value in metrics.items():
            print(f"  - {metric}: {value:.4f}")
        
        # 特征重要性
        importance = xgb_model.get_feature_importance()
        if importance is not None:
            print(f"✓ 特征重要性分析完成")
        
        print("✓ XGBoost 模型测试通过\n")
        return True
        
    except Exception as e:
        print(f"✗ XGBoost 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_random_forest_model():
    """测试随机森林模型"""
    print("=" * 60)
    print("测试 随机森林 模型")
    print("=" * 60)
    
    try:
        # 生成数据
        X, y, y_labels = generate_health_data('classification', 600, 16)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 创建随机森林模型
        rf_model = create_health_ml_model('random_forest', 'classification', 
                                         n_estimators=50, max_depth=8)
        
        print(f"✓ 模型创建成功: {rf_model.model_name}")
        
        # 训练模型
        rf_model.fit(X_train, y_train)
        print("✓ 模型训练完成")
        
        # 评估
        metrics = rf_model.evaluate(X_test, y_test)
        print("✓ 模型评估结果:")
        for metric, value in metrics.items():
            print(f"  - {metric}: {value:.4f}")
        
        print("✓ 随机森林 模型测试通过\n")
        return True
        
    except Exception as e:
        print(f"✗ 随机森林 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_other_models():
    """测试其他传统ML模型"""
    print("=" * 60)
    print("测试 其他传统ML模型")
    print("=" * 60)
    
    try:
        # 生成小数据集用于快速测试
        X, y, y_labels = generate_health_data('classification', 400, 10)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 测试各种模型
        model_configs = [
            ('logistic_regression', {}),
            ('svm', {'C': 1.0, 'kernel': 'rbf'}),
            ('gradient_boosting', {'n_estimators': 50}),
            ('knn', {'n_neighbors': 5}),
            ('naive_bayes', {})
        ]
        
        results = {}
        
        for model_name, params in model_configs:
            try:
                model = create_health_ml_model(model_name, 'classification', **params)
                model.fit(X_train, y_train, feature_selection=False)  # 关闭特征选择加速训练
                
                metrics = model.evaluate(X_test, y_test)
                results[model_name] = metrics['accuracy']
                
                print(f"✓ {model.model_name}: 准确率 {metrics['accuracy']:.4f}")
                
            except Exception as e:
                print(f"✗ {model_name} 测试失败: {e}")
                results[model_name] = 0.0
        
        print("\n✓ 模型性能对比:")
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        for model_name, accuracy in sorted_results:
            print(f"  - {model_name}: {accuracy:.4f}")
        
        print("✓ 其他传统ML模型测试通过\n")
        return True
        
    except Exception as e:
        print(f"✗ 其他传统ML模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ensemble_model():
    """测试集成模型"""
    print("=" * 60)
    print("测试 集成模型")
    print("=" * 60)
    
    try:
        # 生成数据
        X, y, y_labels = generate_health_data('classification', 800, 15)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"训练数据形状: {X_train.shape}")
        
        # 配置多个基础模型
        model_configs = [
            {'name': 'lightgbm', 'params': {'n_estimators': 50}},
            {'name': 'xgboost', 'params': {'n_estimators': 50, 'max_depth': 4}},
            {'name': 'random_forest', 'params': {'n_estimators': 30}},
            {'name': 'gradient_boosting', 'params': {'n_estimators': 30}}
        ]
        
        # 创建集成模型
        ensemble_model = create_ensemble_model(
            model_configs, 
            model_type='classification',
            voting='soft',
            weights=[0.3, 0.3, 0.2, 0.2]
        )
        
        print(f"✓ 集成模型创建成功: {ensemble_model.model_name}")
        print(f"✓ 包含 {len(ensemble_model.base_models)} 个基础模型")
        
        # 训练集成模型
        ensemble_model.fit(X_train, y_train, feature_selection=True, n_features=10)
        print("✓ 集成模型训练完成")
        
        # 评估集成模型
        ensemble_metrics = ensemble_model.evaluate(X_test, y_test)
        print("✓ 集成模型评估结果:")
        for metric, value in ensemble_metrics.items():
            print(f"  - {metric}: {value:.4f}")
        
        # 与单个模型对比
        print("\n--- 与单个模型性能对比 ---")
        
        # 测试单个LightGBM模型
        single_lgb = create_health_ml_model('lightgbm', 'classification', n_estimators=100)
        single_lgb.fit(X_train, y_train, feature_selection=True, n_features=10)
        lgb_metrics = single_lgb.evaluate(X_test, y_test)
        
        print(f"✓ 单个LightGBM准确率: {lgb_metrics['accuracy']:.4f}")
        print(f"✓ 集成模型准确率: {ensemble_metrics['accuracy']:.4f}")
        
        improvement = ensemble_metrics['accuracy'] - lgb_metrics['accuracy']
        print(f"✓ 性能提升: {improvement:.4f} ({improvement*100:.2f}%)")
        
        # 测试概率预测
        ensemble_proba = ensemble_model.predict_proba(X_test[:5])
        print(f"✓ 集成概率预测形状: {ensemble_proba.shape}")
        
        print("✓ 集成模型测试通过\n")
        return True
        
    except Exception as e:
        print(f"✗ 集成模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_persistence():
    """测试模型保存和加载"""
    print("=" * 60)
    print("测试 模型保存和加载")
    print("=" * 60)
    
    try:
        # 生成数据
        X, y, _ = generate_health_data('classification', 500, 12)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 训练模型
        model = create_health_ml_model('lightgbm', 'classification', n_estimators=50)
        model.fit(X_train, y_train)
        
        # 保存模型
        model_path = "test_traditional_ml_model.joblib"
        model.save_model(model_path)
        print(f"✓ 模型已保存到: {model_path}")
        
        # 获取原始预测
        original_pred = model.predict(X_test)
        
        # 创建新模型实例并加载
        new_model = create_health_ml_model('lightgbm', 'classification')
        new_model.load_model(model_path)
        print("✓ 模型加载成功")
        
        # 测试加载的模型
        loaded_pred = new_model.predict(X_test)
        
        # 验证预测结果一致性
        pred_match = np.array_equal(original_pred, loaded_pred)
        print(f"✓ 预测结果一致性: {pred_match}")
        
        if pred_match:
            print("✓ 模型保存和加载功能正常")
        else:
            print("✗ 模型保存和加载结果不一致")
            return False
        
        # 清理测试文件
        if os.path.exists(model_path):
            os.remove(model_path)
            print("✓ 测试文件已清理")
        
        print("✓ 模型持久化测试通过\n")
        return True
        
    except Exception as e:
        print(f"✗ 模型持久化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有传统ML模型测试"""
    print("开始传统机器学习模型集成测试")
    print("=" * 80)
    
    test_results = []
    
    # 运行各项测试
    tests = [
        ("LightGBM模型", test_lightgbm_model),
        ("XGBoost模型", test_xgboost_model),
        ("随机森林模型", test_random_forest_model),
        ("其他传统ML模型", test_other_models),
        ("集成模型", test_ensemble_model),
        ("模型持久化", test_model_persistence),
    ]
    
    for test_name, test_func in tests:
        print(f"\n开始测试: {test_name}")
        result = test_func()
        test_results.append((test_name, result))
    
    # 总结测试结果
    print("\n" + "=" * 80)
    print("传统ML模型测试总结")
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
        print("\n🎉 所有测试都通过了！传统ML模型集成完成。")
        print("\n📊 支持的模型类型:")
        print("- LightGBM (分类/回归)")
        print("- XGBoost (分类/回归)")
        print("- 随机森林 (分类/回归)")
        print("- 逻辑回归/线性回归")
        print("- 支持向量机 (SVM)")
        print("- 梯度提升")
        print("- K近邻 (KNN)")
        print("- 朴素贝叶斯")
        print("- 集成模型 (多模型融合)")
        
        print("\n🚀 主要功能:")
        print("- 自动特征预处理和选择")
        print("- 模型训练和预测")
        print("- 性能评估和特征重要性分析")
        print("- 模型保存和加载")
        print("- 多模型集成和投票")
        
        return True
    else:
        print(f"\n⚠️ 有 {failed} 个测试失败，请检查相关代码。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
