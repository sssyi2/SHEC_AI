#!/usr/bin/env python3
"""
数据预处理问题诊断
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.data_processor import HealthDataPipeline

def diagnose_nan_issue():
    """诊断nan值产生的原因"""
    
    # 创建简单测试数据
    test_data = pd.DataFrame({
        'patient_id': [1, 2, 3],
        'age': [30, 50, 70],
        'gender': ['M', 'F', 'M'],
        'systolic_pressure': [120, 140, 160],
        'diastolic_pressure': [80, 90, 100],
        'blood_sugar': [5.5, 6.5, 7.5],
        'weight': [70, 80, 75],
        'height': [170, 160, 175],
        'heart_rate': [75, 80, 70],
        'exercise_frequency': [3, 2, 1],
        'created_at': [datetime.now() for _ in range(3)]
    })
    
    print("原始数据:")
    print(test_data)
    print(f"\n原始数据缺失值: {test_data.isnull().sum().sum()}")
    
    # 逐步处理，查看每步的结果
    pipeline = HealthDataPipeline()
    
    # 1. 数据清洗
    print("\n=== 1. 数据清洗 ===")
    cleaned = pipeline.cleaner.clean_numerical_data(test_data)
    cleaned = pipeline.cleaner.clean_categorical_data(cleaned)
    print(f"清洗后缺失值: {cleaned.isnull().sum().sum()}")
    
    # 2. 特征工程
    print("\n=== 2. 特征工程 ===")
    enhanced = pipeline.feature_engineer.create_derived_features(cleaned)
    print(f"特征工程后缺失值: {enhanced.isnull().sum().sum()}")
    print(f"BMI列: {'存在' if 'bmi' in enhanced.columns else '不存在'}")
    
    # 检查BMI值
    if 'bmi' in enhanced.columns:
        print(f"BMI值: {enhanced['bmi'].tolist()}")
        print(f"BMI缺失值: {enhanced['bmi'].isnull().sum()}")
    
    # 检查健康评分
    if 'health_score' in enhanced.columns:
        print(f"健康评分: {enhanced['health_score'].tolist()}")
        print(f"健康评分缺失值: {enhanced['health_score'].isnull().sum()}")
    
    # 3. 标准化
    print("\n=== 3. 标准化 ===")
    normalized = pipeline.normalizer.fit_transform(enhanced)
    print(f"标准化后缺失值: {normalized.isnull().sum().sum()}")
    
    # 检查标准化后的健康评分
    if 'health_score' in normalized.columns:
        print(f"标准化后健康评分: {normalized['health_score'].tolist()}")
        print(f"标准化后健康评分缺失值: {normalized['health_score'].isnull().sum()}")
    
    # 检查所有有缺失值的列
    missing_cols = normalized.columns[normalized.isnull().any()]
    if len(missing_cols) > 0:
        print(f"\n有缺失值的列: {missing_cols.tolist()}")
        for col in missing_cols:
            missing_count = normalized[col].isnull().sum()
            print(f"  {col}: {missing_count} 个缺失值")
    
    return normalized

if __name__ == "__main__":
    diagnose_nan_issue()
