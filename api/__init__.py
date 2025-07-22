# api/__init__.py
"""
API模块初始化文件
避免循环导入，使用动态导入
"""

# 不要在模块级别导入蓝图，避免循环导入
# 蓝图将在需要时动态导入

__all__ = ['health_bp', 'predict_bp', 'models_bp', 'data_bp']
