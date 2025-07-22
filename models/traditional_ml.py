# 传统机器学习模型集成
# 集成LightGBM、XGBoost等传统ML模型用于健康预测和风险评估

import numpy as np
import pandas as pd
import joblib
import json
import os
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 传统ML模型库
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    lgb = None

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

from utils.logger import get_logger

logger = get_logger(__name__)

class HealthMLModel:
    """
    健康预测传统机器学习模型基类
    """
    
    def __init__(self, 
                 model_type: str = 'classification',
                 model_name: str = 'HealthMLModel',
                 random_state: int = 42):
        """
        初始化ML模型
        
        Args:
            model_type: 模型类型 ('classification' 或 'regression')
            model_name: 模型名称
            random_state: 随机种子
        """
        self.model_type = model_type
        self.model_name = model_name
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = None
        self.feature_selector = None
        self.is_fitted = False
        
        # 模型配置
        self.config = {
            'model_type': model_type,
            'model_name': model_name,
            'random_state': random_state,
            'created_at': datetime.now().isoformat(),
        }
        
        logger.info(f"初始化 {model_name} 模型，类型: {model_type}")
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], 
            y: Union[pd.Series, np.ndarray],
            feature_selection: bool = True,
            n_features: int = None) -> 'HealthMLModel':
        """
        训练模型
        
        Args:
            X: 训练特征
            y: 训练标签
            feature_selection: 是否进行特征选择
            n_features: 选择的特征数量
            
        Returns:
            self: 返回自身用于链式调用
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # 数据预处理
        X_processed = self._preprocess_features(X)
        y_processed = self._preprocess_labels(y)
        
        # 特征选择
        if feature_selection:
            X_processed = self._select_features(X_processed, y_processed, n_features)
        
        # 训练模型
        self.model.fit(X_processed, y_processed)
        self.is_fitted = True
        
        logger.info(f"模型 {self.model_name} 训练完成")
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        预测
        
        Args:
            X: 特征数据
            
        Returns:
            np.ndarray: 预测结果
        """
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit方法")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # 数据预处理
        X_processed = self._preprocess_features(X, is_training=False)
        
        # 特征选择
        if self.feature_selector is not None:
            X_processed = self.feature_selector.transform(X_processed)
        
        # 预测
        predictions = self.model.predict(X_processed)
        
        # 标签逆变换
        if self.label_encoder is not None:
            predictions = self.label_encoder.inverse_transform(predictions.astype(int))
        
        return predictions
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        预测概率（仅分类模型）
        
        Args:
            X: 特征数据
            
        Returns:
            np.ndarray: 预测概率
        """
        if self.model_type != 'classification':
            raise ValueError("概率预测仅适用于分类模型")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("当前模型不支持概率预测")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # 数据预处理
        X_processed = self._preprocess_features(X, is_training=False)
        
        # 特征选择
        if self.feature_selector is not None:
            X_processed = self.feature_selector.transform(X_processed)
        
        return self.model.predict_proba(X_processed)
    
    def _preprocess_features(self, X: np.ndarray, is_training: bool = True) -> np.ndarray:
        """预处理特征"""
        if is_training:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def _preprocess_labels(self, y: np.ndarray) -> np.ndarray:
        """预处理标签"""
        if self.model_type == 'classification':
            if y.dtype == 'object' or len(np.unique(y)) < len(y) * 0.1:
                # 字符串标签或类别数量较少，使用标签编码
                self.label_encoder = LabelEncoder()
                y_encoded = self.label_encoder.fit_transform(y)
                return y_encoded
        
        return y
    
    def _select_features(self, X: np.ndarray, y: np.ndarray, n_features: int = None) -> np.ndarray:
        """特征选择"""
        if n_features is None:
            n_features = min(X.shape[1], max(10, X.shape[1] // 2))
        
        if self.model_type == 'classification':
            self.feature_selector = SelectKBest(score_func=f_classif, k=n_features)
        else:
            from sklearn.feature_selection import f_regression
            self.feature_selector = SelectKBest(score_func=f_regression, k=n_features)
        
        X_selected = self.feature_selector.fit_transform(X, y)
        
        logger.info(f"特征选择: {X.shape[1]} -> {X_selected.shape[1]}")
        return X_selected
    
    def evaluate(self, X: Union[pd.DataFrame, np.ndarray], 
                 y: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            X: 测试特征
            y: 测试标签
            
        Returns:
            Dict[str, float]: 评估指标
        """
        predictions = self.predict(X)
        
        if self.model_type == 'classification':
            metrics = {
                'accuracy': accuracy_score(y, predictions),
                'precision': precision_score(y, predictions, average='weighted', zero_division=0),
                'recall': recall_score(y, predictions, average='weighted', zero_division=0),
                'f1_score': f1_score(y, predictions, average='weighted', zero_division=0)
            }
        else:
            metrics = {
                'mse': mean_squared_error(y, predictions),
                'mae': mean_absolute_error(y, predictions),
                'r2_score': r2_score(y, predictions),
                'rmse': np.sqrt(mean_squared_error(y, predictions))
            }
        
        return metrics
    
    def get_feature_importance(self) -> np.ndarray:
        """获取特征重要性"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_).flatten()
        else:
            logger.warning("当前模型不支持特征重要性分析")
            return None
    
    def save_model(self, filepath: str) -> None:
        """保存模型"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_selector': self.feature_selector,
            'config': self.config,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """加载模型"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_selector = model_data['feature_selector']
        self.config = model_data['config']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"模型已从 {filepath} 加载")


class LightGBMHealthModel(HealthMLModel):
    """
    LightGBM健康预测模型
    """
    
    def __init__(self, 
                 model_type: str = 'classification',
                 **lgb_params):
        """
        初始化LightGBM模型
        
        Args:
            model_type: 模型类型
            **lgb_params: LightGBM参数
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM未安装，请安装: pip install lightgbm")
        
        super().__init__(model_type, 'LightGBMHealthModel')
        
        # 默认参数
        default_params = {
            'objective': 'multiclass' if model_type == 'classification' else 'regression',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': self.random_state
        }
        
        # 更新参数
        default_params.update(lgb_params)
        self.lgb_params = default_params
        
        # 创建模型
        if model_type == 'classification':
            self.model = lgb.LGBMClassifier(**default_params)
        else:
            self.model = lgb.LGBMRegressor(**default_params)
        
        self.config.update({'lgb_params': default_params})


class XGBoostHealthModel(HealthMLModel):
    """
    XGBoost健康预测模型
    """
    
    def __init__(self, 
                 model_type: str = 'classification',
                 **xgb_params):
        """
        初始化XGBoost模型
        
        Args:
            model_type: 模型类型
            **xgb_params: XGBoost参数
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost未安装，请安装: pip install xgboost")
        
        super().__init__(model_type, 'XGBoostHealthModel')
        
        # 默认参数
        default_params = {
            'objective': 'multi:softprob' if model_type == 'classification' else 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state,
            'eval_metric': 'mlogloss' if model_type == 'classification' else 'rmse'
        }
        
        # 更新参数
        default_params.update(xgb_params)
        self.xgb_params = default_params
        
        # 创建模型
        if model_type == 'classification':
            self.model = xgb.XGBClassifier(**default_params)
        else:
            self.model = xgb.XGBRegressor(**default_params)
        
        self.config.update({'xgb_params': default_params})


class RandomForestHealthModel(HealthMLModel):
    """
    随机森林健康预测模型
    """
    
    def __init__(self, 
                 model_type: str = 'classification',
                 n_estimators: int = 100,
                 max_depth: int = None,
                 **rf_params):
        """
        初始化随机森林模型
        
        Args:
            model_type: 模型类型
            n_estimators: 树的数量
            max_depth: 最大深度
            **rf_params: 随机森林参数
        """
        super().__init__(model_type, 'RandomForestHealthModel')
        
        # 默认参数
        default_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        # 更新参数
        default_params.update(rf_params)
        self.rf_params = default_params
        
        # 创建模型
        if model_type == 'classification':
            self.model = RandomForestClassifier(**default_params)
        else:
            self.model = RandomForestRegressor(**default_params)
        
        self.config.update({'rf_params': default_params})


class EnsembleHealthModel(HealthMLModel):
    """
    集成健康预测模型
    结合多个不同的ML模型进行预测
    """
    
    def __init__(self, 
                 models: List[HealthMLModel],
                 voting: str = 'soft',  # 'hard' 或 'soft'
                 weights: List[float] = None):
        """
        初始化集成模型
        
        Args:
            models: 基础模型列表
            voting: 投票方式
            weights: 模型权重
        """
        if not models:
            raise ValueError("至少需要一个基础模型")
        
        model_type = models[0].model_type
        if not all(model.model_type == model_type for model in models):
            raise ValueError("所有基础模型必须是相同类型")
        
        super().__init__(model_type, 'EnsembleHealthModel')
        
        self.base_models = models
        self.voting = voting
        self.weights = weights or [1.0] * len(models)
        
        if len(self.weights) != len(models):
            raise ValueError("权重数量必须与模型数量相同")
        
        # 归一化权重
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        self.config.update({
            'num_models': len(models),
            'voting': voting,
            'weights': self.weights
        })
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], 
            y: Union[pd.Series, np.ndarray],
            **kwargs) -> 'EnsembleHealthModel':
        """训练所有基础模型"""
        
        for i, model in enumerate(self.base_models):
            logger.info(f"训练第 {i+1}/{len(self.base_models)} 个模型: {model.model_name}")
            model.fit(X, y, **kwargs)
        
        self.is_fitted = True
        logger.info("集成模型训练完成")
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """集成预测"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        
        predictions = []
        for model in self.base_models:
            pred = model.predict(X)
            predictions.append(pred)
        
        if self.model_type == 'classification':
            if self.voting == 'hard':
                # 硬投票：多数决定
                predictions_array = np.array(predictions).T
                final_predictions = []
                
                for sample_preds in predictions_array:
                    # 加权投票
                    vote_counts = {}
                    for pred, weight in zip(sample_preds, self.weights):
                        vote_counts[pred] = vote_counts.get(pred, 0) + weight
                    
                    final_pred = max(vote_counts, key=vote_counts.get)
                    final_predictions.append(final_pred)
                
                return np.array(final_predictions)
            
            else:
                # 软投票：概率平均
                probas = []
                for model in self.base_models:
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X)
                        probas.append(proba)
                    else:
                        # 如果模型不支持概率预测，使用硬预测转换为one-hot
                        pred = model.predict(X)
                        unique_classes = np.unique(pred)
                        proba = np.eye(len(unique_classes))[pred]
                        probas.append(proba)
                
                # 加权平均概率
                weighted_proba = np.zeros_like(probas[0])
                for proba, weight in zip(probas, self.weights):
                    weighted_proba += proba * weight
                
                return np.argmax(weighted_proba, axis=1)
        
        else:
            # 回归：加权平均
            weighted_pred = np.zeros_like(predictions[0])
            for pred, weight in zip(predictions, self.weights):
                weighted_pred += pred * weight
            
            return weighted_pred
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """集成概率预测"""
        if self.model_type != 'classification':
            raise ValueError("概率预测仅适用于分类模型")
        
        probas = []
        for model in self.base_models:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                probas.append(proba)
        
        if not probas:
            raise ValueError("没有支持概率预测的模型")
        
        # 加权平均概率
        weighted_proba = np.zeros_like(probas[0])
        total_weight = 0
        
        for proba, weight in zip(probas, self.weights):
            weighted_proba += proba * weight
            total_weight += weight
        
        return weighted_proba / total_weight


# 模型工厂函数
def create_health_ml_model(model_name: str, 
                          model_type: str = 'classification',
                          **kwargs) -> HealthMLModel:
    """
    创建健康ML模型
    
    Args:
        model_name: 模型名称 ('lightgbm', 'xgboost', 'random_forest', 'logistic_regression', 'svm')
        model_type: 模型类型 ('classification', 'regression')
        **kwargs: 模型参数
        
    Returns:
        HealthMLModel: 模型实例
    """
    model_name = model_name.lower()
    
    if model_name == 'lightgbm':
        return LightGBMHealthModel(model_type, **kwargs)
    
    elif model_name == 'xgboost':
        return XGBoostHealthModel(model_type, **kwargs)
    
    elif model_name == 'random_forest':
        return RandomForestHealthModel(model_type, **kwargs)
    
    elif model_name == 'logistic_regression':
        model = HealthMLModel(model_type, 'LogisticRegressionHealthModel')
        if model_type == 'classification':
            model.model = LogisticRegression(random_state=42, **kwargs)
        else:
            model.model = LinearRegression(**kwargs)
        return model
    
    elif model_name == 'svm':
        model = HealthMLModel(model_type, 'SVMHealthModel')
        if model_type == 'classification':
            model.model = SVC(random_state=42, probability=True, **kwargs)
        else:
            model.model = SVR(**kwargs)
        return model
    
    elif model_name == 'gradient_boosting':
        model = HealthMLModel(model_type, 'GradientBoostingHealthModel')
        if model_type == 'classification':
            model.model = GradientBoostingClassifier(random_state=42, **kwargs)
        else:
            model.model = GradientBoostingRegressor(random_state=42, **kwargs)
        return model
    
    elif model_name == 'knn':
        model = HealthMLModel(model_type, 'KNNHealthModel')
        if model_type == 'classification':
            model.model = KNeighborsClassifier(**kwargs)
        else:
            model.model = KNeighborsRegressor(**kwargs)
        return model
    
    elif model_name == 'naive_bayes':
        if model_type != 'classification':
            raise ValueError("朴素贝叶斯仅支持分类任务")
        model = HealthMLModel(model_type, 'NaiveBayesHealthModel')
        model.model = GaussianNB(**kwargs)
        return model
    
    else:
        raise ValueError(f"不支持的模型类型: {model_name}")


def create_ensemble_model(model_configs: List[Dict[str, Any]], 
                         model_type: str = 'classification',
                         voting: str = 'soft',
                         weights: List[float] = None) -> EnsembleHealthModel:
    """
    创建集成模型
    
    Args:
        model_configs: 模型配置列表 [{'name': 'lightgbm', 'params': {...}}, ...]
        model_type: 模型类型
        voting: 投票方式
        weights: 模型权重
        
    Returns:
        EnsembleHealthModel: 集成模型
    """
    models = []
    
    for config in model_configs:
        model_name = config['name']
        model_params = config.get('params', {})
        
        model = create_health_ml_model(model_name, model_type, **model_params)
        models.append(model)
    
    return EnsembleHealthModel(models, voting, weights)
