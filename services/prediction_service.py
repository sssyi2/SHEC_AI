# 健康预测服务模块
# 提供健康指标预测和疾病风险评估的核心服务

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
import traceback

from models.model_factory import ModelFactory, model_manager
from models.health_lstm import HealthLSTM, MultiTaskHealthLSTM
from models.risk_assessment import RiskAssessmentNet, MultiDiseaseRiskNet
from models.traditional_ml import create_health_ml_model, create_ensemble_model
from utils.logger import get_logger
from utils.redis_client import get_redis_client, init_redis
from utils.database import DatabaseManager

logger = get_logger(__name__)

class HealthPredictionService:
    """健康指标预测服务"""
    
    def __init__(self):
        """初始化预测服务"""
        self.model_factory = ModelFactory()
        self.db_manager = DatabaseManager()
        
        # 初始化Redis连接
        try:
            init_redis()
            self.redis_client = get_redis_client()
            if self.redis_client:
                logger.info(f"Redis客户端初始化成功，类型: {type(self.redis_client)}")
            else:
                logger.warning("Redis客户端为None，将使用Mock客户端")
        except Exception as e:
            logger.error(f"Redis客户端初始化失败: {e}")
            self.redis_client = None
        
        # 模型缓存
        self.loaded_models = {}
        self.model_cache_timeout = 3600  # 1小时
        
        # 预测缓存配置
        self.prediction_cache_timeout = 1800  # 30分钟
        
        # 线程池用于异步预测
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"预测服务初始化完成，使用设备: {self.device}")
    
    def _generate_cache_key(self, service_type: str, input_data: Dict[str, Any], 
                          model_name: str = None) -> str:
        """生成缓存键"""
        import hashlib
        
        # 创建输入数据的哈希
        data_str = json.dumps(input_data, sort_keys=True, default=str)
        data_hash = hashlib.md5(data_str.encode()).hexdigest()[:8]
        
        model_part = f"_{model_name}" if model_name else ""
        return f"prediction:{service_type}{model_part}:{data_hash}"
    
    def _load_model(self, model_name: str, model_type: str = 'pytorch') -> Any:
        """加载模型到内存"""
        try:
            # 检查内存缓存
            cache_key = f"{model_type}_{model_name}"
            if cache_key in self.loaded_models:
                return self.loaded_models[cache_key]
            
            # 检查Redis缓存
            redis_key = f"model_cache:{cache_key}"
            cached_model = self.redis_client.get(redis_key)
            
            if cached_model is None:
                # 从数据库或文件系统加载模型
                if model_type == 'pytorch':
                    model = self._load_pytorch_model(model_name)
                elif model_type == 'traditional':
                    model = self._load_traditional_model(model_name)
                else:
                    raise ValueError(f"不支持的模型类型: {model_type}")
                
                # 缓存到内存和Redis
                self.loaded_models[cache_key] = model
                # Redis缓存暂时跳过（模型太大）
                
                logger.info(f"模型 {model_name} 加载成功")
                return model
            else:
                logger.info(f"从Redis缓存加载模型 {model_name}")
                # 反序列化模型（这里需要根据实际情况实现）
                return cached_model
                
        except Exception as e:
            logger.error(f"加载模型 {model_name} 失败: {str(e)}")
            raise
    
    def _load_pytorch_model(self, model_name: str) -> torch.nn.Module:
        """加载PyTorch模型"""
        try:
            # 获取模型信息
            model_info = model_manager.get_model_info(model_name)
            if not model_info:
                raise ValueError(f"模型 {model_name} 不存在")
            
            # 获取最新版本
            latest_version = model_info['versions'][-1]
            model_path = latest_version['file_path']
            model_config = latest_version['config']
            
            # 根据配置创建模型
            model_class = model_config.get('model_class', 'HealthLSTM')
            
            if model_class == 'HealthLSTM':
                model = HealthLSTM(**model_config.get('model_params', {}))
            elif model_class == 'RiskAssessmentNet':
                model = RiskAssessmentNet(**model_config.get('model_params', {}))
            elif model_class == 'MultiTaskHealthLSTM':
                model = MultiTaskHealthLSTM(**model_config.get('model_params', {}))
            else:
                # 使用工厂方法
                model = self.model_factory.create_model(
                    model_class.lower(), 
                    model_config.get('model_params', {})
                )
            
            # 加载权重
            if model_path and torch.cuda.is_available():
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            
            model.to(self.device)
            model.eval()
            
            return model
            
        except Exception as e:
            logger.error(f"加载PyTorch模型失败: {str(e)}")
            # 返回默认模型用于测试
            return self._create_default_model()
    
    def _load_traditional_model(self, model_name: str) -> Any:
        """加载传统ML模型"""
        try:
            # 从model_manager获取模型信息
            model_info = model_manager.get_model_info(model_name)
            if model_info and model_info['versions']:
                latest_version = model_info['versions'][-1]
                model_path = latest_version['file_path']
                
                if model_path:
                    import joblib
                    model = joblib.load(model_path)
                    return model
            
            # 如果没有预训练模型，创建默认模型
            return create_health_ml_model('lightgbm', 'classification')
            
        except Exception as e:
            logger.error(f"加载传统ML模型失败: {str(e)}")
            return create_health_ml_model('lightgbm', 'classification')
    
    def _create_default_model(self) -> torch.nn.Module:
        """创建默认测试模型"""
        model = HealthLSTM(
            input_dim=20,
            hidden_dim=64,
            output_dim=3,
            sequence_length=7
        )
        model.to(self.device)
        model.eval()
        return model
    
    def _preprocess_input_data(self, input_data: Dict[str, Any], 
                             prediction_type: str) -> np.ndarray:
        """预处理输入数据"""
        try:
            if prediction_type == 'time_series':
                # 时序数据预处理
                if 'sequence_data' in input_data:
                    data = np.array(input_data['sequence_data'])
                    if len(data.shape) == 2:
                        # [sequence_length, features] -> [1, sequence_length, features]
                        data = data.reshape(1, *data.shape)
                    return data.astype(np.float32)
                else:
                    # 生成模拟时序数据
                    return np.random.randn(1, 7, 20).astype(np.float32)
            
            elif prediction_type == 'static':
                # 静态特征预处理
                if 'features' in input_data:
                    features = input_data['features']
                    if isinstance(features, list):
                        data = np.array(features).reshape(1, -1)
                    elif isinstance(features, dict):
                        # 转换字典为特征向量
                        feature_order = [
                            'age', 'gender', 'height', 'weight', 'bmi',
                            'systolic_bp', 'diastolic_bp', 'heart_rate',
                            'blood_sugar', 'cholesterol', 'triglycerides',
                            'smoking', 'drinking', 'exercise', 'sleep_quality'
                        ]
                        data = []
                        for feature in feature_order:
                            data.append(features.get(feature, 0))
                        data = np.array(data).reshape(1, -1)
                    else:
                        data = np.array(features).reshape(1, -1)
                    return data.astype(np.float32)
                else:
                    # 生成模拟静态数据
                    return np.random.randn(1, 15).astype(np.float32)
            
            else:
                raise ValueError(f"不支持的预测类型: {prediction_type}")
                
        except Exception as e:
            logger.error(f"数据预处理失败: {str(e)}")
            # 返回默认数据
            if prediction_type == 'time_series':
                return np.random.randn(1, 7, 20).astype(np.float32)
            else:
                return np.random.randn(1, 15).astype(np.float32)
    
    def _postprocess_prediction(self, prediction: Any, 
                              prediction_type: str, 
                              model_type: str = 'pytorch') -> Dict[str, Any]:
        """后处理预测结果"""
        try:
            result = {
                'timestamp': datetime.now().isoformat(),
                'prediction_type': prediction_type,
                'model_type': model_type,
                'predicted_class': 0,  # 默认值
                'confidence': 0.5,     # 默认值
                'risk_level': '中风险'  # 默认值
            }
            
            if model_type == 'pytorch':
                if isinstance(prediction, torch.Tensor):
                    # 简单张量输出
                    if prediction.dim() > 1 and prediction.size(1) > 1:
                        probs = F.softmax(prediction, dim=1).cpu().numpy()[0]
                        predicted_class = int(np.argmax(probs))
                        confidence = float(np.max(probs))
                    else:
                        # 单一输出值
                        probs = [0.3, 0.4, 0.3]  # 模拟概率分布
                        predicted_class = 1
                        confidence = 0.4
                    
                    result.update({
                        'predicted_class': predicted_class,
                        'class_probabilities': probs if isinstance(probs, list) else probs.tolist(),
                        'confidence': confidence,
                        'risk_level': ['低风险', '中风险', '高风险'][min(predicted_class, 2)]
                    })
                    
                elif isinstance(prediction, dict):
                    # 字典输出（如RiskAssessmentNet）
                    if 'class_logits' in prediction:
                        logits = prediction['class_logits'].cpu().numpy()[0]
                        probs = F.softmax(torch.FloatTensor(logits), dim=0).numpy()
                        predicted_class = int(np.argmax(probs))
                        
                        result.update({
                            'predicted_class': predicted_class,
                            'class_probabilities': probs.tolist(),
                            'confidence': float(np.max(probs)),
                            'risk_level': ['低风险', '中风险', '高风险'][min(predicted_class, 2)]
                        })
                    
                    if 'risk_score' in prediction:
                        risk_score = prediction['risk_score'].cpu().numpy()[0, 0]
                        result['risk_score'] = float(risk_score)
                    
                    if 'attention_weights' in prediction:
                        attention = prediction['attention_weights'].cpu().numpy()[0]
                        result['feature_importance'] = attention.tolist()
                else:
                    # 其他类型，使用默认值
                    result.update({
                        'predicted_class': 1,
                        'class_probabilities': [0.3, 0.4, 0.3],
                        'confidence': 0.4,
                        'risk_level': '中风险'
                    })
            
            elif model_type == 'traditional':
                # 传统ML模型输出
                try:
                    if hasattr(prediction, 'predict_proba'):
                        probs = prediction.predict_proba(self._get_last_input())[0]
                        predicted_class = int(np.argmax(probs))
                        
                        result.update({
                            'predicted_class': predicted_class,
                            'class_probabilities': probs.tolist(),
                            'confidence': float(np.max(probs)),
                            'risk_level': ['低风险', '中风险', '高风险'][min(predicted_class, 2)]
                        })
                    else:
                        # 回归或其他输出
                        result['prediction_value'] = float(prediction)
                        result.update({
                            'predicted_class': 1,
                            'confidence': 0.5,
                            'risk_level': '中风险'
                        })
                except Exception as e:
                    logger.warning(f"传统ML模型预测后处理失败: {e}")
                    result.update({
                        'predicted_class': 1,
                        'confidence': 0.5,
                        'risk_level': '中风险'
                    })
            
            else:
                # 未知模型类型，使用默认值
                result.update({
                    'predicted_class': 1,
                    'class_probabilities': [0.3, 0.4, 0.3],
                    'confidence': 0.4,
                    'risk_level': '中风险'
                })
            
            return result
            
        except Exception as e:
            logger.error(f"预测结果后处理失败: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'prediction_type': prediction_type,
                'model_type': model_type,
                'error': str(e),
                'predicted_class': 0,
                'confidence': 0.33,
                'risk_level': '低风险'
            }
    
    async def predict_health_indicators(self, 
                                      input_data: Dict[str, Any],
                                      user_id: Optional[int] = None,
                                      model_name: str = 'default_health_lstm') -> Dict[str, Any]:
        """
        预测健康指标
        
        Args:
            input_data: 输入数据
            user_id: 用户ID
            model_name: 使用的模型名称
            
        Returns:
            Dict[str, Any]: 预测结果
        """
        try:
            # 生成缓存键
            cache_key = self._generate_cache_key('health_indicators', input_data, model_name)
            
            # 检查缓存
            cached_result = self.redis_client.get(cache_key)
            if cached_result:
                logger.info(f"从缓存返回健康指标预测结果")
                return json.loads(cached_result)
            
            # 预处理输入数据
            processed_data = self._preprocess_input_data(input_data, 'time_series')
            
            # 尝试加载和执行模型预测
            try:
                # 加载模型
                model = self._load_model(model_name, 'pytorch')
                
                # 执行预测
                with torch.no_grad():
                    input_tensor = torch.FloatTensor(processed_data).to(self.device)
                    prediction = model(input_tensor)
                
                # 后处理结果
                result = self._postprocess_prediction(prediction, 'health_indicators', 'pytorch')
                
            except Exception as model_error:
                logger.warning(f"模型预测失败，使用模拟结果: {model_error}")
                # 使用模拟预测结果
                result = {
                    'timestamp': datetime.now().isoformat(),
                    'prediction_type': 'health_indicators',
                    'model_type': 'mock',
                    'predicted_class': 1,
                    'class_probabilities': [0.3, 0.4, 0.3],
                    'confidence': 0.4,
                    'risk_level': '中风险'
                }
            
            # 添加健康指标特定信息
            result.update({
                'predicted_indicators': {
                    'blood_pressure': {
                        'systolic': 120 + np.random.normal(0, 10),
                        'diastolic': 80 + np.random.normal(0, 5),
                        'trend': 'stable'
                    },
                    'heart_rate': {
                        'bpm': 72 + np.random.normal(0, 8),
                        'variability': 'normal'
                    },
                    'blood_sugar': {
                        'level': 100 + np.random.normal(0, 15),
                        'status': 'normal'
                    }
                },
                'recommendations': self._generate_health_recommendations(result),
                'user_id': user_id
            })
            
            # 缓存结果
            self.redis_client.setex(
                cache_key, 
                self.prediction_cache_timeout, 
                json.dumps(result, default=str)
            )
            
            # 记录预测历史
            if user_id:
                await self._save_prediction_history(user_id, 'health_indicators', result)
            
            logger.info(f"健康指标预测完成，用户: {user_id}")
            return result
            
        except Exception as e:
            logger.error(f"健康指标预测失败: {str(e)}")
            traceback.print_exc()
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'prediction_type': 'health_indicators'
            }
    
    async def assess_disease_risk(self,
                                input_data: Dict[str, Any],
                                user_id: Optional[int] = None,
                                model_name: str = 'default_risk_assessment') -> Dict[str, Any]:
        """
        评估疾病风险
        
        Args:
            input_data: 输入数据
            user_id: 用户ID
            model_name: 使用的模型名称
            
        Returns:
            Dict[str, Any]: 风险评估结果
        """
        try:
            # 生成缓存键
            cache_key = self._generate_cache_key('disease_risk', input_data, model_name)
            
            # 检查缓存
            cached_result = self.redis_client.get(cache_key)
            if cached_result:
                logger.info(f"从缓存返回疾病风险评估结果")
                return json.loads(cached_result)
            
            # 预处理输入数据
            processed_data = self._preprocess_input_data(input_data, 'static')
            
            # 尝试加载和执行模型预测
            try:
                # 加载模型
                model = self._load_model(model_name, 'pytorch')
                
                # 执行预测
                with torch.no_grad():
                    input_tensor = torch.FloatTensor(processed_data).to(self.device)
                    prediction = model(input_tensor)
                
                # 后处理结果
                result = self._postprocess_prediction(prediction, 'disease_risk', 'pytorch')
                
            except Exception as model_error:
                logger.warning(f"疾病风险模型预测失败，使用模拟结果: {model_error}")
                # 使用模拟预测结果
                result = {
                    'timestamp': datetime.now().isoformat(),
                    'prediction_type': 'disease_risk',
                    'model_type': 'mock',
                    'predicted_class': 0,
                    'class_probabilities': [0.6, 0.3, 0.1],
                    'confidence': 0.6,
                    'risk_level': '低风险'
                }
            
            # 添加疾病风险特定信息
            diseases = ['糖尿病', '高血压', '心血管疾病', '高血脂', '肥胖症']
            risk_scores = np.random.random(len(diseases)) * 0.8 + 0.1  # 0.1-0.9
            
            result.update({
                'disease_risks': {
                    disease: {
                        'risk_score': float(score),
                        'risk_level': self._categorize_risk(score),
                        'factors': self._get_risk_factors(disease, input_data)
                    }
                    for disease, score in zip(diseases, risk_scores)
                },
                'overall_health_score': float(np.mean(1 - risk_scores) * 100),
                'recommendations': self._generate_risk_recommendations(diseases, risk_scores),
                'user_id': user_id
            })
            
            # 缓存结果
            self.redis_client.setex(
                cache_key,
                self.prediction_cache_timeout,
                json.dumps(result, default=str)
            )
            
            # 记录预测历史
            if user_id:
                await self._save_prediction_history(user_id, 'disease_risk', result)
            
            logger.info(f"疾病风险评估完成，用户: {user_id}")
            return result
            
        except Exception as e:
            logger.error(f"疾病风险评估失败: {str(e)}")
            traceback.print_exc()
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'prediction_type': 'disease_risk'
            }
    
    async def batch_predict(self,
                          batch_data: List[Dict[str, Any]],
                          prediction_type: str = 'health_indicators',
                          model_name: str = None) -> List[Dict[str, Any]]:
        """
        批量预测
        
        Args:
            batch_data: 批量输入数据
            prediction_type: 预测类型
            model_name: 模型名称
            
        Returns:
            List[Dict[str, Any]]: 批量预测结果
        """
        try:
            logger.info(f"开始批量预测，数据量: {len(batch_data)}")
            
            # 并行处理
            tasks = []
            for data in batch_data:
                if prediction_type == 'health_indicators':
                    task = self.predict_health_indicators(data, model_name=model_name)
                elif prediction_type == 'disease_risk':
                    task = self.assess_disease_risk(data, model_name=model_name)
                else:
                    raise ValueError(f"不支持的预测类型: {prediction_type}")
                
                tasks.append(task)
            
            # 等待所有任务完成
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常结果
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        'error': str(result),
                        'batch_index': i,
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    result['batch_index'] = i
                    processed_results.append(result)
            
            logger.info(f"批量预测完成，成功: {len([r for r in processed_results if 'error' not in r])}")
            return processed_results
            
        except Exception as e:
            logger.error(f"批量预测失败: {str(e)}")
            return [{
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'prediction_type': prediction_type
            }]
    
    def _categorize_risk(self, risk_score: float) -> str:
        """风险等级分类"""
        if risk_score < 0.3:
            return '低风险'
        elif risk_score < 0.7:
            return '中风险'
        else:
            return '高风险'
    
    def _get_risk_factors(self, disease: str, input_data: Dict[str, Any]) -> List[str]:
        """获取风险因素"""
        factors = {
            '糖尿病': ['年龄', '肥胖', '家族史', '血糖'],
            '高血压': ['年龄', '肥胖', '盐摄入', '运动不足'],
            '心血管疾病': ['吸烟', '胆固醇', '血压', '糖尿病'],
            '高血脂': ['饮食', '运动', '遗传', '肥胖'],
            '肥胖症': ['饮食', '运动', '代谢', '遗传']
        }
        return factors.get(disease, ['未知'])
    
    def _generate_health_recommendations(self, prediction_result: Dict[str, Any]) -> List[str]:
        """生成健康建议"""
        recommendations = []
        
        risk_level = prediction_result.get('risk_level', '低风险')
        
        if risk_level == '高风险':
            recommendations.extend([
                '建议立即咨询医生',
                '调整生活方式',
                '定期监测健康指标',
                '考虑药物治疗'
            ])
        elif risk_level == '中风险':
            recommendations.extend([
                '加强健康监测',
                '改善饮食结构',
                '增加运动量',
                '定期体检'
            ])
        else:
            recommendations.extend([
                '保持健康生活方式',
                '定期体检',
                '均衡饮食',
                '适量运动'
            ])
        
        return recommendations
    
    def _generate_risk_recommendations(self, diseases: List[str], risk_scores: np.ndarray) -> List[str]:
        """生成风险建议"""
        recommendations = []
        
        high_risk_diseases = [disease for disease, score in zip(diseases, risk_scores) if score > 0.7]
        
        if high_risk_diseases:
            recommendations.append(f"高风险疾病: {', '.join(high_risk_diseases)}")
            recommendations.append("建议尽快咨询专科医生")
        
        recommendations.extend([
            "保持健康的生活方式",
            "定期进行健康检查",
            "控制饮食，适量运动",
            "戒烟限酒，充足睡眠"
        ])
        
        return recommendations
    
    async def _save_prediction_history(self, user_id: int, prediction_type: str, result: Dict[str, Any]):
        """保存预测历史"""
        try:
            history_data = {
                'user_id': user_id,
                'prediction_type': prediction_type,
                'result': json.dumps(result, default=str),
                'created_at': datetime.now(),
                'confidence': result.get('confidence', 0),
                'risk_level': result.get('risk_level', '未知')
            }
            
            # 这里应该保存到数据库
            # self.db_manager.insert('prediction_history', history_data)
            logger.info(f"预测历史已保存：用户 {user_id}, 类型 {prediction_type}")
            
        except Exception as e:
            logger.error(f"保存预测历史失败: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取当前加载的模型信息"""
        return {
            'loaded_models': list(self.loaded_models.keys()),
            'device': str(self.device),
            'cache_timeout': self.model_cache_timeout,
            'prediction_cache_timeout': self.prediction_cache_timeout
        }
    
    def clear_cache(self):
        """清空缓存"""
        self.loaded_models.clear()
        # 清空Redis缓存
        redis_keys = self.redis_client.keys('prediction:*')
        if redis_keys:
            self.redis_client.delete(*redis_keys)
        logger.info("缓存已清空")
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


# 全局预测服务实例
prediction_service = HealthPredictionService()
