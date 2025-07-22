# 数据预处理管道
# 基于现有数据库结构的智能健康数据处理系统

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import KNNImputer, SimpleImputer
import logging
from dataclasses import dataclass
import json

from utils.database import DatabaseManager, AIDataAccess
from utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class HealthDataConfig:
    """健康数据预处理配置"""
    # 数值范围配置
    BLOOD_PRESSURE_RANGE = {"systolic": (70, 200), "diastolic": (40, 130)}
    BLOOD_SUGAR_RANGE = (3.0, 30.0)  # mmol/L
    BMI_RANGE = (10.0, 50.0)
    HEART_RATE_RANGE = (30, 220)
    TEMPERATURE_RANGE = (35.0, 42.0)  # 摄氏度
    
    # 分类特征映射
    GENDER_MAPPING = {"M": 1, "F": 0, "男": 1, "女": 0}
    RISK_LEVEL_MAPPING = {"低": 0, "中": 1, "高": 2, "极高": 3}
    
    # 特征重要性权重
    FEATURE_WEIGHTS = {
        "age": 0.15,
        "systolic_pressure": 0.20,
        "diastolic_pressure": 0.15,
        "blood_sugar": 0.18,
        "bmi": 0.12,
        "heart_rate": 0.10,
        "exercise_frequency": 0.10
    }

class HealthDataCleaner:
    """健康数据清洗器"""
    
    def __init__(self, config: HealthDataConfig = None):
        self.config = config or HealthDataConfig()
        self.logger = logger
    
    def clean_numerical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗数值数据"""
        df_clean = df.copy()
        
        # 血压清洗
        if 'systolic_pressure' in df_clean.columns:
            sys_range = self.config.BLOOD_PRESSURE_RANGE["systolic"]
            df_clean['systolic_pressure'] = df_clean['systolic_pressure'].clip(
                lower=sys_range[0], upper=sys_range[1]
            )
            
        if 'diastolic_pressure' in df_clean.columns:
            dia_range = self.config.BLOOD_PRESSURE_RANGE["diastolic"]
            df_clean['diastolic_pressure'] = df_clean['diastolic_pressure'].clip(
                lower=dia_range[0], upper=dia_range[1]
            )
        
        # 血糖清洗
        if 'blood_sugar' in df_clean.columns:
            sugar_range = self.config.BLOOD_SUGAR_RANGE
            df_clean['blood_sugar'] = df_clean['blood_sugar'].clip(
                lower=sugar_range[0], upper=sugar_range[1]
            )
        
        # BMI清洗
        if 'bmi' in df_clean.columns:
            bmi_range = self.config.BMI_RANGE
            df_clean['bmi'] = df_clean['bmi'].clip(
                lower=bmi_range[0], upper=bmi_range[1]
            )
        
        # 心率清洗
        if 'heart_rate' in df_clean.columns:
            hr_range = self.config.HEART_RATE_RANGE
            df_clean['heart_rate'] = df_clean['heart_rate'].clip(
                lower=hr_range[0], upper=hr_range[1]
            )
        
        self.logger.info("数值数据清洗完成")
        return df_clean
    
    def clean_categorical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗分类数据"""
        df_clean = df.copy()
        
        # 性别标准化
        if 'gender' in df_clean.columns:
            df_clean['gender'] = df_clean['gender'].map(
                self.config.GENDER_MAPPING
            ).fillna(0)  # 默认为女性
        
        # 移除异常字符
        text_columns = df_clean.select_dtypes(include=['object']).columns
        for col in text_columns:
            df_clean[col] = df_clean[col].astype(str).str.strip()
            df_clean[col] = df_clean[col].replace(['', 'None', 'null', 'NULL'], np.nan)
        
        self.logger.info("分类数据清洗完成")
        return df_clean
    
    def remove_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """移除异常值"""
        df_clean = df.copy()
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        if method == 'iqr':
            for col in numerical_cols:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df_clean[col] = df_clean[col].clip(
                    lower=lower_bound, upper=upper_bound
                )
        
        elif method == 'zscore':
            for col in numerical_cols:
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                df_clean = df_clean[z_scores < 3]
        
        self.logger.info(f"使用{method}方法移除异常值完成")
        return df_clean

class HealthFeatureEngineer:
    """健康数据特征工程器"""
    
    def __init__(self, config: HealthDataConfig = None):
        self.config = config or HealthDataConfig()
        self.logger = logger
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建派生特征"""
        df_enhanced = df.copy()
        
        # 血压相关特征
        if 'systolic_pressure' in df_enhanced.columns and 'diastolic_pressure' in df_enhanced.columns:
            # 脉压差
            df_enhanced['pulse_pressure'] = df_enhanced['systolic_pressure'] - df_enhanced['diastolic_pressure']
            
            # 平均动脉压
            df_enhanced['mean_arterial_pressure'] = (
                df_enhanced['diastolic_pressure'] + 
                (df_enhanced['pulse_pressure'] / 3)
            )
            
            # 高血压风险等级
            df_enhanced['hypertension_risk'] = self._calculate_hypertension_risk(
                df_enhanced['systolic_pressure'], 
                df_enhanced['diastolic_pressure']
            )
        
        # BMI相关特征
        # 如果没有BMI但有身高体重，自动计算BMI
        if 'bmi' not in df_enhanced.columns and 'weight' in df_enhanced.columns and 'height' in df_enhanced.columns:
            # 计算BMI: weight(kg) / (height(cm)/100)^2
            height_m = df_enhanced['height'] / 100  # 转换为米
            df_enhanced['bmi'] = df_enhanced['weight'] / (height_m ** 2)
            self.logger.info("自动计算BMI完成")
        
        if 'bmi' in df_enhanced.columns:
            df_enhanced['bmi_category'] = self._categorize_bmi(df_enhanced['bmi'])
            df_enhanced['obesity_risk'] = (df_enhanced['bmi'] >= 30).astype(int)
        
        # 年龄相关特征
        if 'age' in df_enhanced.columns:
            df_enhanced['age_group'] = pd.cut(
                df_enhanced['age'], 
                bins=[0, 18, 30, 45, 60, 100], 
                labels=['青少年', '青年', '中年', '中老年', '老年']
            )
            df_enhanced['senior_risk'] = (df_enhanced['age'] >= 65).astype(int)
        
        # 综合健康评分
        df_enhanced['health_score'] = self._calculate_health_score(df_enhanced)
        
        self.logger.info("派生特征创建完成")
        return df_enhanced
    
    def _calculate_hypertension_risk(self, systolic: pd.Series, diastolic: pd.Series) -> pd.Series:
        """计算高血压风险等级"""
        conditions = [
            (systolic < 120) & (diastolic < 80),  # 正常
            (systolic < 130) & (diastolic < 80),  # 正常偏高
            ((systolic >= 130) & (systolic < 140)) | ((diastolic >= 80) & (diastolic < 90)),  # 1级高血压
            ((systolic >= 140) & (systolic < 160)) | ((diastolic >= 90) & (diastolic < 100)),  # 2级高血压
            (systolic >= 160) | (diastolic >= 100)  # 3级高血压
        ]
        choices = [0, 1, 2, 3, 4]  # 风险等级
        return pd.Series(np.select(conditions, choices, default=2))
    
    def _categorize_bmi(self, bmi: pd.Series) -> pd.Series:
        """BMI分类"""
        conditions = [
            bmi < 18.5,  # 偏瘦
            (bmi >= 18.5) & (bmi < 24),  # 正常
            (bmi >= 24) & (bmi < 28),  # 超重
            bmi >= 28  # 肥胖
        ]
        choices = ['偏瘦', '正常', '超重', '肥胖']
        return pd.Series(np.select(conditions, choices, default='正常'))
    
    def _calculate_health_score(self, df: pd.DataFrame) -> pd.Series:
        """计算综合健康评分 (0-100)"""
        score = pd.Series(50.0, index=df.index)  # 基础分50
        
        # 血压评分
        if 'hypertension_risk' in df.columns:
            score -= df['hypertension_risk'] * 15  # 高血压风险扣分
        
        # BMI评分
        if 'bmi' in df.columns:
            bmi_optimal = (df['bmi'] >= 18.5) & (df['bmi'] < 24)
            score += bmi_optimal * 20 - (~bmi_optimal) * 10
        
        # 年龄评分
        if 'age' in df.columns:
            score -= (df['age'] - 30).clip(0) * 0.3  # 年龄增长扣分
        
        # 运动评分
        if 'exercise_frequency' in df.columns:
            score += df['exercise_frequency'] * 5
        
        return score.clip(0, 100)
    
    def create_time_features(self, df: pd.DataFrame, time_col: str = 'created_at') -> pd.DataFrame:
        """创建时间特征"""
        if time_col not in df.columns:
            return df
            
        df_time = df.copy()
        df_time[time_col] = pd.to_datetime(df_time[time_col])
        
        # 基础时间特征
        df_time['year'] = df_time[time_col].dt.year
        df_time['month'] = df_time[time_col].dt.month
        df_time['day'] = df_time[time_col].dt.day
        df_time['weekday'] = df_time[time_col].dt.weekday
        df_time['hour'] = df_time[time_col].dt.hour
        
        # 季节特征
        df_time['season'] = df_time['month'].map({
            12: '冬', 1: '冬', 2: '冬',
            3: '春', 4: '春', 5: '春',
            6: '夏', 7: '夏', 8: '夏',
            9: '秋', 10: '秋', 11: '秋'
        })
        
        # 是否工作日
        df_time['is_weekend'] = (df_time['weekday'] >= 5).astype(int)
        
        self.logger.info("时间特征创建完成")
        return df_time

class HealthDataNormalizer:
    """健康数据标准化器"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.logger = logger
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """训练并转换数据"""
        df_normalized = df.copy()
        
        # 处理数值特征
        numerical_cols = df_normalized.select_dtypes(include=[np.number]).columns
        if target_col and target_col in numerical_cols:
            numerical_cols = numerical_cols.drop(target_col)
        
        if len(numerical_cols) > 0:
            self.scalers['standard'] = StandardScaler()
            df_normalized[numerical_cols] = self.scalers['standard'].fit_transform(
                df_normalized[numerical_cols]
            )
        
        # 处理分类特征
        categorical_cols = df_normalized.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_normalized[col].nunique() < 100:  # 避免高基数分类变量
                self.encoders[col] = LabelEncoder()
                df_normalized[col] = self.encoders[col].fit_transform(
                    df_normalized[col].astype(str)
                )
        
        self.logger.info("数据标准化完成")
        return df_normalized
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """使用已训练的转换器转换新数据"""
        df_normalized = df.copy()
        
        # 数值特征标准化
        if 'standard' in self.scalers:
            numerical_cols = df_normalized.select_dtypes(include=[np.number]).columns
            numerical_cols = [col for col in numerical_cols if col in self.scalers['standard'].feature_names_in_]
            if len(numerical_cols) > 0:
                df_normalized[numerical_cols] = self.scalers['standard'].transform(
                    df_normalized[numerical_cols]
                )
        
        # 分类特征编码
        for col, encoder in self.encoders.items():
            if col in df_normalized.columns:
                try:
                    df_normalized[col] = encoder.transform(df_normalized[col].astype(str))
                except ValueError:
                    # 处理未见过的类别
                    self.logger.warning(f"列 {col} 包含未见过的类别，使用默认值")
                    df_normalized[col] = 0
        
        return df_normalized
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'knn') -> pd.DataFrame:
        """处理缺失值"""
        df_imputed = df.copy()
        
        if strategy == 'knn':
            # 数值特征使用KNN插补
            numerical_cols = df_imputed.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                self.imputers['knn'] = KNNImputer(n_neighbors=5)
                df_imputed[numerical_cols] = self.imputers['knn'].fit_transform(
                    df_imputed[numerical_cols]
                )
            
            # 分类特征使用众数插补
            categorical_cols = df_imputed.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                mode_value = df_imputed[col].mode()
                if len(mode_value) > 0:
                    df_imputed[col].fillna(mode_value[0], inplace=True)
        
        elif strategy == 'median':
            self.imputers['median'] = SimpleImputer(strategy='median')
            numerical_cols = df_imputed.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                df_imputed[numerical_cols] = self.imputers['median'].fit_transform(
                    df_imputed[numerical_cols]
                )
        
        self.logger.info(f"使用{strategy}策略处理缺失值完成")
        return df_imputed

class HealthDataPipeline:
    """健康数据预处理管道"""
    
    def __init__(self, config: HealthDataConfig = None):
        self.config = config or HealthDataConfig()
        self.cleaner = HealthDataCleaner(config)
        self.feature_engineer = HealthFeatureEngineer(config)
        self.normalizer = HealthDataNormalizer()
        self.db_access = AIDataAccess()
        self.logger = logger
        
        # 管道步骤配置
        self.steps = [
            'load_data',
            'clean_data',
            'handle_missing',
            'feature_engineering',
            'normalize_data',
            'validate_data'
        ]
    
    def load_patient_data(self, patient_id: int = None, 
                         start_date: datetime = None,
                         end_date: datetime = None) -> pd.DataFrame:
        """从数据库加载患者健康数据"""
        try:
            # 获取患者健康指标数据
            health_metrics = self.db_access.get_patient_health_metrics(
                patient_id=patient_id,
                start_date=start_date,
                end_date=end_date
            )
            
            if not health_metrics:
                self.logger.warning(f"未找到患者 {patient_id} 的健康数据")
                return pd.DataFrame()
            
            # 转换为DataFrame
            df = pd.DataFrame(health_metrics)
            
            # 添加用户基础信息
            if patient_id:
                user_info = self.db_access.get_user_by_id(patient_id)
                if user_info:
                    df['age'] = user_info.get('age')
                    df['gender'] = user_info.get('gender')
            
            self.logger.info(f"加载患者数据完成，共 {len(df)} 条记录")
            return df
            
        except Exception as e:
            self.logger.error(f"加载患者数据失败: {e}")
            return pd.DataFrame()
    
    def process(self, df: pd.DataFrame, target_col: str = None,
                training_mode: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """执行完整的数据预处理管道"""
        
        pipeline_stats = {
            'original_shape': df.shape,
            'steps_completed': [],
            'processing_time': {},
            'data_quality': {}
        }
        
        start_time = datetime.now()
        processed_df = df.copy()
        
        # 1. 数据清洗
        step_start = datetime.now()
        processed_df = self.cleaner.clean_numerical_data(processed_df)
        processed_df = self.cleaner.clean_categorical_data(processed_df)
        processed_df = self.cleaner.remove_outliers(processed_df)
        pipeline_stats['processing_time']['clean_data'] = (datetime.now() - step_start).total_seconds()
        pipeline_stats['steps_completed'].append('clean_data')
        
        # 2. 处理缺失值
        step_start = datetime.now()
        processed_df = self.normalizer.handle_missing_values(processed_df)
        pipeline_stats['processing_time']['handle_missing'] = (datetime.now() - step_start).total_seconds()
        pipeline_stats['steps_completed'].append('handle_missing')
        
        # 3. 特征工程
        step_start = datetime.now()
        processed_df = self.feature_engineer.create_derived_features(processed_df)
        if 'created_at' in processed_df.columns:
            processed_df = self.feature_engineer.create_time_features(processed_df)
        pipeline_stats['processing_time']['feature_engineering'] = (datetime.now() - step_start).total_seconds()
        pipeline_stats['steps_completed'].append('feature_engineering')
        
        # 4. 数据标准化
        step_start = datetime.now()
        if training_mode:
            processed_df = self.normalizer.fit_transform(processed_df, target_col)
        else:
            processed_df = self.normalizer.transform(processed_df)
        pipeline_stats['processing_time']['normalize_data'] = (datetime.now() - step_start).total_seconds()
        pipeline_stats['steps_completed'].append('normalize_data')
        
        # 5. 数据验证
        step_start = datetime.now()
        quality_stats = self._validate_processed_data(processed_df)
        pipeline_stats['data_quality'] = quality_stats
        pipeline_stats['processing_time']['validate_data'] = (datetime.now() - step_start).total_seconds()
        pipeline_stats['steps_completed'].append('validate_data')
        
        # 总体统计
        pipeline_stats['final_shape'] = processed_df.shape
        pipeline_stats['total_time'] = (datetime.now() - start_time).total_seconds()
        
        self.logger.info(f"数据预处理管道完成，耗时 {pipeline_stats['total_time']:.2f} 秒")
        
        return processed_df, pipeline_stats
    
    def _validate_processed_data(self, df: pd.DataFrame) -> Dict:
        """验证处理后的数据质量"""
        stats = {
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'shape': df.shape,
            'duplicate_rows': df.duplicated().sum(),
            'numerical_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(include=['object']).columns)
        }
        
        # 检查数据质量问题
        quality_issues = []
        
        if stats['duplicate_rows'] > 0:
            quality_issues.append(f"发现 {stats['duplicate_rows']} 行重复数据")
        
        missing_cols = [col for col, count in stats['missing_values'].items() if count > 0]
        if missing_cols:
            quality_issues.append(f"以下列仍有缺失值: {missing_cols}")
        
        stats['quality_issues'] = quality_issues
        return stats
    
    def prepare_for_training(self, patient_ids: List[int] = None, 
                           days_back: int = 90) -> Tuple[pd.DataFrame, Dict]:
        """为模型训练准备数据"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        all_data = []
        
        if patient_ids:
            for patient_id in patient_ids:
                patient_data = self.load_patient_data(
                    patient_id=patient_id,
                    start_date=start_date,
                    end_date=end_date
                )
                if not patient_data.empty:
                    all_data.append(patient_data)
        else:
            # 加载所有患者数据
            all_patients_data = self.db_access.get_all_health_metrics(
                start_date=start_date,
                end_date=end_date
            )
            if all_patients_data:
                all_data = [pd.DataFrame(all_patients_data)]
        
        if not all_data:
            raise ValueError("未找到可用的训练数据")
        
        # 合并所有数据
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # 处理数据
        processed_df, stats = self.process(combined_df, training_mode=True)
        
        self.logger.info(f"训练数据准备完成: {processed_df.shape}")
        return processed_df, stats
    
    def prepare_for_prediction(self, patient_id: int) -> Tuple[pd.DataFrame, Dict]:
        """为预测准备单个患者的数据"""
        # 获取最近的健康数据
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # 最近30天数据
        
        patient_data = self.load_patient_data(
            patient_id=patient_id,
            start_date=start_date,
            end_date=end_date
        )
        
        if patient_data.empty:
            raise ValueError(f"未找到患者 {patient_id} 的健康数据")
        
        # 使用已训练的管道处理数据
        processed_df, stats = self.process(patient_data, training_mode=False)
        
        self.logger.info(f"预测数据准备完成: {processed_df.shape}")
        return processed_df, stats


# 使用示例和测试函数
def example_usage():
    """数据预处理管道使用示例"""
    
    # 初始化管道
    pipeline = HealthDataPipeline()
    
    try:
        # 1. 为训练准备数据
        print("=== 训练数据准备 ===")
        train_data, train_stats = pipeline.prepare_for_training(days_back=180)
        print(f"训练数据形状: {train_data.shape}")
        print(f"处理时间: {train_stats['total_time']:.2f} 秒")
        
        # 2. 为预测准备数据
        print("\n=== 预测数据准备 ===")
        pred_data, pred_stats = pipeline.prepare_for_prediction(patient_id=1)
        print(f"预测数据形状: {pred_data.shape}")
        print(f"处理时间: {pred_stats['total_time']:.2f} 秒")
        
        # 3. 数据质量报告
        print("\n=== 数据质量报告 ===")
        print(f"缺失值情况: {pred_stats['data_quality']['missing_values']}")
        print(f"数据类型: {pred_stats['data_quality']['data_types']}")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")

if __name__ == "__main__":
    example_usage()
