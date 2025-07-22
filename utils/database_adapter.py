"""
数据库适配器 - 兼容现有SHEC-PSIMS数据库结构
将现有的复杂表结构映射到AI服务所需的简化接口
"""

from typing import Dict, List, Optional, Any
import logging
from utils.database import DatabaseManager

logger = logging.getLogger(__name__)

class SHECDatabaseAdapter:
    """SHEC数据库适配器类"""
    
    def __init__(self):
        self.db = DatabaseManager()
    
    def get_patient_info(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        获取患者基本信息
        映射 user 表 -> AI服务的patient概念
        """
        query = """
        SELECT 
            u.user_id as patient_id,
            u.username,
            u.real_name,
            u.age,
            u.gender,
            u.phone,
            u.email,
            u.id_card,
            u.created_at,
            u.updated_at
        FROM user u 
        WHERE u.user_id = %s AND u.user_type = 'patient'
        """
        
        result = self.db.fetch_one(query, (user_id,))
        return dict(result) if result else None
    
    def get_health_records(self, patient_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取健康记录
        映射 health_record 表 -> AI服务的health_metrics概念
        """
        query = """
        SELECT 
            hr.health_record_id as metric_id,
            hr.patient_id,
            hr.systolic_pressure,
            hr.diastolic_pressure, 
            hr.blood_sugar,
            hr.blood_sugar_type,
            hr.bmi,
            hr.weight,
            hr.height,
            hr.heart_rate,
            hr.body_temperature,
            hr.exercise_frequency,
            hr.smoking_status,
            hr.drinking_status,
            hr.medication_usage,
            hr.measurement_time,
            hr.created_at
        FROM health_record hr
        WHERE hr.patient_id = %s
        ORDER BY hr.measurement_time DESC
        LIMIT %s
        """
        
        results = self.db.fetch_all(query, (patient_id, limit))
        return [dict(row) for row in results] if results else []
    
    def get_medical_history(self, patient_id: int) -> List[Dict[str, Any]]:
        """
        获取医疗记录历史
        映射 medical_record 表
        """
        query = """
        SELECT 
            mr.record_id,
            mr.patient_id,
            mr.diagnosis,
            mr.treatment_plan,
            mr.medication,
            mr.doctor_notes,
            mr.record_date,
            mr.created_at
        FROM medical_record mr
        WHERE mr.patient_id = %s
        ORDER BY mr.record_date DESC
        """
        
        results = self.db.fetch_all(query, (patient_id,))
        return [dict(row) for row in results] if results else []
    
    def save_ai_prediction(self, patient_id: int, prediction_data: Dict[str, Any]) -> int:
        """
        保存AI预测结果
        使用现有的 ai_prediction_results 表
        """
        query = """
        INSERT INTO ai_prediction_results (
            patient_id, prediction_type, model_name, model_version,
            input_data, prediction_result, confidence_score, 
            risk_level, recommendations, expires_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        params = (
            patient_id,
            prediction_data.get('prediction_type'),
            prediction_data.get('model_name'),
            prediction_data.get('model_version'),
            prediction_data.get('input_data'),  # JSON
            prediction_data.get('prediction_result'),  # JSON
            prediction_data.get('confidence_score'),
            prediction_data.get('risk_level'),
            prediction_data.get('recommendations'),  # JSON
            prediction_data.get('expires_at')
        )
        
        return self.db.execute_query(query, params)
    
    def get_ai_predictions(self, patient_id: int, prediction_type: str = None) -> List[Dict[str, Any]]:
        """
        获取AI预测历史
        """
        base_query = """
        SELECT 
            apr.id as result_id,
            apr.patient_id,
            apr.prediction_type,
            apr.model_name,
            apr.model_version,
            apr.input_data,
            apr.prediction_result,
            apr.confidence_score,
            apr.risk_level,
            apr.recommendations,
            apr.created_at
        FROM ai_prediction_results apr
        WHERE apr.patient_id = %s
        """
        
        params = [patient_id]
        
        if prediction_type:
            base_query += " AND apr.prediction_type = %s"
            params.append(prediction_type)
        
        base_query += " ORDER BY apr.created_at DESC"
        
        results = self.db.fetch_all(base_query, params)
        return [dict(row) for row in results] if results else []
    
    def get_active_ai_models(self, model_type: str = None) -> List[Dict[str, Any]]:
        """
        获取激活的AI模型配置
        """
        base_query = """
        SELECT 
            am.id as model_id,
            am.model_name,
            am.model_version,
            am.model_type,
            am.description,
            am.model_path,
            am.config_data,
            am.performance_metrics,
            am.is_active,
            am.status
        FROM ai_models am
        WHERE am.is_active = true
        """
        
        params = []
        
        if model_type:
            base_query += " AND am.model_type = %s"
            params.append(model_type)
        
        base_query += " ORDER BY am.created_at DESC"
        
        results = self.db.fetch_all(base_query, params)
        return [dict(row) for row in results] if results else []
    
    def get_patients_for_prediction(self, risk_threshold: str = '中') -> List[Dict[str, Any]]:
        """
        获取需要进行AI预测的患者列表
        结合用户表和最近的预测结果
        """
        query = """
        SELECT DISTINCT
            u.user_id as patient_id,
            u.username,
            u.real_name,
            u.age,
            u.gender,
            COALESCE(latest_pred.risk_level, '未知') as last_risk_level,
            COALESCE(latest_pred.created_at, u.created_at) as last_prediction_time,
            hr_count.record_count
        FROM user u
        LEFT JOIN (
            SELECT 
                patient_id,
                risk_level,
                created_at,
                ROW_NUMBER() OVER (PARTITION BY patient_id ORDER BY created_at DESC) as rn
            FROM ai_prediction_results
        ) latest_pred ON u.user_id = latest_pred.patient_id AND latest_pred.rn = 1
        LEFT JOIN (
            SELECT patient_id, COUNT(*) as record_count
            FROM health_record 
            WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            GROUP BY patient_id
        ) hr_count ON u.user_id = hr_count.patient_id
        WHERE u.user_type = 'patient'
          AND u.status = 'active'
          AND (
              latest_pred.risk_level IS NULL 
              OR latest_pred.risk_level IN ('中', '高', '极高')
              OR latest_pred.created_at < DATE_SUB(NOW(), INTERVAL 7 DAY)
          )
        ORDER BY 
            CASE latest_pred.risk_level 
                WHEN '极高' THEN 1
                WHEN '高' THEN 2
                WHEN '中' THEN 3
                ELSE 4
            END,
            latest_pred.created_at ASC
        """
        
        results = self.db.fetch_all(query)
        return [dict(row) for row in results] if results else []

# 全局适配器实例
db_adapter = SHECDatabaseAdapter()
