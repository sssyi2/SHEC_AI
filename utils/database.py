"""
数据库连接管理模块 - 适配现有SHEC_PSIMS数据库结构
提供MySQL数据库连接和操作功能
"""

import mysql.connector
from mysql.connector import pooling
import logging
import json
from contextlib import contextmanager
from datetime import datetime, date

# 全局连接池
connection_pool = None

def init_database(app):
    """初始化数据库连接池"""
    global connection_pool
    
    try:
        # 数据库配置
        db_config = {
            'host': app.config['DB_HOST'],
            'port': app.config['DB_PORT'],
            'database': app.config['DB_NAME'],
            'user': app.config['DB_USER'],
            'password': app.config['DB_PASSWORD'],
            'charset': app.config.get('DB_CHARSET', 'utf8mb4'),
            'autocommit': True,
            'use_unicode': True,
            'sql_mode': 'STRICT_TRANS_TABLES',
        }
        
        # 创建连接池 - 压力测试优化配置
        connection_pool = pooling.MySQLConnectionPool(
            pool_name="shec_ai_pool",
            pool_size=20,              # 增加连接池大小
            pool_reset_session=False,  # 减少重置开销
            connection_timeout=10,     # 连接超时
            **db_config
        )
        
        # 测试连接
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            
        app.logger.info("数据库连接池初始化成功")
        
        # 检查AI相关表是否存在
        DatabaseManager.check_ai_tables()
        
    except Exception as e:
        app.logger.error(f"数据库连接初始化失败: {str(e)}")
        raise e

@contextmanager
def get_db_connection():
    """获取数据库连接上下文管理器"""
    connection = None
    try:
        connection = connection_pool.get_connection()
        yield connection
    except Exception as e:
        if connection:
            connection.rollback()
        logging.error(f"数据库操作错误: {str(e)}")
        raise e
    finally:
        if connection:
            connection.close()

def execute_query(query, params=None, fetch_one=False, fetch_all=True):
    """执行查询语句"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, params or ())
            
            if query.strip().upper().startswith('SELECT'):
                if fetch_one:
                    return cursor.fetchone()
                elif fetch_all:
                    return cursor.fetchall()
                else:
                    return cursor
            else:
                conn.commit()
                return cursor.rowcount
                
    except Exception as e:
        logging.error(f"查询执行错误: {str(e)}")
        raise e

def execute_many(query, params_list):
    """批量执行语句"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            conn.commit()
            return cursor.rowcount
            
    except Exception as e:
        logging.error(f"批量执行错误: {str(e)}")
        raise e

class DatabaseManager:
    """数据库管理类 - 适配现有schema"""
    
    @staticmethod
    def check_ai_tables():
        """检查AI相关表是否存在"""
        required_tables = [
            'ai_models', 'ai_prediction_results', 'patient_health_metrics',
            'ai_model_cache_config', 'ai_model_performance', 'patient_ai_preferences',
            'prediction_tasks', 'prediction_analysis'
        ]
        
        try:
            missing_tables = []
            for table in required_tables:
                query = """
                SELECT COUNT(*) as count FROM information_schema.tables 
                WHERE table_schema = DATABASE() AND table_name = %s
                """
                result = execute_query(query, (table,), fetch_one=True)
                if result['count'] == 0:
                    missing_tables.append(table)
            
            if missing_tables:
                logging.warning(f"缺少AI相关表: {missing_tables}")
            else:
                logging.info("所有AI相关表检查完成")
                
        except Exception as e:
            logging.error(f"检查AI表时出错: {str(e)}")
    
    @staticmethod
    def get_user_by_id(user_id):
        """根据用户ID获取用户信息（适配现有user表）"""
        query = """
        SELECT user_id, user_name, age, gender, email, phone_number, 
               real_name, address, department 
        FROM user WHERE user_id = %s
        """
        return execute_query(query, (user_id,), fetch_one=True)
    
    @staticmethod
    def get_user_by_username(username):
        """根据用户名获取用户信息"""
        query = """
        SELECT user_id, user_name, age, gender, email, phone_number, 
               real_name, password_hash
        FROM user WHERE user_name = %s
        """
        return execute_query(query, (username,), fetch_one=True)
    
    @staticmethod
    def get_health_record(patient_id):
        """获取患者健康记录（使用现有health_record表）"""
        query = """
        SELECT health_record_id, patient_id, name, gender, age, 
               height, weight, blood_pressure, heart_rate,
               admission_date, discharge_date, status, 
               last_prediction_date, risk_level, ai_score
        FROM health_record WHERE patient_id = %s
        """
        return execute_query(query, (patient_id,), fetch_one=True)
    
    @staticmethod
    def get_patient_health_metrics(patient_id, limit=10):
        """获取患者健康指标历史数据"""
        query = """
        SELECT id, patient_id, record_date, age, gender, 
               disease_name, systolic_pressure, diastolic_pressure, 
               blood_sugar, bmi, other_metrics, data_source, created_at
        FROM patient_health_metrics 
        WHERE patient_id = %s 
        ORDER BY record_date DESC, created_at DESC
        LIMIT %s
        """
        return execute_query(query, (patient_id, limit))
    
    @staticmethod
    def save_health_metrics(patient_id, metrics_data):
        """保存患者健康指标数据"""
        query = """
        INSERT INTO patient_health_metrics 
        (patient_id, record_date, age, gender, disease_name,
         systolic_pressure, diastolic_pressure, blood_sugar, bmi, 
         other_metrics, data_source)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        params = (
            patient_id,
            metrics_data.get('record_date', date.today()),
            metrics_data.get('age'),
            metrics_data.get('gender'),
            metrics_data.get('disease_name'),
            metrics_data.get('systolic_pressure'),
            metrics_data.get('diastolic_pressure'),
            metrics_data.get('blood_sugar'),
            metrics_data.get('bmi'),
            json.dumps(metrics_data.get('other_metrics')) if metrics_data.get('other_metrics') else None,
            metrics_data.get('data_source', 'ai_prediction')
        )
        
        return execute_query(query, params, fetch_all=False)
    
    @staticmethod
    def save_prediction_result(prediction_data):
        """保存AI预测结果（使用现有ai_prediction_results表）"""
        query = """
        INSERT INTO ai_prediction_results 
        (patient_id, prediction_id, prediction_type, input_data, 
         prediction_data, confidence_score, model_version, prediction_period)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        params = (
            prediction_data.get('patient_id'),
            prediction_data.get('prediction_id'),
            prediction_data.get('prediction_type'),
            json.dumps(prediction_data.get('input_data')),      # JSON字段
            json.dumps(prediction_data.get('prediction_data')), # JSON字段
            prediction_data.get('confidence_score'),
            prediction_data.get('model_version'),
            prediction_data.get('prediction_period', 30)
        )
        
        return execute_query(query, params, fetch_all=False)
    
    @staticmethod
    def get_prediction_results(patient_id, prediction_type=None, limit=10):
        """获取患者的预测结果"""
        if prediction_type:
            query = """
            SELECT id, patient_id, prediction_id, prediction_type, 
                   input_data, prediction_data, confidence_score, 
                   model_version, created_at
            FROM ai_prediction_results 
            WHERE patient_id = %s AND prediction_type = %s
            ORDER BY created_at DESC LIMIT %s
            """
            params = (patient_id, prediction_type, limit)
        else:
            query = """
            SELECT id, patient_id, prediction_id, prediction_type, 
                   input_data, prediction_data, confidence_score, 
                   model_version, created_at
            FROM ai_prediction_results 
            WHERE patient_id = %s
            ORDER BY created_at DESC LIMIT %s
            """
            params = (patient_id, limit)
        
        return execute_query(query, params)
    
    @staticmethod
    def update_health_record_ai_info(patient_id, ai_score, risk_level):
        """更新健康记录的AI相关信息"""
        query = """
        UPDATE health_record 
        SET ai_score = %s, risk_level = %s, last_prediction_date = NOW()
        WHERE patient_id = %s
        """
        return execute_query(query, (ai_score, risk_level, patient_id), fetch_all=False)
    
    @staticmethod
    def get_ai_models():
        """获取可用的AI模型列表"""
        query = """
        SELECT id, model_name, model_version, model_type, 
               model_path, configuration, is_active, created_at
        FROM ai_models 
        WHERE is_active = 1
        ORDER BY created_at DESC
        """
        return execute_query(query)
    
    @staticmethod
    def save_prediction_task(task_data):
        """保存预测任务"""
        query = """
        INSERT INTO prediction_tasks 
        (task_id, patient_id, task_type, status, input_parameters, priority)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        params = (
            task_data.get('task_id'),
            task_data.get('patient_id'),
            task_data.get('task_type'),
            task_data.get('status', 'pending'),
            json.dumps(task_data.get('input_parameters')),
            task_data.get('priority', 5)
        )
        
        return execute_query(query, params, fetch_all=False)
    
    @staticmethod
    def update_prediction_task(task_id, updates):
        """更新预测任务状态"""
        set_clauses = []
        params = []
        
        for key, value in updates.items():
            if key in ['status', 'start_time', 'end_time', 'error_message', 'result_id', 'retry_count']:
                set_clauses.append(f"{key} = %s")
                params.append(value)
        
        if not set_clauses:
            return 0
        
        query = f"""
        UPDATE prediction_tasks 
        SET {', '.join(set_clauses)}, updated_at = NOW()
        WHERE task_id = %s
        """
        params.append(task_id)
        
        return execute_query(query, params, fetch_all=False)
    
    @staticmethod
    def get_patient_ai_preferences(patient_id):
        """获取患者AI预测偏好"""
        query = """
        SELECT id, patient_id, enable_auto_prediction, prediction_frequency,
               notification_enabled, risk_threshold, preferred_models
        FROM patient_ai_preferences 
        WHERE patient_id = %s
        """
        return execute_query(query, (patient_id,), fetch_one=True)
    
    @staticmethod
    def save_patient_ai_preferences(patient_id, preferences):
        """保存患者AI预测偏好"""
        query = """
        INSERT INTO patient_ai_preferences 
        (patient_id, enable_auto_prediction, prediction_frequency, 
         notification_enabled, risk_threshold, preferred_models)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        enable_auto_prediction = VALUES(enable_auto_prediction),
        prediction_frequency = VALUES(prediction_frequency),
        notification_enabled = VALUES(notification_enabled),
        risk_threshold = VALUES(risk_threshold),
        preferred_models = VALUES(preferred_models),
        updated_at = NOW()
        """
        
        params = (
            patient_id,
            preferences.get('enable_auto_prediction', True),
            preferences.get('prediction_frequency', 7),
            preferences.get('notification_enabled', True),
            preferences.get('risk_threshold', 0.7),
            json.dumps(preferences.get('preferred_models')) if preferences.get('preferred_models') else None
        )
        
        return execute_query(query, params, fetch_all=False)
    
    @staticmethod
    def get_model_cache_config(model_id):
        """获取模型缓存配置"""
        query = """
        SELECT id, model_id, cache_key_prefix, cache_ttl, 
               enable_cache, max_cache_size
        FROM ai_model_cache_config 
        WHERE model_id = %s AND enable_cache = 1
        """
        return execute_query(query, (model_id,), fetch_one=True)
    
    @staticmethod
    def save_model_performance(model_id, performance_data):
        """保存模型性能数据"""
        query = """
        INSERT INTO ai_model_performance 
        (model_id, date, prediction_count, avg_response_time, 
         avg_confidence, error_count, cache_hit_rate)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        prediction_count = prediction_count + VALUES(prediction_count),
        avg_response_time = (avg_response_time + VALUES(avg_response_time)) / 2,
        avg_confidence = (avg_confidence + VALUES(avg_confidence)) / 2,
        error_count = error_count + VALUES(error_count),
        cache_hit_rate = VALUES(cache_hit_rate)
        """
        
        params = (
            model_id,
            performance_data.get('date', date.today()),
            performance_data.get('prediction_count', 1),
            performance_data.get('avg_response_time'),
            performance_data.get('avg_confidence'),
            performance_data.get('error_count', 0),
            performance_data.get('cache_hit_rate')
        )
        
        return execute_query(query, params, fetch_all=False)

# AI数据访问对象类
class AIDataAccess:
    """AI相关的数据访问对象"""
    
    @staticmethod
    def get_patient_for_prediction(patient_id):
        """获取用于AI预测的患者数据"""
        # 获取基本信息
        user_info = DatabaseManager.get_user_by_id(patient_id)
        if not user_info:
            return None
        
        # 获取健康记录
        health_record = DatabaseManager.get_health_record(patient_id)
        
        # 获取最近的健康指标
        recent_metrics = DatabaseManager.get_patient_health_metrics(patient_id, 5)
        
        # 获取AI偏好设置
        ai_preferences = DatabaseManager.get_patient_ai_preferences(patient_id)
        
        return {
            'user_info': user_info,
            'health_record': health_record,
            'recent_metrics': recent_metrics,
            'ai_preferences': ai_preferences
        }
    
    @staticmethod
    def save_prediction_with_analysis(patient_id, prediction_result):
        """保存预测结果并更新健康记录"""
        try:
            # 保存预测结果
            prediction_id = DatabaseManager.save_prediction_result(prediction_result)
            
            # 更新健康记录的AI信息
            if 'ai_score' in prediction_result and 'risk_level' in prediction_result:
                DatabaseManager.update_health_record_ai_info(
                    patient_id, 
                    prediction_result['ai_score'], 
                    prediction_result['risk_level']
                )
            
            # 保存性能数据
            if 'model_id' in prediction_result:
                performance_data = {
                    'prediction_count': 1,
                    'avg_response_time': prediction_result.get('response_time'),
                    'avg_confidence': prediction_result.get('confidence_score'),
                    'error_count': 0
                }
                DatabaseManager.save_model_performance(
                    prediction_result['model_id'], 
                    performance_data
                )
            
            return prediction_id
            
        except Exception as e:
            logging.error(f"保存预测结果失败: {str(e)}")
            raise e
    
    @staticmethod
    def get_prediction_history(patient_id, days=30):
        """获取患者的预测历史"""
        query = """
        SELECT apr.id, apr.prediction_type, apr.prediction_data, 
               apr.confidence_score, apr.model_version, apr.created_at,
               am.model_name
        FROM ai_prediction_results apr
        LEFT JOIN ai_models am ON apr.model_version = am.model_version
        WHERE apr.patient_id = %s 
          AND apr.created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
        ORDER BY apr.created_at DESC
        """
        return execute_query(query, (patient_id, days))
    
    @staticmethod
    def get_model_statistics():
        """获取模型统计信息"""
        query = """
        SELECT 
            am.model_name,
            am.model_version,
            am.is_active,
            COUNT(apr.id) as total_predictions,
            AVG(apr.confidence_score) as avg_confidence,
            MAX(apr.created_at) as last_used
        FROM ai_models am
        LEFT JOIN ai_prediction_results apr ON am.model_version = apr.model_version
        GROUP BY am.id, am.model_name, am.model_version, am.is_active
        ORDER BY total_predictions DESC
        """
        return execute_query(query)
    
    @staticmethod
    def get_pending_prediction_tasks(limit=10):
        """获取待处理的预测任务"""
        query = """
        SELECT pt.id, pt.task_id, pt.patient_id, pt.task_type,
               pt.input_parameters, pt.priority, pt.retry_count,
               u.real_name as patient_name
        FROM prediction_tasks pt
        LEFT JOIN user u ON pt.patient_id = u.user_id
        WHERE pt.status = 'pending'
        ORDER BY pt.priority DESC, pt.created_at ASC
        LIMIT %s
        """
        return execute_query(query, (limit,))

# 数据库工具函数
def format_datetime(dt):
    """格式化日期时间"""
    if isinstance(dt, datetime):
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(dt, date):
        return dt.strftime('%Y-%m-%d')
    return dt

def parse_json_field(json_str):
    """解析JSON字段"""
    if not json_str:
        return None
    try:
        return json.loads(json_str) if isinstance(json_str, str) else json_str
    except (json.JSONDecodeError, TypeError):
        return None

def serialize_for_cache(data):
    """序列化数据用于缓存"""
    def convert_datetime(obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return obj
    
    return json.dumps(data, default=convert_datetime, ensure_ascii=False)
