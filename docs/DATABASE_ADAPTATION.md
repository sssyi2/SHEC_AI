# SHEC AI 数据库模块适配完成

## ✅ 修改总结

### 1. **完全适配现有schema.sql结构**
- ❌ **移除了冲突的表创建** (users, health_data, predictions)
- ✅ **适配现有表结构** (user, health_record, ai_prediction_results等)
- ✅ **使用正确的字段名** (user_id而不是id, patient_id等)

### 2. **新增的数据库功能**

#### DatabaseManager类方法：
```python
# 用户相关
DatabaseManager.get_user_by_id(user_id)
DatabaseManager.get_user_by_username(username)

# 健康记录相关
DatabaseManager.get_health_record(patient_id)
DatabaseManager.get_patient_health_metrics(patient_id, limit)
DatabaseManager.save_health_metrics(patient_id, metrics_data)

# AI预测相关
DatabaseManager.save_prediction_result(prediction_data)
DatabaseManager.get_prediction_results(patient_id, prediction_type, limit)
DatabaseManager.update_health_record_ai_info(patient_id, ai_score, risk_level)

# AI模型相关
DatabaseManager.get_ai_models()
DatabaseManager.get_model_cache_config(model_id)
DatabaseManager.save_model_performance(model_id, performance_data)

# 任务管理相关
DatabaseManager.save_prediction_task(task_data)
DatabaseManager.update_prediction_task(task_id, updates)

# AI偏好相关
DatabaseManager.get_patient_ai_preferences(patient_id)
DatabaseManager.save_patient_ai_preferences(patient_id, preferences)
```

#### AIDataAccess类方法：
```python
# 综合数据获取
AIDataAccess.get_patient_for_prediction(patient_id)
AIDataAccess.save_prediction_with_analysis(patient_id, prediction_result)

# 历史数据分析
AIDataAccess.get_prediction_history(patient_id, days)
AIDataAccess.get_model_statistics()

# 任务队列管理
AIDataAccess.get_pending_prediction_tasks(limit)

# 批量操作
AIDataAccess.batch_update_health_metrics(metrics_list)
```

### 3. **工具函数**
```python
format_datetime(dt)          # 格式化日期时间
parse_json_field(json_str)   # 解析JSON字段
serialize_for_cache(data)    # 序列化用于缓存
```

## 🔧 使用方法

### 1. 初始化数据库连接
```python
from flask import Flask
from utils.database import init_database

app = Flask(__name__)

# 配置数据库连接参数
app.config['DB_HOST'] = 'localhost'
app.config['DB_PORT'] = 3306
app.config['DB_NAME'] = 'shec_psims'
app.config['DB_USER'] = 'your_username'
app.config['DB_PASSWORD'] = 'your_password'

# 初始化数据库连接池
init_database(app)
```

### 2. 使用数据库操作
```python
from utils.database import DatabaseManager, AIDataAccess

# 获取患者信息
user_info = DatabaseManager.get_user_by_id(1)

# 获取用于AI预测的完整数据
patient_data = AIDataAccess.get_patient_for_prediction(1)

# 保存预测结果
prediction_result = {
    'patient_id': 1,
    'prediction_id': 'pred_20250718_001',
    'prediction_type': 'health_trend',
    'input_data': {...},
    'prediction_data': {...},
    'confidence_score': 0.87,
    'model_version': 'v1.0.0'
}
AIDataAccess.save_prediction_with_analysis(1, prediction_result)
```

## 🗃️ 数据库表映射

### 现有表使用说明：
| 原表名 | 用途 | 主要字段 |
|--------|------|----------|
| `user` | 用户信息 | user_id, user_name, age, gender |
| `health_record` | 健康记录 | patient_id, ai_score, risk_level |
| `ai_prediction_results` | AI预测结果 | patient_id, prediction_type, confidence_score |
| `patient_health_metrics` | 健康指标历史 | patient_id, systolic_pressure, blood_sugar |
| `ai_models` | AI模型配置 | model_name, model_version, is_active |
| `prediction_tasks` | 预测任务队列 | task_id, patient_id, status, priority |

## ⚠️ 重要注意事项

### 1. **字段名适配**
- 使用 `user_id` 而不是 `id`
- 使用 `patient_id` 引用用户
- JSON字段需要序列化/反序列化

### 2. **外键关系**
- `ai_prediction_results.patient_id` → `user.user_id`
- `patient_health_metrics.patient_id` → `user.user_id`
- `ai_model_cache_config.model_id` → `ai_models.id`

### 3. **连接池配置**
- 必须先调用 `init_database(app)` 初始化连接池
- 连接池大小默认为10，可根据需要调整
- 自动处理连接的获取和释放

## 🚀 优势

### 1. **完全兼容现有系统**
- ✅ 不破坏现有数据结构
- ✅ 不影响其他模块
- ✅ 保持向后兼容性

### 2. **专为AI优化**
- ✅ 支持JSON数据存储
- ✅ 预测结果缓存机制
- ✅ 模型性能监控
- ✅ 任务队列管理

### 3. **生产级别功能**
- ✅ 连接池管理
- ✅ 错误处理和日志
- ✅ 事务支持
- ✅ 批量操作优化

## 📝 下一步

1. **配置数据库连接参数**
2. **测试数据库连接**
3. **开始使用AI预测功能**
4. **监控模型性能**

现在您的database.py已经完全适配现有的schema.sql结构，可以安全使用而不会产生任何冲突！
