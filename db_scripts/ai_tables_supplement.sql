-- AI系统补充表结构
-- 为现有SHEC-PSIMS数据库添加AI功能所需的表

-- 患者健康指标扩展表
CREATE TABLE IF NOT EXISTS `patient_health_metrics` (
  `metric_id` int(11) NOT NULL AUTO_INCREMENT,
  `patient_id` int(11) NOT NULL,
  `metric_type` varchar(50) NOT NULL COMMENT '指标类型：血压、血糖、心率等',
  `metric_value` decimal(10,2) DEFAULT NULL COMMENT '指标数值',
  `metric_unit` varchar(20) DEFAULT NULL COMMENT '单位',
  `measurement_method` varchar(50) DEFAULT NULL COMMENT '测量方法',
  `measurement_device` varchar(100) DEFAULT NULL COMMENT '测量设备',
  `quality_score` decimal(3,2) DEFAULT '1.00' COMMENT '数据质量评分0-1',
  `is_abnormal` tinyint(1) DEFAULT '0' COMMENT '是否异常',
  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`metric_id`),
  KEY `idx_patient_type` (`patient_id`, `metric_type`),
  KEY `idx_created_at` (`created_at`),
  CONSTRAINT `fk_patient_health_metrics_patient` FOREIGN KEY (`patient_id`) REFERENCES `user` (`user_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='患者健康指标扩展表';

-- AI模型缓存配置表
CREATE TABLE IF NOT EXISTS `ai_model_cache_config` (
  `config_id` int(11) NOT NULL AUTO_INCREMENT,
  `model_name` varchar(100) NOT NULL,
  `cache_key_pattern` varchar(255) NOT NULL COMMENT '缓存键模式',
  `cache_ttl` int(11) DEFAULT '3600' COMMENT '缓存过期时间（秒）',
  `max_cache_size` int(11) DEFAULT '1000' COMMENT '最大缓存条数',
  `cache_strategy` enum('LRU','FIFO','TTL') DEFAULT 'LRU' COMMENT '缓存策略',
  `is_enabled` tinyint(1) DEFAULT '1',
  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`config_id`),
  UNIQUE KEY `uk_model_name` (`model_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='AI模型缓存配置';

-- AI模型性能监控表
CREATE TABLE IF NOT EXISTS `ai_model_performance` (
  `performance_id` int(11) NOT NULL AUTO_INCREMENT,
  `model_name` varchar(100) NOT NULL,
  `model_version` varchar(50) NOT NULL,
  `evaluation_date` date NOT NULL,
  `accuracy` decimal(5,4) DEFAULT NULL COMMENT '准确率',
  `precision_score` decimal(5,4) DEFAULT NULL COMMENT '精确率',
  `recall` decimal(5,4) DEFAULT NULL COMMENT '召回率',
  `f1_score` decimal(5,4) DEFAULT NULL COMMENT 'F1分数',
  `auc_score` decimal(5,4) DEFAULT NULL COMMENT 'AUC分数',
  `prediction_count` int(11) DEFAULT '0' COMMENT '预测次数',
  `avg_response_time` decimal(8,2) DEFAULT NULL COMMENT '平均响应时间（毫秒）',
  `error_count` int(11) DEFAULT '0' COMMENT '错误次数',
  `notes` text COMMENT '备注',
  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`performance_id`),
  KEY `idx_model_date` (`model_name`, `evaluation_date`),
  KEY `idx_model_version` (`model_name`, `model_version`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='AI模型性能监控';

-- 患者AI偏好设置表
CREATE TABLE IF NOT EXISTS `patient_ai_preferences` (
  `preference_id` int(11) NOT NULL AUTO_INCREMENT,
  `patient_id` int(11) NOT NULL,
  `enable_ai_prediction` tinyint(1) DEFAULT '1' COMMENT '启用AI预测',
  `enable_risk_notification` tinyint(1) DEFAULT '1' COMMENT '启用风险通知',
  `preferred_prediction_frequency` enum('daily','weekly','monthly') DEFAULT 'weekly' COMMENT '预测频率偏好',
  `notification_threshold` enum('low','medium','high') DEFAULT 'medium' COMMENT '通知阈值',
  `data_sharing_consent` tinyint(1) DEFAULT '0' COMMENT '数据共享同意',
  `preferred_language` varchar(10) DEFAULT 'zh-cn' COMMENT '偏好语言',
  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`preference_id`),
  UNIQUE KEY `uk_patient_id` (`patient_id`),
  CONSTRAINT `fk_patient_ai_preferences_patient` FOREIGN KEY (`patient_id`) REFERENCES `user` (`user_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='患者AI偏好设置';

-- 预测分析记录表
CREATE TABLE IF NOT EXISTS `prediction_analysis` (
  `analysis_id` int(11) NOT NULL AUTO_INCREMENT,
  `prediction_id` int(11) NOT NULL COMMENT '关联ai_prediction_results表',
  `analysis_type` varchar(50) NOT NULL COMMENT '分析类型：趋势、异常检测等',
  `analysis_result` json DEFAULT NULL COMMENT '分析结果JSON',
  `confidence_level` decimal(5,4) DEFAULT NULL COMMENT '置信水平',
  `feature_importance` json DEFAULT NULL COMMENT '特征重要性',
  `interpretation` text COMMENT '结果解释',
  `recommendations` json DEFAULT NULL COMMENT '建议JSON',
  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`analysis_id`),
  KEY `idx_prediction_id` (`prediction_id`),
  KEY `idx_analysis_type` (`analysis_type`),
  CONSTRAINT `fk_prediction_analysis_prediction` FOREIGN KEY (`prediction_id`) REFERENCES `ai_prediction_results` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='预测分析记录';

-- 插入默认的AI模型缓存配置
INSERT IGNORE INTO `ai_model_cache_config` (`model_name`, `cache_key_pattern`, `cache_ttl`, `max_cache_size`, `cache_strategy`) VALUES
('HealthRiskAssessment', 'risk_prediction:{patient_id}', 3600, 1000, 'LRU'),
('DiabetesRiskModel', 'diabetes_risk:{patient_id}', 7200, 500, 'TTL'),
('HypertensionRiskModel', 'hypertension_risk:{patient_id}', 7200, 500, 'TTL'),
('CardiovascularRiskModel', 'cardiovascular_risk:{patient_id}', 14400, 300, 'TTL');

-- 创建索引以提高查询性能
CREATE INDEX IF NOT EXISTS idx_health_record_patient_time ON health_record(patient_id, measurement_time);
CREATE INDEX IF NOT EXISTS idx_ai_prediction_patient_type ON ai_prediction_results(patient_id, prediction_type, created_at);
CREATE INDEX IF NOT EXISTS idx_ai_prediction_risk_level ON ai_prediction_results(risk_level, created_at);

-- 为现有表添加一些有用的索引
ALTER TABLE `user` ADD INDEX IF NOT EXISTS idx_user_type_status(user_type, status);
ALTER TABLE `health_record` ADD INDEX IF NOT EXISTS idx_health_record_created(created_at);
ALTER TABLE `ai_prediction_results` ADD INDEX IF NOT EXISTS idx_ai_prediction_expires(expires_at);

-- 数据表创建完成提示
SELECT 'AI系统补充表结构创建完成' AS status;
