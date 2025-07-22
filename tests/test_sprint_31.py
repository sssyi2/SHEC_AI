# Sprint 3.1 核心服务开发测试脚本
# 测试预测服务、缓存服务、API接口等核心功能

import sys
import os
import asyncio
import json
import time
import requests
from typing import Dict, Any
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logger import get_logger

logger = get_logger(__name__)

class Sprint31Tester:
    """Sprint 3.1 测试器"""
    
    def __init__(self):
        self.test_results = {
            'prediction_service': {'passed': 0, 'failed': 0, 'errors': []},
            'cache_service': {'passed': 0, 'failed': 0, 'errors': []},
            'api_interfaces': {'passed': 0, 'failed': 0, 'errors': []},
            'validation_modules': {'passed': 0, 'failed': 0, 'errors': []},
            'overall': {'start_time': None, 'end_time': None}
        }
    
    def run_all_tests(self):
        """运行所有测试"""
        print("=" * 60)
        print("🎯 Sprint 3.1: 核心服务开发测试")
        print("=" * 60)
        
        self.test_results['overall']['start_time'] = datetime.now()
        
        try:
            # 测试各个模块
            self.test_prediction_service()
            self.test_cache_service()
            self.test_validation_modules()
            self.test_api_interfaces()
            
        except Exception as e:
            logger.error(f"测试过程中发生异常: {str(e)}")
        
        finally:
            self.test_results['overall']['end_time'] = datetime.now()
            self.print_summary()
    
    def test_prediction_service(self):
        """测试预测服务"""
        print("\n📋 测试 1: 预测服务模块")
        print("-" * 40)
        
        try:
            # 测试模块导入
            try:
                from services.prediction_service import prediction_service, HealthPredictionService
                print("✅ 预测服务模块导入成功")
                self._record_pass('prediction_service')
            except ImportError as e:
                print(f"❌ 预测服务模块导入失败: {e}")
                self._record_fail('prediction_service', str(e))
                return
            
            # 测试服务初始化
            try:
                service = HealthPredictionService()
                print("✅ 预测服务初始化成功")
                self._record_pass('prediction_service')
            except Exception as e:
                print(f"❌ 预测服务初始化失败: {e}")
                self._record_fail('prediction_service', str(e))
                return
            
            # 测试健康指标预测
            try:
                async def test_health_prediction():
                    test_data = {
                        'sequence_data': [[1.0, 2.0, 3.0] for _ in range(7)],
                        'user_context': {'age': 30, 'gender': 'male'}
                    }
                    
                    result = await service.predict_health_indicators(
                        input_data=test_data,
                        user_id=1,
                        model_name='test_model'
                    )
                    
                    required_fields = ['timestamp', 'prediction_type', 'predicted_class']
                    for field in required_fields:
                        if field not in result:
                            raise ValueError(f"预测结果缺少字段: {field}")
                    
                    return True
                
                # 运行异步测试
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                success = loop.run_until_complete(test_health_prediction())
                loop.close()
                
                if success:
                    print("✅ 健康指标预测功能正常")
                    self._record_pass('prediction_service')
                
            except Exception as e:
                print(f"❌ 健康指标预测测试失败: {e}")
                self._record_fail('prediction_service', str(e))
            
            # 测试疾病风险评估
            try:
                async def test_risk_assessment():
                    test_data = {
                        'features': {'age': 45, 'gender': 'female', 'bmi': 28.5},
                        'user_context': {}
                    }
                    
                    result = await service.assess_disease_risk(
                        input_data=test_data,
                        user_id=1,
                        model_name='test_model'
                    )
                    
                    required_fields = ['timestamp', 'prediction_type', 'risk_level']
                    for field in required_fields:
                        if field not in result:
                            raise ValueError(f"风险评估结果缺少字段: {field}")
                    
                    return True
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                success = loop.run_until_complete(test_risk_assessment())
                loop.close()
                
                if success:
                    print("✅ 疾病风险评估功能正常")
                    self._record_pass('prediction_service')
                
            except Exception as e:
                print(f"❌ 疾病风险评估测试失败: {e}")
                self._record_fail('prediction_service', str(e))
            
            # 测试批量预测
            try:
                async def test_batch_prediction():
                    batch_data = [
                        {'sequence_data': [[1.0, 2.0] for _ in range(5)]},
                        {'sequence_data': [[2.0, 3.0] for _ in range(5)]}
                    ]
                    
                    results = await service.batch_predict(
                        batch_data=batch_data,
                        prediction_type='health_indicators'
                    )
                    
                    if len(results) != 2:
                        raise ValueError(f"批量预测结果数量不匹配: 期望2，实际{len(results)}")
                    
                    return True
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                success = loop.run_until_complete(test_batch_prediction())
                loop.close()
                
                if success:
                    print("✅ 批量预测功能正常")
                    self._record_pass('prediction_service')
                
            except Exception as e:
                print(f"❌ 批量预测测试失败: {e}")
                self._record_fail('prediction_service', str(e))
            
            # 测试模型信息获取
            try:
                model_info = service.get_model_info()
                if 'device' in model_info and 'loaded_models' in model_info:
                    print("✅ 模型信息获取功能正常")
                    self._record_pass('prediction_service')
                else:
                    raise ValueError("模型信息格式不正确")
                    
            except Exception as e:
                print(f"❌ 模型信息获取测试失败: {e}")
                self._record_fail('prediction_service', str(e))
            
        except Exception as e:
            print(f"❌ 预测服务测试过程异常: {e}")
            self._record_fail('prediction_service', str(e))
    
    def test_cache_service(self):
        """测试缓存服务"""
        print("\n📋 测试 2: 缓存服务模块")
        print("-" * 40)
        
        try:
            # 测试模块导入
            try:
                from services.cache_service import cache_manager, CacheManager, CacheType
                print("✅ 缓存服务模块导入成功")
                self._record_pass('cache_service')
            except ImportError as e:
                print(f"❌ 缓存服务模块导入失败: {e}")
                self._record_fail('cache_service', str(e))
                return
            
            # 测试缓存管理器初始化
            try:
                manager = CacheManager()
                print("✅ 缓存管理器初始化成功")
                self._record_pass('cache_service')
            except Exception as e:
                print(f"❌ 缓存管理器初始化失败: {e}")
                self._record_fail('cache_service', str(e))
                return
            
            # 测试基本缓存操作
            try:
                # 设置缓存
                test_key = "test_key"
                test_data = {"message": "Hello Cache", "timestamp": datetime.now().isoformat()}
                
                success = manager.set(CacheType.TEMPORARY, test_key, test_data, ttl=60)
                if not success:
                    raise Exception("缓存设置失败")
                
                # 获取缓存
                cached_data = manager.get(CacheType.TEMPORARY, test_key)
                if cached_data != test_data:
                    raise Exception("缓存数据不匹配")
                
                # 检查存在性
                exists = manager.exists(CacheType.TEMPORARY, test_key)
                if not exists:
                    raise Exception("缓存存在性检查失败")
                
                # 删除缓存
                deleted = manager.delete(CacheType.TEMPORARY, test_key)
                if not deleted:
                    raise Exception("缓存删除失败")
                
                print("✅ 基本缓存操作功能正常")
                self._record_pass('cache_service')
                
            except Exception as e:
                print(f"❌ 基本缓存操作测试失败: {e}")
                self._record_fail('cache_service', str(e))
            
            # 测试缓存统计信息
            try:
                stats = manager.get_stats()
                required_stats = ['total_requests', 'hit_rate_percent']
                for stat in required_stats:
                    if stat not in stats:
                        raise ValueError(f"缓存统计信息缺少字段: {stat}")
                
                print("✅ 缓存统计信息功能正常")
                self._record_pass('cache_service')
                
            except Exception as e:
                print(f"❌ 缓存统计信息测试失败: {e}")
                self._record_fail('cache_service', str(e))
            
            # 测试缓存清理
            try:
                # 设置一些测试数据
                for i in range(3):
                    manager.set(CacheType.TEMPORARY, f"test_{i}", f"data_{i}")
                
                # 按类型清理
                deleted_count = manager.clear_by_type(CacheType.TEMPORARY)
                
                print(f"✅ 缓存清理功能正常，清理了 {deleted_count} 个项目")
                self._record_pass('cache_service')
                
            except Exception as e:
                print(f"❌ 缓存清理测试失败: {e}")
                self._record_fail('cache_service', str(e))
            
        except Exception as e:
            print(f"❌ 缓存服务测试过程异常: {e}")
            self._record_fail('cache_service', str(e))
    
    def test_validation_modules(self):
        """测试验证模块"""
        print("\n📋 测试 3: 数据验证模块")
        print("-" * 40)
        
        try:
            # 测试模块导入
            try:
                from utils.validators import (
                    validate_prediction_input,
                    validate_risk_assessment_input,
                    validate_batch_prediction_input
                )
                print("✅ 验证模块导入成功")
                self._record_pass('validation_modules')
            except ImportError as e:
                print(f"❌ 验证模块导入失败: {e}")
                self._record_fail('validation_modules', str(e))
                return
            
            # 测试预测输入验证
            try:
                # 有效数据
                valid_data = {
                    'sequence_data': [[1.0, 2.0, 3.0] for _ in range(7)],
                    'user_context': {'age': 30}
                }
                result = validate_prediction_input(valid_data)
                if not result['valid']:
                    raise ValueError("有效数据验证失败")
                
                # 无效数据
                invalid_data = {}
                result = validate_prediction_input(invalid_data)
                if result['valid']:
                    raise ValueError("无效数据未被正确拒绝")
                
                print("✅ 预测输入验证功能正常")
                self._record_pass('validation_modules')
                
            except Exception as e:
                print(f"❌ 预测输入验证测试失败: {e}")
                self._record_fail('validation_modules', str(e))
            
            # 测试风险评估输入验证
            try:
                # 有效数据
                valid_data = {
                    'features': {'age': 45, 'gender': 'male', 'bmi': 25.0},
                    'user_context': {}
                }
                result = validate_risk_assessment_input(valid_data)
                if not result['valid']:
                    raise ValueError("有效数据验证失败")
                
                # 无效数据（缺少必要字段）
                invalid_data = {
                    'features': {'bmi': 25.0}  # 缺少age和gender
                }
                result = validate_risk_assessment_input(invalid_data)
                if result['valid']:
                    raise ValueError("无效数据未被正确拒绝")
                
                print("✅ 风险评估输入验证功能正常")
                self._record_pass('validation_modules')
                
            except Exception as e:
                print(f"❌ 风险评估输入验证测试失败: {e}")
                self._record_fail('validation_modules', str(e))
            
            # 测试批量预测输入验证
            try:
                # 有效数据
                valid_data = {
                    'batch_data': [
                        {'features': {'age': 30, 'gender': 'male'}},
                        {'features': {'age': 25, 'gender': 'female'}}
                    ],
                    'prediction_type': 'disease_risk'
                }
                result = validate_batch_prediction_input(valid_data)
                if not result['valid']:
                    raise ValueError("有效数据验证失败")
                
                # 无效数据（批量数据为空）
                invalid_data = {
                    'batch_data': [],
                    'prediction_type': 'disease_risk'
                }
                result = validate_batch_prediction_input(invalid_data)
                if result['valid']:
                    raise ValueError("无效数据未被正确拒绝")
                
                print("✅ 批量预测输入验证功能正常")
                self._record_pass('validation_modules')
                
            except Exception as e:
                print(f"❌ 批量预测输入验证测试失败: {e}")
                self._record_fail('validation_modules', str(e))
                
        except Exception as e:
            print(f"❌ 验证模块测试过程异常: {e}")
            self._record_fail('validation_modules', str(e))
    
    def test_api_interfaces(self):
        """测试API接口"""
        print("\n📋 测试 4: API接口模块")
        print("-" * 40)
        
        try:
            # 测试API模块导入
            try:
                from api.predict import predict_bp
                print("✅ API模块导入成功")
                self._record_pass('api_interfaces')
            except ImportError as e:
                print(f"❌ API模块导入失败: {e}")
                self._record_fail('api_interfaces', str(e))
                return
            
            # 测试蓝图注册
            try:
                if predict_bp.name == 'predict':
                    print("✅ API蓝图注册正常")
                    self._record_pass('api_interfaces')
                else:
                    raise ValueError("蓝图名称不正确")
            except Exception as e:
                print(f"❌ API蓝图测试失败: {e}")
                self._record_fail('api_interfaces', str(e))
            
            # 测试响应格式函数
            try:
                from api.predict import api_response
                
                # 测试成功响应
                response, status_code = api_response(data={'test': True}, message="测试成功")
                if status_code != 200:
                    raise ValueError("成功响应状态码不正确")
                
                response_data = response.get_json()
                required_fields = ['timestamp', 'status', 'message', 'data']
                for field in required_fields:
                    if field not in response_data:
                        raise ValueError(f"响应缺少字段: {field}")
                
                # 测试错误响应
                response, status_code = api_response(error="测试错误", status_code=400)
                if status_code != 400:
                    raise ValueError("错误响应状态码不正确")
                
                print("✅ API响应格式功能正常")
                self._record_pass('api_interfaces')
                
            except Exception as e:
                print(f"❌ API响应格式测试失败: {e}")
                self._record_fail('api_interfaces', str(e))
            
            # 测试数据验证函数
            try:
                from api.predict import validate_health_data
                
                # 有效数据
                valid_data = {'age': 30, 'gender': 'male'}
                is_valid, error = validate_health_data(valid_data)
                if not is_valid:
                    raise ValueError("有效数据验证失败")
                
                # 无效数据
                invalid_data = {'age': 'invalid'}
                is_valid, error = validate_health_data(invalid_data)
                if is_valid:
                    raise ValueError("无效数据未被正确拒绝")
                
                print("✅ API数据验证功能正常")
                self._record_pass('api_interfaces')
                
            except Exception as e:
                print(f"❌ API数据验证测试失败: {e}")
                self._record_fail('api_interfaces', str(e))
            
            # 测试模拟预测结果函数
            try:
                from api.predict import mock_prediction_result
                
                # 健康预测
                health_result = mock_prediction_result('health')
                required_fields = ['timestamp', 'prediction_type', 'predicted_class']
                for field in required_fields:
                    if field not in health_result:
                        raise ValueError(f"健康预测结果缺少字段: {field}")
                
                # 风险评估
                risk_result = mock_prediction_result('risk')
                required_fields = ['timestamp', 'prediction_type', 'disease_risks']
                for field in required_fields:
                    if field not in risk_result:
                        raise ValueError(f"风险评估结果缺少字段: {field}")
                
                print("✅ 模拟预测结果功能正常")
                self._record_pass('api_interfaces')
                
            except Exception as e:
                print(f"❌ 模拟预测结果测试失败: {e}")
                self._record_fail('api_interfaces', str(e))
                
        except Exception as e:
            print(f"❌ API接口测试过程异常: {e}")
            self._record_fail('api_interfaces', str(e))
    
    def _record_pass(self, category: str):
        """记录测试通过"""
        self.test_results[category]['passed'] += 1
    
    def _record_fail(self, category: str, error: str):
        """记录测试失败"""
        self.test_results[category]['failed'] += 1
        self.test_results[category]['errors'].append(error)
    
    def print_summary(self):
        """打印测试摘要"""
        print("\n" + "=" * 60)
        print("📊 Sprint 3.1 测试摘要")
        print("=" * 60)
        
        total_passed = 0
        total_failed = 0
        
        for category, results in self.test_results.items():
            if category == 'overall':
                continue
                
            passed = results['passed']
            failed = results['failed']
            total = passed + failed
            
            total_passed += passed
            total_failed += failed
            
            if total > 0:
                pass_rate = (passed / total) * 100
                status = "✅" if failed == 0 else "⚠️" if pass_rate >= 70 else "❌"
                
                print(f"\n{status} {category.replace('_', ' ').title()}")
                print(f"   通过: {passed}/{total} ({pass_rate:.1f}%)")
                
                if failed > 0:
                    print(f"   失败: {failed}")
                    for error in results['errors'][:3]:  # 只显示前3个错误
                        print(f"   - {error}")
                    if len(results['errors']) > 3:
                        print(f"   - ... 还有 {len(results['errors']) - 3} 个错误")
        
        # 总体统计
        total_tests = total_passed + total_failed
        if total_tests > 0:
            overall_pass_rate = (total_passed / total_tests) * 100
            overall_status = "✅" if total_failed == 0 else "⚠️" if overall_pass_rate >= 70 else "❌"
            
            print(f"\n{overall_status} 总体结果")
            print(f"   总测试数: {total_tests}")
            print(f"   通过: {total_passed} ({overall_pass_rate:.1f}%)")
            print(f"   失败: {total_failed}")
            
            # 计算测试时间
            if self.test_results['overall']['start_time'] and self.test_results['overall']['end_time']:
                duration = self.test_results['overall']['end_time'] - self.test_results['overall']['start_time']
                print(f"   测试耗时: {duration.total_seconds():.2f}秒")
        
        # Sprint 3.1 完成状态
        print(f"\n🎯 Sprint 3.1 核心服务开发状态:")
        
        if overall_pass_rate >= 90:
            print("   🎉 优秀 - 所有核心功能已完成并测试通过")
            sprint_status = "完成"
        elif overall_pass_rate >= 70:
            print("   ✅ 良好 - 主要功能已完成，部分功能需要优化")
            sprint_status = "基本完成"
        else:
            print("   ⚠️  需要改进 - 还有重要功能需要修复")
            sprint_status = "未完成"
        
        print(f"\n📋 Sprint 3.1 交付物清单:")
        deliverables = [
            ("预测服务模块", "✅" if self.test_results['prediction_service']['failed'] == 0 else "⚠️"),
            ("缓存策略实现", "✅" if self.test_results['cache_service']['failed'] == 0 else "⚠️"),
            ("RESTful API接口", "✅" if self.test_results['api_interfaces']['failed'] == 0 else "⚠️"),
            ("数据验证模块", "✅" if self.test_results['validation_modules']['failed'] == 0 else "⚠️"),
            ("缓存优化机制", "✅" if self.test_results['cache_service']['failed'] == 0 else "⚠️")
        ]
        
        for deliverable, status in deliverables:
            print(f"   {status} {deliverable}")
        
        print(f"\n🚀 当前 Sprint 状态: {sprint_status}")
        
        if sprint_status == "完成":
            print("   可以开始 Sprint 3.2 或继续完善其他功能")
        elif sprint_status == "基本完成":
            print("   建议修复主要问题后再进入下一阶段")
        else:
            print("   需要重点关注失败的测试项目")

def main():
    """主函数"""
    print("🔧 开始 Sprint 3.1 核心服务开发测试...")
    
    tester = Sprint31Tester()
    tester.run_all_tests()
    
    print("\n" + "=" * 60)
    print("测试完成！查看上方摘要了解详细结果。")
    print("=" * 60)

if __name__ == "__main__":
    main()
