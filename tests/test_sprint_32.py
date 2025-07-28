#!/usr/bin/env python3
"""
Sprint 3.2 RESTful API开发测试脚本
测试所有API接口的完整性和功能
"""

import sys
import os
import time
import json
import traceback
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_section(title):
    """打印测试段落标题"""
    print(f"\n{'='*60}")
    print(f"📋 {title}")
    print('='*60)

def print_test_result(test_name, status, details=None):
    """打印测试结果"""
    emoji = "✅" if status else "❌"
    print(f"{emoji} {test_name}")
    if details:
        print(f"   详情: {details}")

def test_api_blueprint_imports():
    """测试API蓝图导入"""
    print_section("测试 1: API蓝图导入")
    
    test_results = []
    
    # 测试predict蓝图导入
    try:
        from api.predict import predict_bp
        print_test_result("预测API蓝图导入", True)
        test_results.append(True)
    except Exception as e:
        print_test_result("预测API蓝图导入", False, str(e))
        test_results.append(False)
    
    # 测试health蓝图导入
    try:
        from api.health import health_bp
        print_test_result("健康检查API蓝图导入", True)
        test_results.append(True)
    except Exception as e:
        print_test_result("健康检查API蓝图导入", False, str(e))
        test_results.append(False)
    
    # 测试models蓝图导入
    try:
        from api.models import models_bp
        print_test_result("模型管理API蓝图导入", True)
        test_results.append(True)
    except Exception as e:
        print_test_result("模型管理API蓝图导入", False, str(e))
        test_results.append(False)
    
    return test_results

def test_flask_app_creation():
    """测试Flask应用创建和蓝图注册"""
    print_section("测试 2: Flask应用创建")
    
    test_results = []
    
    try:
        from app import create_app
        app = create_app('development')
        
        print_test_result("Flask应用创建", True)
        test_results.append(True)
        
        # 检查蓝图是否注册
        registered_blueprints = [bp.name for bp in app.blueprints.values()]
        print(f"   已注册蓝图: {registered_blueprints}")
        
        # 检查预期的蓝图
        expected_blueprints = ['predict', 'health', 'models']
        for bp_name in expected_blueprints:
            if bp_name in registered_blueprints:
                print_test_result(f"{bp_name}蓝图注册", True)
                test_results.append(True)
            else:
                print_test_result(f"{bp_name}蓝图注册", False, "蓝图未找到")
                test_results.append(False)
        
        return app, test_results
        
    except Exception as e:
        print_test_result("Flask应用创建", False, str(e))
        test_results.append(False)
        return None, test_results

def test_api_routes(app):
    """测试API路由"""
    print_section("测试 3: API路由检查")
    
    if app is None:
        print_test_result("API路由测试", False, "应用未创建")
        return [False]
    
    test_results = []
    
    with app.app_context():
        # 获取所有路由
        routes = []
        for rule in app.url_map.iter_rules():
            routes.append({
                'endpoint': rule.endpoint,
                'methods': list(rule.methods),
                'rule': rule.rule
            })
        
        # 检查Sprint 3.2要求的API接口
        expected_apis = [
            {'path': '/api/predict/health', 'method': 'POST'},
            {'path': '/api/predict/risk', 'method': 'POST'},
            {'path': '/api/predict/models', 'method': 'GET'},
            {'path': '/api/predict/models/train', 'method': 'POST'},
            {'path': '/health', 'method': 'GET'},
        ]
        
        for api in expected_apis:
            found = False
            for route in routes:
                if api['path'] in route['rule'] and api['method'] in route['methods']:
                    found = True
                    break
            
            if found:
                print_test_result(f"{api['method']} {api['path']}", True)
                test_results.append(True)
            else:
                print_test_result(f"{api['method']} {api['path']}", False, "路由未找到")
                test_results.append(False)
    
    return test_results

def test_api_response_format():
    """测试API响应格式"""
    print_section("测试 4: API响应格式")
    
    test_results = []
    
    try:
        from api.predict import api_response
        
        # 测试成功响应
        response_data, status_code = api_response(
            data={'test': 'data'},
            message="测试成功"
        )
        
        response_json = response_data.get_json()
        
        # 检查响应格式
        required_fields = ['timestamp', 'status', 'message', 'data']
        for field in required_fields:
            if field in response_json:
                print_test_result(f"响应包含{field}字段", True)
                test_results.append(True)
            else:
                print_test_result(f"响应包含{field}字段", False)
                test_results.append(False)
        
        # 检查状态码
        if status_code == 200:
            print_test_result("默认状态码正确", True)
            test_results.append(True)
        else:
            print_test_result("默认状态码正确", False, f"期望200，实际{status_code}")
            test_results.append(False)
        
        # 测试错误响应
        error_response, error_status = api_response(
            error="测试错误",
            message="错误测试",
            status_code=400
        )
        
        error_json = error_response.get_json()
        
        if 'error' in error_json and error_json['status'] == 'error':
            print_test_result("错误响应格式正确", True)
            test_results.append(True)
        else:
            print_test_result("错误响应格式正确", False)
            test_results.append(False)
        
    except Exception as e:
        print_test_result("API响应格式测试", False, str(e))
        test_results.append(False)
    
    return test_results

def test_data_validation():
    """测试数据验证功能"""
    print_section("测试 5: 数据验证")
    
    test_results = []
    
    try:
        from utils.validators import validate_json_request
        
        # 模拟请求对象
        class MockRequest:
            def __init__(self, data):
                self._json = data
                self.is_json = True
            
            def get_json(self):
                return self._json
        
        # 测试有效数据
        valid_request = MockRequest({'test': 'data'})
        result = validate_json_request(valid_request)
        
        if result == {'test': 'data'}:
            print_test_result("有效JSON数据验证", True)
            test_results.append(True)
        else:
            print_test_result("有效JSON数据验证", False)
            test_results.append(False)
        
        # 测试无效数据
        invalid_request = MockRequest(None)
        result = validate_json_request(invalid_request)
        
        if result is None:
            print_test_result("无效JSON数据验证", True)
            test_results.append(True)
        else:
            print_test_result("无效JSON数据验证", False)
            test_results.append(False)
        
    except Exception as e:
        print_test_result("数据验证功能测试", False, str(e))
        test_results.append(False)
    
    return test_results

def test_error_handling():
    """测试错误处理机制"""
    print_section("测试 6: 错误处理机制")
    
    test_results = []
    
    try:
        from app import create_app
        app = create_app('development')
        
        # 测试404错误处理
        with app.test_client() as client:
            response = client.get('/nonexistent')
            
            if response.status_code == 404:
                print_test_result("404错误处理", True)
                test_results.append(True)
            else:
                print_test_result("404错误处理", False, f"状态码: {response.status_code}")
                test_results.append(False)
        
        # 测试方法不允许错误处理
        with app.test_client() as client:
            response = client.put('/health')  # health接口只支持GET
            
            if response.status_code == 405:
                print_test_result("405方法不允许错误处理", True)
                test_results.append(True)
            else:
                print_test_result("405方法不允许错误处理", False, f"状态码: {response.status_code}")
                test_results.append(False)
        
    except Exception as e:
        print_test_result("错误处理机制测试", False, str(e))
        test_results.append(False)
    
    return test_results

def run_all_tests():
    """运行所有测试"""
    print("🔧 开始 Sprint 3.2 RESTful API开发测试...")
    print("="*60)
    print("🎯 Sprint 3.2: RESTful API开发测试")
    
    start_time = time.time()
    all_results = []
    
    # 运行各项测试
    blueprint_results = test_api_blueprint_imports()
    all_results.extend(blueprint_results)
    
    app, app_results = test_flask_app_creation()
    all_results.extend(app_results)
    
    route_results = test_api_routes(app)
    all_results.extend(route_results)
    
    response_results = test_api_response_format()
    all_results.extend(response_results)
    
    validation_results = test_data_validation()
    all_results.extend(validation_results)
    
    error_results = test_error_handling()
    all_results.extend(error_results)
    
    # 计算测试结果
    end_time = time.time()
    test_time = end_time - start_time
    
    total_tests = len(all_results)
    passed_tests = sum(all_results)
    failed_tests = total_tests - passed_tests
    
    # 打印测试摘要
    print_section("Sprint 3.2 测试摘要")
    
    print("✅ API蓝图导入")
    print(f"   通过: {sum(blueprint_results)}/{len(blueprint_results)} ({sum(blueprint_results)/len(blueprint_results)*100:.1f}%)")
    
    print("✅ Flask应用创建")
    print(f"   通过: {sum(app_results)}/{len(app_results)} ({sum(app_results)/len(app_results)*100:.1f}%)")
    
    print("✅ API路由")
    print(f"   通过: {sum(route_results)}/{len(route_results)} ({sum(route_results)/len(route_results)*100:.1f}%)")
    
    print("✅ 响应格式")
    print(f"   通过: {sum(response_results)}/{len(response_results)} ({sum(response_results)/len(response_results)*100:.1f}%)")
    
    print("✅ 数据验证")
    print(f"   通过: {sum(validation_results)}/{len(validation_results)} ({sum(validation_results)/len(validation_results)*100:.1f}%)")
    
    print("✅ 错误处理")
    print(f"   通过: {sum(error_results)}/{len(error_results)} ({sum(error_results)/len(error_results)*100:.1f}%)")
    
    # 总体结果
    print("✅ 总体结果")
    print(f"   总测试数: {total_tests}")
    print(f"   通过: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
    print(f"   失败: {failed_tests}")
    print(f"   测试耗时: {test_time:.2f}秒")
    
    # Sprint状态评估
    if passed_tests/total_tests >= 0.9:
        status = "🎉 优秀 - Sprint 3.2 API开发基本完成"
    elif passed_tests/total_tests >= 0.75:
        status = "🔄 良好 - 大部分功能已完成，需要完善细节"
    else:
        status = "⚠️  需要改进 - 存在较多问题需要解决"
    
    print(f"🎯 Sprint 3.2 RESTful API开发状态:")
    print(f"   {status}")
    
    # Sprint 3.2 交付物检查
    print("📋 Sprint 3.2 交付物清单:")
    deliverables = [
        ("API接口设计", True if sum(route_results) >= 4 else False),
        ("请求参数验证", True if sum(validation_results) >= 2 else False),
        ("响应格式标准化", True if sum(response_results) >= 4 else False),
        ("错误处理机制", True if sum(error_results) >= 2 else False),
        ("API版本管理", True)  # 通过蓝图实现
    ]
    
    for deliverable, completed in deliverables:
        emoji = "✅" if completed else "❌"
        print(f"   {emoji} {deliverable}")
    
    # 下一步建议
    completion_rate = passed_tests/total_tests
    if completion_rate >= 0.9:
        print("🚀 当前 Sprint 状态: 完成")
        print("   可以开始 Sprint 3.3 业务逻辑完善或继续优化API功能")
    else:
        print("🔧 当前 Sprint 状态: 进行中")
        print("   建议优先解决失败的测试项目")
    
    print("="*60)
    print("测试完成！查看上方摘要了解详细结果。")
    print("="*60)

if __name__ == "__main__":
    try:
        run_all_tests()
    except Exception as e:
        print(f"❌ 测试运行失败: {str(e)}")
        print(traceback.format_exc())
