#!/usr/bin/env python3
"""
配置验证工具
用于验证不同环境下的配置是否正确
"""

import os
import sys
import json
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

def test_config_import():
    """测试配置导入是否正常"""
    print("🔍 测试配置导入...")
    try:
        from config.settings import get_config, config
        print("✅ 配置模块导入成功")
        return True
    except ImportError as e:
        print(f"❌ 配置模块导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 配置模块导入异常: {e}")
        return False

def test_pytorch_import():
    """测试PyTorch导入"""
    print("\n🔍 测试PyTorch导入...")
    try:
        from config.settings import TORCH_AVAILABLE
        if TORCH_AVAILABLE:
            import torch
            print(f"✅ PyTorch可用: {torch.__version__}")
            print(f"   设备支持: {'CUDA' if torch.cuda.is_available() else 'CPU only'}")
        else:
            print("⚠️  PyTorch不可用，使用CPU模式")
        return True
    except Exception as e:
        print(f"❌ PyTorch测试失败: {e}")
        return False

def test_environment_configs():
    """测试不同环境配置"""
    print("\n🔍 测试不同环境配置...")
    
    try:
        from config.settings import config
        
        environments = ['development', 'testing', 'production', 'docker']
        
        for env_name in environments:
            print(f"\n📋 {env_name.upper()} 环境:")
            
            if env_name in config:
                cfg = config[env_name]
                
                # 创建配置实例
                cfg_instance = cfg()
                
                # 显示关键配置
                print(f"   数据库主机: {cfg_instance.DB_HOST}")
                print(f"   数据库端口: {cfg_instance.DB_PORT}")
                print(f"   数据库名称: {cfg_instance.DB_NAME}")
                print(f"   Redis主机: {cfg_instance.REDIS_HOST}")
                print(f"   Redis端口: {cfg_instance.REDIS_PORT}")
                print(f"   调试模式: {cfg_instance.DEBUG}")
                print(f"   设备类型: {cfg_instance.DEVICE}")
                
            else:
                print(f"   ❌ 环境配置不存在")
        
        return True
        
    except Exception as e:
        print(f"❌ 环境配置测试失败: {e}")
        return False

def test_docker_environment():
    """测试Docker环境检测"""
    print("\n🔍 测试Docker环境检测...")
    
    # 保存原始环境变量
    original_env = {}
    test_vars = ['RUNNING_IN_DOCKER', 'FLASK_ENV']
    
    for var in test_vars:
        if var in os.environ:
            original_env[var] = os.environ[var]
    
    try:
        from config.settings import get_config
        
        # 测试非Docker环境
        for var in test_vars:
            if var in os.environ:
                del os.environ[var]
        
        cfg1 = get_config()
        print(f"   默认环境: {cfg1.__class__.__name__}")
        
        # 测试Docker环境
        os.environ['RUNNING_IN_DOCKER'] = 'true'
        os.environ['FLASK_ENV'] = 'docker'
        
        # 重新导入模块以应用环境变量
        import importlib
        import config.settings
        importlib.reload(config.settings)
        
        cfg2 = config.settings.get_config()
        print(f"   Docker环境: {cfg2.__class__.__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Docker环境检测失败: {e}")
        return False
        
    finally:
        # 恢复原始环境变量
        for var in test_vars:
            if var in os.environ:
                del os.environ[var]
        
        for var, value in original_env.items():
            os.environ[var] = value

def test_database_config():
    """测试数据库配置"""
    print("\n🔍 测试数据库配置...")
    
    try:
        from config.settings import get_config
        
        # 测试各种数据库配置
        configs_to_test = [
            {'FLASK_ENV': 'development'},
            {'FLASK_ENV': 'docker', 'RUNNING_IN_DOCKER': 'true'},
            {'FLASK_ENV': 'production'}
        ]
        
        for env_vars in configs_to_test:
            # 设置环境变量
            original_vars = {}
            for key, value in env_vars.items():
                if key in os.environ:
                    original_vars[key] = os.environ[key]
                os.environ[key] = value
            
            try:
                # 重新导入配置
                import importlib
                import config.settings
                importlib.reload(config.settings)
                
                cfg = config.settings.get_config()
                cfg_instance = cfg()
                
                print(f"\n   环境变量 {env_vars}:")
                print(f"   配置类: {cfg.__class__.__name__}")
                print(f"   DB连接: {cfg_instance.DB_USER}@{cfg_instance.DB_HOST}:{cfg_instance.DB_PORT}/{cfg_instance.DB_NAME}")
                print(f"   Redis: {cfg_instance.REDIS_HOST}:{cfg_instance.REDIS_PORT}")
                
            finally:
                # 恢复环境变量
                for key in env_vars:
                    if key in original_vars:
                        os.environ[key] = original_vars[key]
                    elif key in os.environ:
                        del os.environ[key]
        
        return True
        
    except Exception as e:
        print(f"❌ 数据库配置测试失败: {e}")
        return False

def generate_config_summary():
    """生成配置摘要"""
    print("\n📋 生成配置摘要...")
    
    try:
        from config.settings import config, get_config, TORCH_AVAILABLE
        
        summary = {
            "torch_available": TORCH_AVAILABLE,
            "available_environments": list(config.keys()),
            "current_config": None,
            "database_configs": {},
            "redis_configs": {}
        }
        
        # 获取当前配置
        current_cfg = get_config()
        current_cfg_instance = current_cfg()
        summary["current_config"] = {
            "class": current_cfg.__name__,
            "debug": current_cfg_instance.DEBUG,
            "device": current_cfg_instance.DEVICE
        }
        
        # 收集各环境的数据库和Redis配置
        for env_name, cfg_class in config.items():
            if env_name != 'default':
                cfg_instance = cfg_class()
                summary["database_configs"][env_name] = {
                    "host": cfg_instance.DB_HOST,
                    "port": cfg_instance.DB_PORT,
                    "database": cfg_instance.DB_NAME,
                    "user": cfg_instance.DB_USER
                }
                summary["redis_configs"][env_name] = {
                    "host": cfg_instance.REDIS_HOST,
                    "port": cfg_instance.REDIS_PORT,
                    "db": cfg_instance.REDIS_DB
                }
        
        # 保存摘要
        summary_file = "config_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"   配置摘要已保存到: {summary_file}")
        print(f"   当前配置类: {summary['current_config']['class']}")
        print(f"   PyTorch支持: {summary['torch_available']}")
        print(f"   可用环境: {', '.join(summary['available_environments'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置摘要生成失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 SHEC AI 配置验证工具")
    print("=" * 50)
    
    tests = [
        test_config_import,
        test_pytorch_import,
        test_environment_configs,
        test_docker_environment,
        test_database_config,
        generate_config_summary
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ 测试 {test_func.__name__} 异常: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 测试结果摘要:")
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_func, result) in enumerate(zip(tests, results)):
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {i+1}. {test_func.__name__}: {status}")
    
    print(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有配置测试通过！")
        return 0
    else:
        print("⚠️  部分测试失败，请检查配置")
        return 1

if __name__ == "__main__":
    exit(main())
