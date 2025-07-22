# 认证和授权模块（简化版本）
# 提供基本的认证和授权功能，支持JWT token机制

from functools import wraps
from flask import request, jsonify, g, current_app
from typing import Dict, Any, Optional, List
import jwt
from datetime import datetime, timedelta
import hashlib
import uuid

from utils.logger import get_logger

logger = get_logger(__name__)

class AuthConfig:
    """认证配置"""
    SECRET_KEY = "shec_ai_secret_key_change_in_production"
    TOKEN_EXPIRY_HOURS = 24
    REFRESH_TOKEN_EXPIRY_DAYS = 7
    ALGORITHM = "HS256"

class AuthService:
    """认证服务类"""
    
    @staticmethod
    def generate_token(user_data: Dict[str, Any]) -> str:
        """生成JWT token"""
        try:
            payload = {
                'user_id': user_data.get('user_id'),
                'username': user_data.get('username'),
                'role': user_data.get('role', 'user'),
                'exp': datetime.utcnow() + timedelta(hours=AuthConfig.TOKEN_EXPIRY_HOURS),
                'iat': datetime.utcnow(),
                'jti': str(uuid.uuid4())  # JWT ID
            }
            
            token = jwt.encode(payload, AuthConfig.SECRET_KEY, algorithm=AuthConfig.ALGORITHM)
            return token
            
        except Exception as e:
            logger.error(f"生成token失败: {str(e)}")
            raise
    
    @staticmethod
    def verify_token(token: str) -> Optional[Dict[str, Any]]:
        """验证JWT token"""
        try:
            payload = jwt.decode(token, AuthConfig.SECRET_KEY, algorithms=[AuthConfig.ALGORITHM])
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token已过期")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"无效的token: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"验证token失败: {str(e)}")
            return None
    
    @staticmethod
    def hash_password(password: str) -> str:
        """密码哈希"""
        # 简单的密码哈希（生产环境应使用bcrypt）
        salt = "shec_ai_salt"
        return hashlib.sha256((password + salt).encode()).hexdigest()
    
    @staticmethod
    def verify_password(password: str, hashed_password: str) -> bool:
        """验证密码"""
        return AuthService.hash_password(password) == hashed_password

# 模拟用户数据（生产环境应连接数据库）
MOCK_USERS = {
    'admin': {
        'user_id': 1,
        'username': 'admin',
        'password_hash': AuthService.hash_password('admin123'),
        'role': 'admin',
        'email': 'admin@shec.com',
        'created_at': datetime.utcnow()
    },
    'doctor': {
        'user_id': 2,
        'username': 'doctor',
        'password_hash': AuthService.hash_password('doctor123'),
        'role': 'doctor',
        'email': 'doctor@shec.com',
        'created_at': datetime.utcnow()
    },
    'user': {
        'user_id': 3,
        'username': 'user',
        'password_hash': AuthService.hash_password('user123'),
        'role': 'user',
        'email': 'user@shec.com',
        'created_at': datetime.utcnow()
    }
}

def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """认证用户"""
    try:
        user = MOCK_USERS.get(username)
        if not user:
            return None
        
        if AuthService.verify_password(password, user['password_hash']):
            # 返回用户信息（不包含密码）
            user_data = user.copy()
            del user_data['password_hash']
            return user_data
        
        return None
        
    except Exception as e:
        logger.error(f"用户认证失败: {str(e)}")
        return None

def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """根据ID获取用户信息"""
    try:
        for user in MOCK_USERS.values():
            if user['user_id'] == user_id:
                user_data = user.copy()
                del user_data['password_hash']
                return user_data
        return None
        
    except Exception as e:
        logger.error(f"获取用户信息失败: {str(e)}")
        return None

def require_auth(f):
    """认证装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None
        
        # 从Header获取token
        auth_header = request.headers.get('Authorization')
        if auth_header:
            try:
                token = auth_header.split(' ')[1]  # Bearer <token>
            except IndexError:
                return jsonify({
                    'error': 'Invalid authorization header format',
                    'message': '授权头格式错误'
                }), 401
        
        # 从参数获取token（备用方案）
        if not token:
            token = request.args.get('token')
        
        if not token:
            return jsonify({
                'error': 'Token is missing',
                'message': '缺少认证token'
            }), 401
        
        # 验证token
        user_data = AuthService.verify_token(token)
        if not user_data:
            return jsonify({
                'error': 'Token is invalid or expired',
                'message': 'Token无效或已过期'
            }), 401
        
        # 将用户信息存储到g对象
        g.current_user = user_data
        
        return f(*args, **kwargs)
    
    return decorated_function

def require_role(allowed_roles: List[str]):
    """角色权限装饰器"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not hasattr(g, 'current_user') or not g.current_user:
                return jsonify({
                    'error': 'Authentication required',
                    'message': '需要认证'
                }), 401
            
            user_role = g.current_user.get('role')
            if user_role not in allowed_roles:
                return jsonify({
                    'error': 'Insufficient permissions',
                    'message': '权限不足'
                }), 403
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator

def get_current_user() -> Optional[Dict[str, Any]]:
    """获取当前用户信息"""
    return getattr(g, 'current_user', None)

def optional_auth(f):
    """可选认证装饰器（不强制要求认证）"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None
        
        # 尝试获取token
        auth_header = request.headers.get('Authorization')
        if auth_header:
            try:
                token = auth_header.split(' ')[1]
            except IndexError:
                pass
        
        if not token:
            token = request.args.get('token')
        
        # 如果有token，尝试验证
        if token:
            user_data = AuthService.verify_token(token)
            if user_data:
                g.current_user = user_data
            else:
                g.current_user = None
        else:
            g.current_user = None
        
        return f(*args, **kwargs)
    
    return decorated_function

# 权限检查函数
def can_access_user_data(target_user_id: int) -> bool:
    """检查是否可以访问用户数据"""
    current_user = get_current_user()
    if not current_user:
        return False
    
    # 管理员可以访问所有用户数据
    if current_user.get('role') == 'admin':
        return True
    
    # 医生可以访问患者数据
    if current_user.get('role') == 'doctor':
        return True
    
    # 用户只能访问自己的数据
    if current_user.get('user_id') == target_user_id:
        return True
    
    return False

def can_manage_models() -> bool:
    """检查是否可以管理模型"""
    current_user = get_current_user()
    if not current_user:
        return False
    
    allowed_roles = ['admin', 'doctor']
    return current_user.get('role') in allowed_roles

def can_clear_cache() -> bool:
    """检查是否可以清除缓存"""
    current_user = get_current_user()
    if not current_user:
        return False
    
    allowed_roles = ['admin']
    return current_user.get('role') in allowed_roles

# 登录API装饰器的快捷函数
def admin_required(f):
    """管理员权限装饰器"""
    return require_role(['admin'])(require_auth(f))

def doctor_or_admin_required(f):
    """医生或管理员权限装饰器"""
    return require_role(['doctor', 'admin'])(require_auth(f))

# API密钥验证（用于系统间调用）
class APIKeyAuth:
    """API密钥认证"""
    
    # 模拟API密钥数据
    API_KEYS = {
        'shec_system_key_001': {
            'name': 'SHEC System',
            'permissions': ['read', 'write', 'admin'],
            'created_at': datetime.utcnow()
        },
        'shec_monitor_key_002': {
            'name': 'SHEC Monitor',
            'permissions': ['read'],
            'created_at': datetime.utcnow()
        }
    }
    
    @staticmethod
    def verify_api_key(api_key: str) -> Optional[Dict[str, Any]]:
        """验证API密钥"""
        return APIKeyAuth.API_KEYS.get(api_key)
    
    @staticmethod
    def require_api_key(required_permission: str = 'read'):
        """API密钥装饰器"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                api_key = request.headers.get('X-API-Key')
                if not api_key:
                    return jsonify({
                        'error': 'API key is missing',
                        'message': '缺少API密钥'
                    }), 401
                
                key_info = APIKeyAuth.verify_api_key(api_key)
                if not key_info:
                    return jsonify({
                        'error': 'Invalid API key',
                        'message': '无效的API密钥'
                    }), 401
                
                if required_permission not in key_info.get('permissions', []):
                    return jsonify({
                        'error': 'Insufficient API permissions',
                        'message': 'API权限不足'
                    }), 403
                
                g.api_key_info = key_info
                return f(*args, **kwargs)
            
            return decorated_function
        return decorator

# 速率限制（简单实现）
class RateLimiter:
    """简单的速率限制器"""
    
    def __init__(self):
        self.requests = {}  # {user_id: [(timestamp, count), ...]}
        self.limits = {
            'user': (100, 3600),      # 用户：100次/小时
            'doctor': (500, 3600),    # 医生：500次/小时
            'admin': (1000, 3600),    # 管理员：1000次/小时
            'anonymous': (20, 3600)   # 匿名：20次/小时
        }
    
    def is_allowed(self, user_id: Optional[int], user_role: str = 'anonymous') -> bool:
        """检查是否允许请求"""
        try:
            current_time = datetime.utcnow()
            key = f"{user_role}_{user_id}" if user_id else f"{user_role}_anonymous"
            
            limit_count, limit_window = self.limits.get(user_role, self.limits['anonymous'])
            
            # 清理过期记录
            if key in self.requests:
                self.requests[key] = [
                    (timestamp, count) for timestamp, count in self.requests[key]
                    if (current_time - timestamp).seconds < limit_window
                ]
            else:
                self.requests[key] = []
            
            # 计算当前窗口内的请求数
            current_count = sum(count for _, count in self.requests[key])
            
            if current_count >= limit_count:
                return False
            
            # 记录本次请求
            self.requests[key].append((current_time, 1))
            return True
            
        except Exception as e:
            logger.error(f"速率限制检查失败: {str(e)}")
            return True  # 出错时允许请求

# 全局速率限制器实例
rate_limiter = RateLimiter()

def rate_limit(f):
    """速率限制装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        current_user = get_current_user()
        user_id = current_user.get('user_id') if current_user else None
        user_role = current_user.get('role', 'anonymous') if current_user else 'anonymous'
        
        if not rate_limiter.is_allowed(user_id, user_role):
            return jsonify({
                'error': 'Rate limit exceeded',
                'message': '请求频率超限，请稍后再试'
            }), 429
        
        return f(*args, **kwargs)
    
    return decorated_function
