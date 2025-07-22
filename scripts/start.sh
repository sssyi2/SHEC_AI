#!/bin/bash

# SHEC AI 项目启动脚本

set -e

echo "🚀 启动SHEC AI医疗预测系统..."

# 检查Docker和Docker Compose
if ! command -v docker &> /dev/null; then
    echo "❌ Docker未安装，请先安装Docker"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose未安装，请先安装Docker Compose"
    exit 1
fi

# 创建必要的目录
echo "📁 创建必要的目录..."
mkdir -p logs models/saved_models data/uploads data/datasets config/ssl

# 生成自签名SSL证书（用于开发）
if [ ! -f "config/ssl/cert.pem" ]; then
    echo "🔐 生成自签名SSL证书..."
    openssl req -x509 -newkey rsa:4096 -keyout config/ssl/key.pem -out config/ssl/cert.pem -days 365 -nodes \
        -subj "/C=CN/ST=Beijing/L=Beijing/O=SHEC/OU=AI/CN=localhost"
fi

# 设置环境变量
export COMPOSE_PROJECT_NAME=shec-ai
export COMPOSE_HTTP_TIMEOUT=120

# 检查.env文件
if [ ! -f ".env" ]; then
    echo "⚙️  创建环境配置文件..."
    cat > .env << EOF
# 数据库配置
POSTGRES_DB=shec_ai
POSTGRES_USER=shec_user
POSTGRES_PASSWORD=shec_password_$(date +%s)

# Redis配置
REDIS_PASSWORD=redis_password_$(date +%s)

# JWT密钥
JWT_SECRET_KEY=jwt_secret_$(openssl rand -hex 32)

# Flask配置
FLASK_ENV=docker
FLASK_DEBUG=false

# GPU配置
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility
EOF
    echo "✅ 环境配置文件已创建: .env"
fi

# 检查GPU支持
if command -v nvidia-smi &> /dev/null; then
    echo "🔥 检测到NVIDIA GPU，启用GPU支持"
    GPU_SUPPORT="--profile gpu"
else
    echo "💻 未检测到NVIDIA GPU，使用CPU模式"
    GPU_SUPPORT=""
fi

# 构建和启动服务
echo "🔨 构建Docker镜像..."
docker-compose build --parallel

echo "🌟 启动服务..."
docker-compose up -d $GPU_SUPPORT

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 30

# 健康检查
echo "🏥 执行健康检查..."
for i in {1..10}; do
    if curl -f http://localhost/api/health >/dev/null 2>&1; then
        echo "✅ 服务启动成功！"
        break
    else
        echo "⏳ 等待服务启动... ($i/10)"
        sleep 10
    fi
    
    if [ $i -eq 10 ]; then
        echo "❌ 服务启动失败，请检查日志"
        docker-compose logs --tail=50
        exit 1
    fi
done

# 显示访问信息
echo ""
echo "🎉 SHEC AI医疗预测系统启动成功！"
echo ""
echo "📊 访问地址："
echo "   - 主应用: https://localhost"
echo "   - API文档: https://localhost/api/health"
echo "   - 监控面板: http://localhost:3000 (admin/admin123)"
echo "   - Prometheus: http://localhost:9090"
echo ""
echo "🔧 管理命令："
echo "   - 查看日志: docker-compose logs -f"
echo "   - 停止服务: docker-compose down"
echo "   - 重启服务: docker-compose restart"
echo ""
echo "📝 更多信息请查看 README.md"
echo ""
