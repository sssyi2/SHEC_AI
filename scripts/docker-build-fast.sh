#!/bin/bash

# Docker 快速构建脚本
# 使用多种优化策略加速构建过程

set -e

echo "🚀 启动 Docker 快速构建流程..."

# 颜色输出函数
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查 Docker 是否运行
if ! docker info > /dev/null 2>&1; then
    error "Docker 未运行，请启动 Docker Desktop"
    exit 1
fi

success "Docker 运行正常"

# 启用 BuildKit 加速构建
export DOCKER_BUILDKIT=1
info "已启用 BuildKit 加速构建"

# 构建选项
BUILD_TYPE=${1:-"minimal"}  # 默认最小化构建
IMAGE_TAG="shec-ai"

case $BUILD_TYPE in
    "minimal")
        info "执行最小化构建（用于快速测试）..."
        DOCKERFILE="Dockerfile.minimal"
        IMAGE_TAG="shec-ai-minimal"
        ;;
    "optimized")
        info "执行优化构建（分层缓存）..."
        DOCKERFILE="Dockerfile.optimized"
        IMAGE_TAG="shec-ai-optimized"
        ;;
    "full")
        info "执行完整构建..."
        DOCKERFILE="Dockerfile"
        IMAGE_TAG="shec-ai-full"
        ;;
    *)
        error "未知构建类型: $BUILD_TYPE"
        echo "用法: $0 [minimal|optimized|full]"
        exit 1
        ;;
esac

# 记录开始时间
START_TIME=$(date +%s)

info "开始构建镜像: $IMAGE_TAG"
info "使用 Dockerfile: $DOCKERFILE"

# 执行构建
if docker build \
    -f $DOCKERFILE \
    -t $IMAGE_TAG \
    --progress=plain \
    --no-cache \
    .; then
    
    END_TIME=$(date +%s)
    BUILD_TIME=$((END_TIME - START_TIME))
    
    success "构建完成！"
    success "构建时间: ${BUILD_TIME} 秒"
    success "镜像标签: $IMAGE_TAG"
    
    # 显示镜像信息
    info "镜像详情:"
    docker images $IMAGE_TAG --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    
    # 询问是否启动测试
    read -p "是否启动容器进行测试？(y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        info "启动测试容器..."
        docker run -d \
            --name shec-ai-test \
            -p 5000:5000 \
            --rm \
            $IMAGE_TAG
        
        sleep 3
        
        # 测试健康检查
        if curl -f http://localhost:5000/api/health > /dev/null 2>&1; then
            success "健康检查通过！"
            success "访问地址: http://localhost:5000"
        else
            warning "健康检查失败，请检查容器日志"
            docker logs shec-ai-test
        fi
        
        info "使用以下命令查看容器日志:"
        echo "docker logs shec-ai-test"
        info "使用以下命令停止容器:"
        echo "docker stop shec-ai-test"
    fi
    
else
    error "构建失败！"
    exit 1
fi
