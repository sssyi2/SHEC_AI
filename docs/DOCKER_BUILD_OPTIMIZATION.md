# Docker 构建性能优化指南

## 🚀 快速构建文件说明

### 核心文件
- `.dockerignore` - 排除不必要的文件，减少构建上下文
- `Dockerfile.minimal` - 最小化构建，仅包含核心功能（1-2分钟）
- `Dockerfile.optimized` - 优化构建，分层缓存策略（5-10分钟）
- `requirements-*.txt` - 分离的依赖文件，充分利用 Docker 缓存层

### 构建脚本
- `scripts/docker-build-fast.sh` - Linux/Mac 快速构建脚本
- `scripts/docker-build-fast.bat` - Windows 快速构建脚本

## 📊 构建时间对比

| 构建类型 | 文件 | 预期时间 | 功能完整度 |
|---------|------|----------|-----------|
| 最小化 | `Dockerfile.minimal` | 1-2分钟 | 核心API功能 |
| 优化版 | `Dockerfile.optimized` | 5-10分钟 | 完整功能（无GPU） |
| 完整版 | `Dockerfile` | 15-30分钟 | 全功能（含GPU） |

## 🎯 使用方法

### Windows 用户
```powershell
# 最小化构建（推荐测试）
.\scripts\docker-build-fast.bat minimal

# 优化构建
.\scripts\docker-build-fast.bat optimized

# 完整构建
.\scripts\docker-build-fast.bat full
```

### Linux/Mac 用户
```bash
# 给脚本执行权限
chmod +x scripts/docker-build-fast.sh

# 最小化构建
./scripts/docker-build-fast.sh minimal

# 优化构建
./scripts/docker-build-fast.sh optimized
```

## 🔧 优化策略

### 1. 分层缓存
- 系统依赖层（很少变化）
- 基础 Python 包层（偶尔变化）
- Web 框架层（较少变化）
- 应用代码层（经常变化）

### 2. 构建上下文优化
- `.dockerignore` 排除不必要文件
- 减少构建上下文大小 90%+

### 3. 依赖安装优化
- 分离 requirements 文件
- **Docker 镜像源在 daemon.json 中已配置**
- **pip 包下载通过 Docker 镜像源加速**
- CPU 版本 PyTorch（开发环境）

## 🚀 立即测试

```powershell
# 进入项目目录
cd E:\vuework\SHEC-PSIMS\SHEC_AI

# 执行最小化构建（最快）
.\scripts\docker-build-fast.bat minimal
```

预期结果：1-2分钟内完成构建并可以测试基本API功能。
