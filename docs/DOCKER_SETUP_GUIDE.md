# Docker 安装和部署指导

**目标**: 完成SHEC AI项目的Docker化部署  
**状态**: Docker配置文件已完成，等待Docker安装  
**优先级**: 🔴 高（部署必需）

---

## 📦 当前Docker配置状态

### ✅ 已完成的配置文件
```
✅ Dockerfile              # Flask应用容器配置
✅ docker-compose.yml       # 多服务编排配置
✅ config/nginx.conf        # Nginx反向代理配置
✅ config/redis.conf        # Redis缓存配置
✅ config/prometheus.yml    # 监控配置
✅ scripts/start.sh         # Linux启动脚本
✅ scripts/start.bat        # Windows启动脚本
```

### 🔍 配置文件检查结果
```bash
Dockerfile: ✅ 60行完整配置
- 基于Python 3.11-slim
- GPU支持(CUDA)
- 依赖安装和环境配置
- Gunicorn生产服务器

docker-compose.yml: ✅ 215行完整配置
- PostgreSQL数据库服务
- Redis缓存服务  
- Flask AI应用服务
- Nginx反向代理
- Prometheus监控
- 完整的网络和卷配置
```

---

## 🛠️ Docker 安装指导

### Windows 系统安装

#### 方案1: Docker Desktop (推荐)
1. **下载Docker Desktop**
   ```
   https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe
   ```

2. **系统要求**
   - Windows 10 64位：Pro、Enterprise或Education版本
   - 或Windows 11 64位：Home或Pro版本
   - 启用WSL 2功能
   - 启用虚拟化功能

3. **安装步骤**
   ```powershell
   # 1. 启用WSL 2
   dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
   dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
   
   # 2. 重启系统
   
   # 3. 设置WSL 2为默认版本
   wsl --set-default-version 2
   
   # 4. 运行Docker Desktop安装程序
   ```

4. **GPU支持配置** (NVIDIA GPU)
   ```powershell
   # 安装NVIDIA Container Toolkit
   # Docker Desktop会自动处理GPU支持
   ```

#### 方案2: Docker Engine (无GUI)
```powershell
# 使用Chocolatey安装
choco install docker-desktop

# 或使用winget安装
winget install Docker.DockerDesktop
```

### 验证安装
```powershell
# 检查Docker版本
docker --version

# 检查Docker Compose
docker-compose --version

# 检查GPU支持(如果有NVIDIA GPU)
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### 可用国内镜像源列表

| 镜像源 | 地址 | 提供方 | 可用性 |
|--------|------|--------|--------|
| DaoCloud | `https://docker.m.daocloud.io` | DaoCloud | ⭐⭐⭐⭐⭐ |
| ImgDB | `https://docker.imgdb.de` | 第三方 | ⭐⭐⭐⭐ |
| Unsee | `https://docker-0.unsee.tech` | 第三方 | ⭐⭐⭐ |
| HLMirror | `https://docker.hlmirror.com` | 第三方 | ⭐⭐⭐ |
| 1MS | `https://docker.1ms.run` | 第三方 | ⭐⭐⭐⭐ |
| Func.ink | `https://func.ink` | 第三方 | ⭐⭐⭐ |
| Lispy | `https://lispy.org` | 第三方 | ⭐⭐⭐ |
| XGB1993 | `https://docker.xiaogenban1993.com` | 第三方 | ⭐⭐⭐ |
---

## 🚀 部署测试步骤

### 1. 预部署检查
```powershell
# 切换到项目目录
cd E:\vuework\SHEC-PSIMS\SHEC_AI

# 检查配置文件
docker-compose config

# 检查Dockerfile语法
docker build -t shec-ai-test . --no-cache
```

### 2. 开发环境启动
```powershell
# 使用启动脚本（推荐）
.\scripts\start.bat

# 或手动启动
docker-compose up -d
```

### 3. 服务验证
```powershell
# 检查服务状态
docker-compose ps

# 检查应用健康状态
curl http://localhost/api/health

# 查看日志
docker-compose logs shec_ai
```

### 4. 完整部署验证
```powershell
# 1. 数据库连接测试
curl http://localhost/api/health

# 2. 缓存服务测试
docker exec shec_redis redis-cli ping

# 3. AI预测API测试
curl -X POST http://localhost/api/predict/health \
  -H "Content-Type: application/json" \
  -d '{"age": 45, "gender": "M", "symptoms": ["头痛", "发热"]}'
```

---

## 🔧 配置文件详解

### docker-compose.yml 服务架构
```yaml
services:
  postgres:     # 数据库服务
    - 端口: 5432
    - 数据持久化: postgres_data卷
    - 健康检查: pg_isready
  
  redis:        # 缓存服务
    - 端口: 6379  
    - 配置文件: config/redis.conf
    - 健康检查: redis-cli ping
  
  shec_ai:      # Flask应用
    - 端口: 5000
    - GPU支持: 配置runtime: nvidia
    - 依赖: postgres, redis
  
  nginx:        # 反向代理
    - 端口: 80, 443
    - SSL支持: 自签名证书
    - 负载均衡: upstream配置
  
  prometheus:   # 监控服务
    - 端口: 9090
    - 指标收集: Flask应用指标
```

### 环境变量配置
```bash
# .env 文件示例
FLASK_ENV=docker
DATABASE_URL=postgresql://shec_user:shec_password@postgres:5432/shec_ai
REDIS_URL=redis://redis:6379/0
SECRET_KEY=your-secret-key-here
```

---

## 🐛 常见问题解决

### 1. Docker启动失败
```powershell
# 检查WSL 2状态
wsl --list --verbose

# 重启Docker Desktop
# 任务栏右键Docker图标 -> Restart
```

### 2. GPU支持问题
```powershell
# 检查NVIDIA驱动
nvidia-smi

# 验证Docker GPU支持
docker run --rm --gpus all hello-world
```

### 3. 端口冲突
```powershell
# 检查端口占用
netstat -an | findstr :5000
netstat -an | findstr :6379

# 修改docker-compose.yml中的端口映射
```

### 4. 数据库连接失败
```powershell
# 检查数据库容器状态
docker-compose logs postgres

# 手动连接测试
docker exec -it shec_postgres psql -U shec_user -d shec_ai
```

---

## 📋 完成标准

Docker化部署配置**真正完成**的标准：

### ✅ 基础要求
- [ ] Docker Desktop安装并运行正常
- [ ] docker-compose up 成功启动所有服务
- [ ] 所有容器健康检查通过

### ✅ 功能验证
- [ ] Flask应用容器正常启动
- [ ] 数据库连接正常
- [ ] Redis缓存功能正常
- [ ] Nginx反向代理工作
- [ ] 监控服务可访问

### ✅ API测试
- [ ] `/api/health` 返回正常状态
- [ ] 预测API接口响应正常
- [ ] 数据库读写操作正常
- [ ] 缓存读写操作正常

### ✅ 性能测试
- [ ] GPU容器内可访问(如果有GPU)
- [ ] 负载测试通过
- [ ] 日志记录正常
- [ ] 监控指标收集正常

---

## 🎯 下一步行动

### 立即行动 (今天)
1. **安装Docker Desktop**
2. **配置WSL 2环境**
3. **验证GPU支持**

### 验证测试 (明天)
1. **运行部署脚本**
2. **完整功能测试**
3. **性能基准测试**

### 优化调整 (本周内)
1. **监控配置优化**
2. **安全策略配置**
3. **生产环境准备**

---

*文档创建时间: 2025年7月18日*  
*状态: Docker配置就绪，等待安装验证*
