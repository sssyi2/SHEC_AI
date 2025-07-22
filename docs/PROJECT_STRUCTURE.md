# SHEC AI 项目结构说明

## 📁 项目目录结构

```
SHEC_AI/
├── 📁 api/                    # RESTful API接口
├── 📁 config/                 # 配置文件
├── 📁 data/                   # 数据存储
├── 📁 db_scripts/             # 数据库脚本
├── 📁 docs/                   # 项目文档 📚
│   ├── DATABASE_ADAPTATION.md
│   ├── DOCKER_BUILD_OPTIMIZATION.md
│   ├── DOCKER_SETUP_GUIDE.md
│   ├── ITERATION_PLAN.md
│   └── WEEK1_COMPLETION_REPORT.md
├── 📁 docker/                 # Docker相关文件 🐳
│   ├── compose/               # Docker Compose文件
│   │   ├── docker-compose.yml
│   │   ├── docker-compose.dev.yml
│   │   ├── docker-compose.gpu.yml
│   │   └── docker-compose.local.yml
│   ├── images/                # Dockerfile镜像文件
│   │   ├── Dockerfile
│   │   ├── Dockerfile.minimal
│   │   ├── Dockerfile.optimized
│   │   └── Dockerfile.ultra-minimal
│   ├── mysql/                 # MySQL配置
│   ├── nginx/                 # Nginx配置
│   ├── prometheus/            # Prometheus监控配置
│   └── redis/                 # Redis配置
├── 📁 examples/               # 使用示例
├── 📁 logs/                   # 日志文件
├── 📁 models/                 # AI模型定义
├── 📁 requirements/           # 依赖管理 📦
│   ├── requirements-base.txt
│   ├── requirements-ml.txt
│   └── requirements-web.txt
├── 📁 scripts/                # 工具脚本
├── 📁 services/               # 业务服务层
├── 📁 tests/                  # 测试文件 🧪
│   ├── test_sprint_31.py
│   ├── test_traditional_ml.py
│   └── test_training_framework.py
├── 📁 utils/                  # 工具函数
├── 📄 app.py                  # 主应用入口
├── 📄 app_compatible.py       # 兼容版本应用
├── 📄 dev_start.py           # 开发启动脚本
├── 📄 diagnose_nan.py        # 诊断工具
├── 📄 deploy.sh              # 部署脚本
├── 📄 docker-compose.yml     # 主要Docker编排文件
├── 📄 environment.yml        # Conda环境文件
└── 📄 requirements.txt       # 统一依赖文件
```

## 🗂️ 文件整理说明

### 已整理的文件类别：

1. **📚 文档文件** → `docs/` 目录
   - 所有 `.md` 文档文件已移动到专门的文档目录
   - 包括设计文档、部署指南、完成报告等

2. **🐳 Docker文件** → `docker/` 目录
   - `compose/` - 所有 docker-compose 文件
   - `images/` - 所有 Dockerfile 文件
   - 各服务配置目录（mysql, nginx, prometheus, redis）

3. **📦 依赖文件** → `requirements/` 目录
   - 按功能分类的依赖文件
   - 主 `requirements.txt` 通过 `-r` 引用子文件

4. **🧪 测试文件** → `tests/` 目录
   - 所有测试相关文件统一管理

### 保留在根目录的重要文件：

- `app.py` - 主应用程序入口
- `docker-compose.yml` - 主要的Docker编排文件（从compose/复制）
- `requirements.txt` - 统一的依赖管理文件
- `environment.yml` - Conda环境文件
- `deploy.sh` - 部署脚本
- 核心业务目录（api/, models/, services/, utils/等）

## 🚀 使用说明

### 开发环境启动：
```bash
# 安装依赖
pip install -r requirements.txt

# 启动开发服务器
python dev_start.py
```

### Docker环境启动：
```bash
# 启动全部服务
docker-compose up -d

# 启动开发环境
docker-compose -f docker/compose/docker-compose.dev.yml up -d
```

### 运行测试：
```bash
# 运行指定测试
python tests/test_sprint_31.py
```

## 📈 项目状态

- **Sprint 3.1**: ✅ 完成 (100% 测试通过)
- **文档管理**: ✅ 完成 (已整理至docs目录)
- **Docker配置**: ✅ 完成 (已整理至docker目录)
- **依赖管理**: ✅ 完成 (已分类至requirements目录)

---
**维护者**: SHEC AI团队  
**更新时间**: 2025年7月21日
