@echo off
REM SHEC AI 项目启动脚本 (Windows版本)

setlocal enabledelayedexpansion

echo 🚀 启动SHEC AI医疗预测系统...

REM 检查Docker
where docker >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Docker未安装，请先安装Docker Desktop
    pause
    exit /b 1
)

REM 检查Docker Compose
docker compose version >nul 2>nul
if %errorlevel% neq 0 (
    docker-compose --version >nul 2>nul
    if !errorlevel! neq 0 (
        echo ❌ Docker Compose未安装，请先安装Docker Compose
        pause
        exit /b 1
    )
)

REM 创建必要的目录
echo 📁 创建必要的目录...
if not exist "logs" mkdir logs
if not exist "models\saved_models" mkdir models\saved_models
if not exist "data\uploads" mkdir data\uploads
if not exist "data\datasets" mkdir data\datasets
if not exist "config\ssl" mkdir config\ssl

REM 生成自签名SSL证书（用于开发）
if not exist "config\ssl\cert.pem" (
    echo 🔐 生成自签名SSL证书...
    openssl req -x509 -newkey rsa:4096 -keyout config\ssl\key.pem -out config\ssl\cert.pem -days 365 -nodes -subj "/C=CN/ST=Beijing/L=Beijing/O=SHEC/OU=AI/CN=localhost" 2>nul
    if !errorlevel! neq 0 (
        echo ⚠️  OpenSSL未找到，跳过SSL证书生成
        echo    请手动生成SSL证书或使用HTTP访问
    )
)

REM 设置环境变量
set COMPOSE_PROJECT_NAME=shec-ai
set COMPOSE_HTTP_TIMEOUT=120

REM 检查.env文件
if not exist ".env" (
    echo ⚙️  创建环境配置文件...
    (
        echo # 数据库配置
        echo POSTGRES_DB=shec_ai
        echo POSTGRES_USER=shec_user
        echo POSTGRES_PASSWORD=shec_password_%random%
        echo.
        echo # Redis配置
        echo REDIS_PASSWORD=redis_password_%random%
        echo.
        echo # JWT密钥
        echo JWT_SECRET_KEY=jwt_secret_%random%%random%
        echo.
        echo # Flask配置
        echo FLASK_ENV=docker
        echo FLASK_DEBUG=false
        echo.
        echo # GPU配置
        echo NVIDIA_VISIBLE_DEVICES=all
        echo NVIDIA_DRIVER_CAPABILITIES=compute,utility
    ) > .env
    echo ✅ 环境配置文件已创建: .env
)

REM 检查GPU支持
nvidia-smi >nul 2>nul
if %errorlevel% equ 0 (
    echo 🔥 检测到NVIDIA GPU，启用GPU支持
    set GPU_SUPPORT=--profile gpu
) else (
    echo 💻 未检测到NVIDIA GPU，使用CPU模式
    set GPU_SUPPORT=
)

REM 构建和启动服务
echo 🔨 构建Docker镜像...
docker-compose build --parallel

echo 🌟 启动服务...
docker-compose up -d %GPU_SUPPORT%

REM 等待服务启动
echo ⏳ 等待服务启动...
timeout /t 30 /nobreak >nul

REM 健康检查
echo 🏥 执行健康检查...
set /a attempts=0
:healthcheck
set /a attempts+=1
curl -f http://localhost/api/health >nul 2>nul
if %errorlevel% equ 0 (
    echo ✅ 服务启动成功！
    goto success
)

if %attempts% lss 10 (
    echo ⏳ 等待服务启动... (%attempts%/10)
    timeout /t 10 /nobreak >nul
    goto healthcheck
)

echo ❌ 服务启动失败，请检查日志
docker-compose logs --tail=50
pause
exit /b 1

:success
REM 显示访问信息
echo.
echo 🎉 SHEC AI医疗预测系统启动成功！
echo.
echo 📊 访问地址：
echo    - 主应用: https://localhost
echo    - API文档: https://localhost/api/health
echo    - 监控面板: http://localhost:3000 (admin/admin123)
echo    - Prometheus: http://localhost:9090
echo.
echo 🔧 管理命令：
echo    - 查看日志: docker-compose logs -f
echo    - 停止服务: docker-compose down
echo    - 重启服务: docker-compose restart
echo.
echo 📝 更多信息请查看 README.md
echo.
echo 按任意键退出...
pause >nul
