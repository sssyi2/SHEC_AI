@echo off
REM Docker 快速构建脚本 - Windows 版本
REM 使用多种优化策略加速构建过程
REM 注意：如果在PowerShell中出现乱码，请使用 docker-build-fast.ps1
REM 用法: docker-build-fast.bat [minimal|optimized|full|ultra]

chcp 65001 >nul
setlocal enabledelayedexpansion

if "%1"=="ultra" (
    echo [INFO] 启动 Docker 超快构建流程 ^(30-60秒^)...
) else (
    echo [INFO] 启动 Docker 快速构建流程...
)
echo.

REM 检查 Docker 是否运行
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker 未运行，请启动 Docker Desktop
    pause
    exit /b 1
)

echo [SUCCESS] Docker 运行正常

REM 检查镜像源配置
echo [INFO] 检查 Docker 镜像源配置...
docker info | findstr "Registry Mirrors" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] 未检测到镜像源配置，建议配置以加速下载
    echo [INFO] 参考文档: DOCKER_MIRROR_CONFIG.md
) else (
    echo [SUCCESS] 已配置 Docker 镜像源
)

REM 启用 BuildKit 加速构建
set DOCKER_BUILDKIT=1
echo ✅ 已启用 BuildKit 加速构建

REM 构建选项
set BUILD_TYPE=%1
if "%BUILD_TYPE%"=="" set BUILD_TYPE=minimal

if "%BUILD_TYPE%"=="ultra" (
    echo [INFO] 执行超快构建（30-60秒完成）...
    set DOCKERFILE=Dockerfile.ultra-minimal
    set IMAGE_TAG=shec-ai-ultra
) else if "%BUILD_TYPE%"=="minimal" (
    echo [INFO] 执行最小化构建（用于快速测试）...
    set DOCKERFILE=Dockerfile.minimal
    set IMAGE_TAG=shec-ai-minimal
) else if "%BUILD_TYPE%"=="optimized" (
    echo [INFO] 执行优化构建（分层缓存）...
    set DOCKERFILE=Dockerfile.optimized
    set IMAGE_TAG=shec-ai-optimized
) else if "%BUILD_TYPE%"=="full" (
    echo [INFO] 执行完整构建...
    set DOCKERFILE=Dockerfile
    set IMAGE_TAG=shec-ai-full
) else (
    echo [ERROR] 未知构建类型: %BUILD_TYPE%
    echo 用法: %0 [ultra^|minimal^|optimized^|full]
    pause
    exit /b 1
)

REM 记录开始时间
set START_TIME=%time%

echo.
echo 🔨 开始构建镜像: %IMAGE_TAG%
echo 📄 使用 Dockerfile: %DOCKERFILE%
echo.

REM 执行构建
docker build -f %DOCKERFILE% -t %IMAGE_TAG% --progress=plain --no-cache .

if errorlevel 1 (
    echo.
    echo ❌ 构建失败！
    pause
    exit /b 1
)

REM 记录结束时间
set END_TIME=%time%

echo.
echo ✅ 构建完成！
echo ✅ 镜像标签: %IMAGE_TAG%
echo.

REM 显示镜像信息
echo 📊 镜像详情:
docker images %IMAGE_TAG% --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
echo.

REM 询问是否启动测试
set /p REPLY=是否启动容器进行测试？(y/n): 
if /i "%REPLY%"=="y" (
    echo.
    echo 🚀 启动测试容器...
    
    REM 停止可能存在的旧容器
    docker stop shec-ai-test >nul 2>&1
    
    REM 启动新容器
    docker run -d --name shec-ai-test -p 5000:5000 --rm %IMAGE_TAG%
    
    echo 等待容器启动...
    timeout /t 5 /nobreak >nul
    
    REM 测试健康检查
    curl -f http://localhost:5000/api/health >nul 2>&1
    if errorlevel 1 (
        echo ⚠️ 健康检查失败，请检查容器日志
        echo 查看容器日志:
        docker logs shec-ai-test
    ) else (
        echo ✅ 健康检查通过！
        echo ✅ 访问地址: http://localhost:5000
    )
    
    echo.
    echo 📝 常用命令:
    echo   查看日志: docker logs shec-ai-test
    echo   停止容器: docker stop shec-ai-test
    echo.
)

echo 🎉 构建流程完成！
pause
