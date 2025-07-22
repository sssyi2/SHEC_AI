# Docker 快速构建脚本 - PowerShell 版本
# 使用多种优化策略加速构建过程

param(
    [string]$BuildType = "minimal",
    [switch]$UltraFast
)

# 设置控制台编码为UTF-8
$OutputEncoding = [console]::InputEncoding = [console]::OutputEncoding = New-Object System.Text.UTF8Encoding

# 颜色输出函数
function Write-Info { Write-Host "[INFO] $args" -ForegroundColor Blue }
function Write-Success { Write-Host "[SUCCESS] $args" -ForegroundColor Green }
function Write-Warning { Write-Host "[WARNING] $args" -ForegroundColor Yellow }
function Write-Error { Write-Host "[ERROR] $args" -ForegroundColor Red }

Write-Host ""
if ($UltraFast) {
    Write-Info "启动 Docker 超快构建流程 (30-60秒)..."
} else {
    Write-Info "启动 Docker 快速构建流程..."
}
Write-Host ""

# 检查 Docker 是否运行
try {
    docker info | Out-Null
    Write-Success "Docker 运行正常"
} catch {
    Write-Error "Docker 未运行，请启动 Docker Desktop"
    Read-Host "按任意键退出"
    exit 1
}

# 检查镜像源配置
Write-Info "检查 Docker 镜像源配置..."
$registryMirrors = docker info | Select-String "Registry Mirrors"
if ($registryMirrors) {
    Write-Success "已配置 Docker 镜像源"
} else {
    Write-Warning "未检测到镜像源配置，建议配置以加速下载"
    Write-Info "参考文档: DOCKER_MIRROR_CONFIG.md"
}

# 启用 BuildKit 加速构建
$env:DOCKER_BUILDKIT = "1"
Write-Info "已启用 BuildKit 加速构建"

# 构建选项
$dockerfile = ""
$imageTag = ""

# 优先检查是否使用超快模式
if ($UltraFast) {
    Write-Info "执行超快构建（30-60秒完成）..."
    $dockerfile = "Dockerfile.ultra-minimal"
    $imageTag = "shec-ai-ultra"
} else {
    switch ($BuildType) {
        "minimal" {
            Write-Info "执行最小化构建（用于快速测试）..."
            $dockerfile = "Dockerfile.minimal"
            $imageTag = "shec-ai-minimal"
        }
        "optimized" {
            Write-Info "执行优化构建（分层缓存）..."
            $dockerfile = "Dockerfile.optimized"
            $imageTag = "shec-ai-optimized"
        }
        "full" {
            Write-Info "执行完整构建..."
            $dockerfile = "Dockerfile"
            $imageTag = "shec-ai-full"
        }
        default {
            Write-Error "未知构建类型: $BuildType"
            Write-Host "用法: .\docker-build-fast.ps1 [minimal|optimized|full] [-UltraFast]"
            exit 1
        }
    }
}

# 记录开始时间
$startTime = Get-Date

Write-Host ""
Write-Info "开始构建镜像: $imageTag"
Write-Info "使用 Dockerfile: $dockerfile"
Write-Host ""

# 执行构建
try {
    docker build -f $dockerfile -t $imageTag --progress=plain --no-cache .
    
    $endTime = Get-Date
    $buildTime = ($endTime - $startTime).TotalSeconds
    
    Write-Host ""
    Write-Success "构建完成！"
    Write-Success "构建时间: $([math]::Round($buildTime, 2)) 秒"
    Write-Success "镜像标签: $imageTag"
    
    # 显示镜像信息
    Write-Info "镜像详情:"
    docker images $imageTag --format "table {{.Repository}}`t{{.Tag}}`t{{.Size}}`t{{.CreatedAt}}"
    
    # 询问是否启动测试
    Write-Host ""
    $reply = Read-Host "是否启动容器进行测试？(y/n)"
    if ($reply -match "^[Yy]") {
        Write-Info "启动测试容器..."
        
        # 停止可能存在的旧容器
        try { docker stop shec-ai-test | Out-Null } catch {}
        
        # 启动新容器
        docker run -d --name shec-ai-test -p 5000:5000 --rm $imageTag | Out-Null
        
        Write-Info "等待容器启动..."
        Start-Sleep 5
        
        # 测试健康检查
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:5000/api/health" -TimeoutSec 10 -UseBasicParsing
            if ($response.StatusCode -eq 200) {
                Write-Success "健康检查通过！"
                Write-Success "访问地址: http://localhost:5000"
            } else {
                Write-Warning "健康检查返回状态码: $($response.StatusCode)"
            }
        } catch {
            Write-Warning "健康检查失败，请检查容器日志"
            Write-Info "查看容器日志: docker logs shec-ai-test"
        }
        
        Write-Host ""
        Write-Info "常用命令:"
        Write-Host "  查看日志: docker logs shec-ai-test"
        Write-Host "  停止容器: docker stop shec-ai-test"
        Write-Host ""
    }
    
} catch {
    Write-Host ""
    Write-Error "构建失败！"
    Write-Error $_.Exception.Message
    exit 1
}

Write-Success "构建流程完成！"
Read-Host "按任意键退出"
