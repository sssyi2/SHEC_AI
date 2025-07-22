# pip镜像源测试脚本
# 测试各个镜像源的可用性和速度

Write-Host "🔍 测试可用的pip镜像源..." -ForegroundColor Cyan
Write-Host ""

$mirrors = @(
    @{name="阿里云"; url="https://mirrors.aliyun.com/pypi/simple/"; host="mirrors.aliyun.com"},
    @{name="清华大学"; url="https://pypi.tuna.tsinghua.edu.cn/simple/"; host="pypi.tuna.tsinghua.edu.cn"},
    @{name="豆瓣"; url="https://pypi.douban.com/simple/"; host="pypi.douban.com"},
    @{name="网易"; url="https://mirrors.163.com/pypi/simple/"; host="mirrors.163.com"},
    @{name="华为云"; url="https://mirrors.huaweicloud.com/repository/pypi/simple/"; host="mirrors.huaweicloud.com"},
    @{name="中科大"; url="https://pypi.mirrors.ustc.edu.cn/simple/"; host="pypi.mirrors.ustc.edu.cn"}
)

$working_mirrors = @()

foreach ($mirror in $mirrors) {
    Write-Host "测试 $($mirror.name): " -NoNewline -ForegroundColor Yellow
    
    $start_time = Get-Date
    try {
        $response = Invoke-WebRequest -Uri $mirror.url -TimeoutSec 10 -UseBasicParsing
        $end_time = Get-Date
        $response_time = ($end_time - $start_time).TotalMilliseconds
        
        if ($response.StatusCode -eq 200) {
            Write-Host "✅ 可用 ($([math]::Round($response_time))ms)" -ForegroundColor Green
            $working_mirrors += $mirror
        }
    }
    catch {
        Write-Host "❌ 不可用: $($_.Exception.Message.Split(':')[0])" -ForegroundColor Red
    }
}

Write-Host ""
if ($working_mirrors.Count -gt 0) {
    Write-Host "🚀 推荐使用以下镜像源:" -ForegroundColor Green
    foreach ($mirror in $working_mirrors[0..2]) {  # 显示前3个可用的
        Write-Host "  • $($mirror.name): $($mirror.url)" -ForegroundColor Cyan
    }
    
    Write-Host ""
    Write-Host "📝 Docker中使用示例:" -ForegroundColor Yellow
    $best_mirror = $working_mirrors[0]
    Write-Host "RUN pip install --no-cache-dir -i $($best_mirror.url) \\" -ForegroundColor White
    Write-Host "    --trusted-host $($best_mirror.host) --timeout 1000 \\" -ForegroundColor White
    Write-Host "    package_name" -ForegroundColor White
} else {
    Write-Host "❌ 没有找到可用的镜像源" -ForegroundColor Red
}

Write-Host ""
Write-Host "⏱️  测试完成！" -ForegroundColor Magenta
