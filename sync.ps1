param([string]$msg = "")

if ([string]::IsNullOrWhiteSpace($msg)) {
  $msg = "update $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
}

# 1) 先拉取（rebase）
git pull --rebase
if ($LASTEXITCODE -ne 0) {
  Write-Host "`n[STOP] pull --rebase failed (likely conflict)."
  Write-Host "Resolve conflicts, then run:"
  Write-Host "  git add <files>"
  Write-Host "  git rebase --continue"
  exit 1
}

# 2) 再提交推送
git add -A
git commit -m "$msg"
git push