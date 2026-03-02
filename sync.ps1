param([string]$msg = "")

if ([string]::IsNullOrWhiteSpace($msg)) {
  $msg = "update $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
}

git pull --rebase
git add -A
git commit -m "$msg"
git push