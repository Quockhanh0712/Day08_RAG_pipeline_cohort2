param($f)
$content = Get-Content $f -Raw
$content = $content -replace 'pick 9caa66b', 'drop 9caa66b'
Set-Content $f $content
