# PCとロボット(既定 192.168.96.215)の時計ずれを測り、閾値を超えていれば自動で合わせる。
#
# Windows本体 (w32time) と WSL (ntpdate) は別々にずれるので両方を確認・補正する。
#   * Windows側の補正には管理者権限が必要。ずれている時だけUACで昇格する。
#   * WSL側の補正は wsl -u root で行うので昇格不要。
#
# 使い方 (PowerShell):
#   .\tools\sync_robot_clock.ps1                # 測定して必要なら補正
#   .\tools\sync_robot_clock.ps1 -CheckOnly     # 測定のみ (ずれていれば終了コード1)
#   .\tools\sync_robot_clock.ps1 -Install       # ログオン時+30分毎に自動実行するタスクを登録
#   .\tools\sync_robot_clock.ps1 -Uninstall     # 上記タスクを削除
# 実機を動かす前 (run_camera_pipeline_test.py の前) に -CheckOnly を挟むと安全。
# 実行ポリシーで弾かれる場合: powershell -ExecutionPolicy Bypass -File tools\sync_robot_clock.ps1

param(
    [string]$Robot = '192.168.96.215',
    [string]$Distro = 'Ubuntu-20.04',
    [double]$Threshold = 0.1,       # [s] これを超えたら補正する
    [switch]$CheckOnly,
    [switch]$Install,
    [switch]$Uninstall,
    [string]$FixWindowsLog = ''     # 内部用: 昇格した子プロセスがWindows側だけ直してログを書く
)

$TaskName = 'SyncRobotClock'

function Test-Admin {
    $p = New-Object Security.Principal.WindowsPrincipal(
        [Security.Principal.WindowsIdentity]::GetCurrent())
    return $p.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Windows時計のロボットに対するずれ [s] (絶対値)。測れなければ $null
function Get-WindowsOffset {
    $out = w32tm /stripchart /computer:$Robot /samples:3 /dataonly 2>&1 | Out-String
    $vals = @([regex]::Matches($out, '([+-]\d+\.\d+)s') | ForEach-Object { [math]::Abs([double]$_.Groups[1].Value) })
    if ($vals.Count -eq 0) { return $null }
    return ($vals | Measure-Object -Minimum).Minimum
}

# WSL時計のロボットに対するずれ [s] (絶対値)。測れなければ $null
function Get-WslOffset {
    $out = wsl -d $Distro -u root -- ntpdate -q $Robot 2>&1 | Out-String
    $m = [regex]::Matches($out, 'offset\s+(-?\d+\.\d+)')
    if ($m.Count -eq 0) { return $null }
    return [math]::Abs([double]$m[$m.Count - 1].Groups[1].Value)
}

function Fix-Windows {
    $svc = Get-Service w32time
    Set-Service w32time -StartupType Automatic
    if ($svc.Status -ne 'Running') { Start-Service w32time }
    w32tm /config /manualpeerlist:$Robot /syncfromflags:manual /update | Out-Null
    w32tm /resync /rediscover | Out-Null
    w32tm /resync /force | Out-Null
}

function Fix-Wsl {
    # -u なしだと "NTP socket is in use" で失敗する
    wsl -d $Distro -u root -- ntpdate -u $Robot | Out-Null
}

function Show([string]$name, $off) {
    if ($null -eq $off) { Write-Host ("  {0,-8}: measurement failed" -f $name) }
    else { Write-Host ("  {0,-8}: {1:N3} s" -f $name, $off) }
}

function Is-Bad($off) { return ($null -eq $off) -or ($off -gt $Threshold) }

if ($Uninstall) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "removed task $TaskName"
    exit 0
}

if ($Install) {
    if (-not (Test-Admin)) {
        # 最上位特権のタスク登録は管理者権限が要るので昇格して自分を呼び直す
        Write-Host 'requesting administrator rights...'
        Start-Process powershell.exe -Verb RunAs -WindowStyle Hidden -Wait -ArgumentList @(
            '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$PSCommandPath`"", '-Install')
        if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
            Write-Host "registered task $TaskName (logon + every 30 min)"; exit 0
        }
        Write-Host "failed to register task $TaskName"; exit 1
    }
    $arg = '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "{0}"' -f $PSCommandPath
    $action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arg
    $t1 = New-ScheduledTaskTrigger -AtLogOn
    $t2 = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 30)
    # 最上位の特権で実行すればUACなしでWindows側も直せる
    $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -RunLevel Highest -LogonType Interactive
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger @($t1, $t2) `
        -Principal $principal -Force -ErrorAction Stop | Out-Null
    exit 0
}

if ($FixWindowsLog -ne '') {
    # 昇格した子プロセス
    Start-Transcript -Path $FixWindowsLog -Force | Out-Null
    Fix-Windows
    Stop-Transcript | Out-Null
    exit 0
}

# 定期実行は画面に出ないので、毎回の結果をログに追記する
$LogPath = Join-Path $env:LOCALAPPDATA 'sync_robot_clock.log'
if (-not $CheckOnly) { Start-Transcript -Path $LogPath -Append | Out-Null }

Write-Host "robot = $Robot, threshold = $Threshold s"
$win = Get-WindowsOffset
$wsl = Get-WslOffset
Write-Host 'before:'; Show 'Windows' $win; Show 'WSL' $wsl
if ($null -eq $win -and $null -eq $wsl) {
    Write-Host "cannot reach $Robot"; exit 2
}
if ($CheckOnly) {
    if ((Is-Bad $win) -or (Is-Bad $wsl)) { Write-Host 'NG: clock offset too large'; exit 1 }
    Write-Host 'OK'; exit 0
}

if (Is-Bad $win) {
    if (Test-Admin) { Fix-Windows }
    else {
        Write-Host 'Windows clock is off: requesting administrator rights...'
        $log = Join-Path $env:TEMP 'sync_robot_clock_win.log'
        $argList = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$PSCommandPath`"",
                     '-Robot', $Robot, '-FixWindowsLog', "`"$log`"")
        Start-Process powershell.exe -Verb RunAs -WindowStyle Hidden -Wait -ArgumentList $argList
    }
}
# WSLはWindowsを直してから測り直す (WSLはWindowsの時計に追従することがある)
$wsl = Get-WslOffset
if (Is-Bad $wsl) { Fix-Wsl }

$win = Get-WindowsOffset
$wsl = Get-WslOffset
Write-Host 'after:'; Show 'Windows' $win; Show 'WSL' $wsl
if ((Is-Bad $win) -or (Is-Bad $wsl)) { Write-Host 'NG: still off'; exit 1 }
Write-Host 'OK'
