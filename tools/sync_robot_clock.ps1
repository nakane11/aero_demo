# PCとロボット(既定 192.168.96.215)の時計ずれを測り、閾値を超えていれば自動で合わせる。
#
# Windows本体 (w32time) と WSL (ntpdate) の両方を確認・補正する。WSLの時計はWindowsの
# 時計へ自動で合わせられるので、必ずWindowsを先に直し、直らなければWSLには触らない。
#   * Windows側の補正には管理者権限が必要。ずれている時だけUACで昇格する。
#   * WSL側の補正は wsl -u root で行うので昇格不要。
#
# 使い方 (PowerShell):
#   .\tools\sync_robot_clock.ps1                # 測定して必要なら補正
#   .\tools\sync_robot_clock.ps1 -CheckOnly     # 測定のみ (ずれていれば終了コード1)
#   .\tools\sync_robot_clock.ps1 -Install       # ログオン時+30分毎に自動実行するタスクを登録
#   .\tools\sync_robot_clock.ps1 -Uninstall     # 上記タスクを削除
# run_camera_pipeline_test.py は起動時にずれを見つけると -Install のタスクを起動して
# このスクリプトで合わせる (--no-clock-sync で無効)。ロボットの電源が入る前は測れない。
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

# WSLカーネルの tick [us] (標準 10000)。測れなければ $null。
# WSLがWindowsより数秒以上先行するとhv_utilsが tick を下げて (実測 9167) 引き戻す。
# struct timex の tick はx86_64で先頭から88バイト目 (PowerShell 5.1 からネイティブexeへ
# 引用符を渡すと壊れるので、pythonのコードは引用符なしで書く)。
function Get-WslTick {
    $py = 'import ctypes,ctypes.util as u,sys;b=ctypes.create_string_buffer(512);' +
          'ctypes.CDLL(u.find_library(chr(99))).adjtimex(b);' +
          'print(int.from_bytes(b.raw[88:96],sys.byteorder,signed=True))'
    # -e: シェルを通さない (; や ( をbashに解釈させない)
    $out = wsl -d $Distro -e python3 -c $py 2>&1 | Out-String
    if ($out -match '^\s*(\d+)\s*$') { return [int]$Matches[1] }
    return $null
}

function Is-Slewing($tick) { return ($null -ne $tick) -and ([math]::Abs($tick - 10000) -gt 100) }

# WSLがWindowsへ引き戻されている最中は待つ。途中でntpdateすると引き戻しは予定の量まで
# 続くので、今度は逆向きにずれる (10/30 実測で +4秒 → −4秒)。
# $QuietSec: Windowsを動かした直後は引き戻しが始まるまで約25秒かかるので、その間も待つ。
function Wait-WslSettled([int]$QuietSec, [int]$TimeoutSec = 300) {
    $start = Get-Date
    $quietSince = Get-Date
    $announced = $false
    while (((Get-Date) - $start).TotalSeconds -lt $TimeoutSec) {
        $tick = Get-WslTick
        if (Is-Slewing $tick) {
            if (-not $announced) {
                Write-Host "WSL clock is being pulled toward Windows (tick=$tick); waiting..."
                $announced = $true
            }
            $quietSince = Get-Date
        } else {
            if (((Get-Date) - $quietSince).TotalSeconds -ge $QuietSec) { return $true }
            if ($QuietSec -gt 0 -and -not (Is-Bad (Get-WslOffset))) { return $true }
        }
        Start-Sleep -Seconds 5
    }
    return $false
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
$tick = Get-WslTick
Write-Host 'before:'; Show 'Windows' $win; Show 'WSL' $wsl; Write-Host "  WSL tick: $tick"
if ($null -eq $win -and $null -eq $wsl) {
    Write-Host "cannot reach $Robot"; exit 2
}
if ($CheckOnly) {
    if ((Is-Bad $win) -or (Is-Bad $wsl)) { Write-Host 'NG: clock offset too large'; exit 1 }
    if (Is-Slewing $tick) { Write-Host 'NG: WSL clock is being pulled toward Windows'; exit 1 }
    Write-Host 'OK'; exit 0
}
$winFixed = $false

if (Is-Bad $win) {
    if (Test-Admin) { Fix-Windows }
    else {
        Write-Host 'Windows clock is off: requesting administrator rights...'
        $log = Join-Path $env:TEMP 'sync_robot_clock_win.log'
        $argList = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$PSCommandPath`"",
                     '-Robot', $Robot, '-FixWindowsLog', "`"$log`"")
        Start-Process powershell.exe -Verb RunAs -WindowStyle Hidden -Wait -ArgumentList $argList
    }
    $win = Get-WindowsOffset
    if (Is-Bad $win) {
        # WSLの時計はWindowsとの差が数秒を超えるとhv_utilsがWindowsへ合わせ直す (遅れは
        # 一気に、先行はtickを下げて約0.09秒/秒で)。Windowsがずれたまま WSL だけ ntpdate
        # すると、実行中に元のずれへ戻っていく (2026-10-30 試行2)。なのでWSLには触らない。
        Write-Host 'after:'; Show 'Windows' $win
        Write-Host 'NG: Windows clock could not be fixed; WSL is left untouched (it follows Windows)'
        exit 1
    }
    $winFixed = $true
}
# WSLはWindowsを直してから測り直す (数秒以下の差は自動では直らないのでここで合わせる)。
# Windowsを動かした直後は、WSLが追従し終える (または追従不要と分かる) まで待つ。
$quiet = 0
if ($winFixed) { $quiet = 30 }
if (-not (Wait-WslSettled $quiet)) { Write-Host 'NG: WSL clock did not settle'; exit 1 }
$wsl = Get-WslOffset
if (Is-Bad $wsl) { Fix-Wsl }

$win = Get-WindowsOffset
$wsl = Get-WslOffset
Write-Host 'after:'; Show 'Windows' $win; Show 'WSL' $wsl
if ((Is-Bad $win) -or (Is-Bad $wsl)) { Write-Host 'NG: still off'; exit 1 }
Write-Host 'OK'
