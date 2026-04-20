$ErrorActionPreference = 'SilentlyContinue'
$start = Get-Date '2026-04-19T11:00:00'
$end = Get-Date '2026-04-19T15:30:00'
$events = Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=$start; EndTime=$end} -MaxEvents 500 |
    Where-Object { $_.Id -in 1,42,107,506,507,1074,6005,6006,6008,6009,12,13,109,41 }
$events | Select-Object TimeCreated, Id, ProviderName, @{n='Msg'; e={($_.Message -split "`n")[0]}} |
    Format-Table -AutoSize | Out-String -Width 240
Write-Output "---"
Write-Output "Count of matching power/boot events: $($events.Count)"
Write-Output "---MEMORY PRESSURE EVENTS---"
Get-WinEvent -FilterHashtable @{LogName='Application'; StartTime=$start; EndTime=$end} -MaxEvents 500 |
    Where-Object { $_.Message -match 'low memory|resource|vmmem|pagefile' } |
    Select-Object TimeCreated, Id, ProviderName, @{n='Msg'; e={($_.Message -split "`n")[0]}} |
    Format-Table -AutoSize | Out-String -Width 240
