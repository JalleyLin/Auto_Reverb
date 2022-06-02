@echo off

echo Beyond Reaper Portable

setlocal EnableDelayedExpansion

net session > nul 2>&1
if %errorLevel% == 0 (
goto START_POINT
)
%SYSTEMROOT%\system32\net session > nul 2>&1
if %errorLevel% == 0 (
goto START_POINT
) else (
call :exit false false
)


:START_POINT

if "%~dp0"=="C:\BeyondReaper\" goto linkdone
rd C:\BeyondReaper
rd /q /s C:\BeyondReaper

mklink /j C:\BeyondReaper "%~dp0"
explorer C:\BeyondReaper


:linkdone

echo Done!

:exit
pause