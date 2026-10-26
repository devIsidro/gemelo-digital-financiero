@echo off
REM Sube los cambios del proyecto a la rama develop con un solo comando.
REM Uso:  publicar "mensaje del cambio"
REM       (si no pones el mensaje, te lo pide)
setlocal
chcp 65001 >nul
cd /d "%~dp0"

for /f "delims=" %%b in ('git rev-parse --abbrev-ref HEAD') do set "RAMA=%%b"
if /i not "%RAMA%"=="develop" (
    echo [X] Estas en la rama %RAMA%. Cambiate a develop con: git checkout develop
    exit /b 1
)

echo == 1/4 Trayendo lo ultimo de GitHub ==
git pull --no-edit origin develop || goto :error
git pull --no-edit origin main || goto :error

set "HAY="
for /f "delims=" %%l in ('git status --porcelain') do set "HAY=1"
if not defined HAY (
    echo.
    echo No hay cambios por subir.
    exit /b 0
)

echo.
echo == 2/4 Cambios que se van a subir ==
git status --short
echo.

set "MSG=%~1"
if "%MSG%"=="" set /p "MSG=Escribe el mensaje del cambio: "
if "%MSG%"=="" (
    echo [X] Falta el mensaje del cambio.
    exit /b 1
)

choice /m "Subir estos cambios a develop"
if errorlevel 2 (
    echo Cancelado. No se subio nada.
    exit /b 0
)

echo.
echo == 3/4 Guardando el cambio ==
git add -A || goto :error
git commit -m "%MSG%" || goto :error

echo.
echo == 4/4 Subiendo a GitHub ==
git push origin develop || goto :error

echo.
echo [OK] Listo. GitHub ya esta corriendo el lint y las pruebas:
echo https://github.com/devIsidro/gemelo-digital-financiero/actions
exit /b 0

:error
echo.
echo [X] Algo fallo. Revisa el mensaje de arriba antes de volver a intentarlo.
exit /b 1
