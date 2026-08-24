@echo off
:: 设置字符集防止终端中文乱码
chcp 65001 >nul

echo =======================================================
echo          自动驾驶场景挖掘引擎 - 后端启动脚本
echo =======================================================
echo.

:: 1. 切换到项目所在目录
cd /d "C:\Users\zhouhuajian\Desktop\数据挖掘"

:: 2. 激活 Python 虚拟环境
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] 正在激活虚拟环境 .venv ...
    call .venv\Scripts\activate.bat
) else (
    echo [ERROR] 找不到虚拟环境 .venv，请确保环境路径正确！
    pause
    exit
)

:: 3. 启动 FastAPI 后端服务
echo [INFO] 正在启动核心引擎服务 (app.py) ...
echo [INFO] 服务启动后，请在浏览器打开 http://localhost:8008
echo.

python app.py

:: 保持窗口不闪退
pause