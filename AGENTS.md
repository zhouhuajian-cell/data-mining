# 项目协作约定

## 工作方式（重要）
- **改代码一律先在这个本地目录改**，改完在本地验证（语法、接口、单测）。
- **不要自行部署到服务器。** 只有用户明确说"部署/上线"时才部署。
- 部署前先备份服务器上的被覆盖文件。

## 代码风格
- 代码要紧凑，不要堆空行。`app.py` 是 `make_compact.py` 压过的风格：顶层 `def` / `@app.` 之间不空行，全文件不出现连续两行空行。
- 前端以 `front.html` 为准，后端以 `app.py` 为准（app.py 首页路由直接读 `front.html`，无需再同步副本）。

## 部署目标（用户明确要求时才执行）
- 服务器 `10.2.248.34`（root 可密钥登录：`~/.ssh/id_ed25519`；另有 algo 账号，密码见私密记录，**不写入仓库**）。
- 部署目录 `/opt/ad_mining`，服务 `ad_mining.service`（`/venv/bin/uvicorn app:app --port 8009`，`WorkingDirectory=/opt/ad_mining`）。
- 重启：`systemctl restart ad_mining`（需 root）。
- 数据根 `/mnt/Data_Platform/zhj_datamining/test`（NAS）。

## 部署踩过的坑（务必注意）
1. **服务器 Python 是 3.8.10，本地是 3.14。** 上传前必须用服务器解释器校验：
   `/venv/bin/python -m py_compile <file>`，否则可能语法不兼容。
2. **服务器 `gpu_patch.py` 比本地新**（有 ultralytics "Both events must be recorded" 计时 bug 的三次重试逻辑），不要覆盖。`db_service.py`、`gpu_manager.py` 也留在服务器。
3. **NAS 是 CIFS 挂载（账号 `ningbo_new`），能建文件、不能删文件**：服务器侧连自己刚建的文件 `rm` 都 Permission denied。清理类操作不要指望服务器侧完成。
4. 本地 Windows 与服务器之间约有 **8 分钟时钟偏差**，看日志时间戳注意。
5. 另有一台 `192.168.20.206`（NAS 本体），SSH 公钥登录被拒。
