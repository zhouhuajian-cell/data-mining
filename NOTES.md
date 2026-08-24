# 项目笔记（NOTES.md）

数据挖掘项目常用说明。

## Git 推送

在项目目录 `C:\Users\zhouhuajian\Desktop\数据挖掘` 下执行：

```powershell
git add -A
git commit -m "提交说明"
git push
```

### 提交身份
- 用户名：`zhouhuajian-cell`
- 邮箱：`Lmj753753@gmail.com`

### 远程仓库
- 名称：`origin`
- 地址：`https://github.com/zhouhuajian-cell/Maxi--.git`
- 分支：`main`

## .gitignore 排除项

以下内容**不会**被推送到 GitHub（符合「只推代码」的要求）：

| 类别 | 规则 |
|------|------|
| 虚拟环境 | `.venv/` |
| 模型文件 | `*.pt` `*.pth` `*.onnx` `*.part` |
| 数据目录 | `workspace/`（images、features.npy、index.faiss、metadata.json） |
| 有效图片 | `有效/` |
| 临时文件 | `_app_out.txt` `_check.txt` |

> 注意：GitHub 单文件上限 100MB，模型文件（如 `yolov8x.pt`）推不上去，也无需推送。

## 检查仓库内容

```powershell
git ls-files      # 列出已跟踪的文件
git status        # 查看当前状态
git log --oneline # 查看提交历史
```
