<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AD Scene Mining Engine</title>
    <style>
        /* Apple-style CSS Variables */
        :root {
            --bg-color: #f5f5f7;
            --surface-color: #ffffff;
            --primary-color: #0071e3;
            --primary-hover: #0077ed;
            --text-main: #1d1d1f;
            --text-secondary: #86868b;
            --border-color: #d2d2d7;
            --danger-color: #ff3b30;
            --success-color: #34c759;
            --warning-color: #ff9f0a;
            --shadow-light: 0 4px 24px rgba(0, 0, 0, 0.04);
            --shadow-hover: 0 10px 40px rgba(0, 0, 0, 0.08);
            --radius-lg: 18px;
            --radius-md: 12px;
            --radius-pill: 98px;
            --transition-smooth: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        }

        /* Base & Typography */
        body { font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Arial, sans-serif; margin: 0; background: var(--bg-color); color: var(--text-main); -webkit-font-smoothing: antialiased; }
        
        /* Glassmorphism Header */
        header { background: rgba(255, 255, 255, 0.72); backdrop-filter: saturate(180%) blur(20px); -webkit-backdrop-filter: saturate(180%) blur(20px); position: sticky; top: 0; z-index: 100; padding: 14px 32px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(0,0,0,0.05); }
        .header-title { font-weight: 600; font-size: 20px; letter-spacing: -0.5px; }
        .header-actions { display: flex; gap: 12px; }

        /* Container & Panels */
        .container { padding: 32px; max-width: 1440px; margin: 0 auto; display: flex; flex-direction: column; gap: 24px; }
        .panel { background: var(--surface-color); border-radius: var(--radius-lg); box-shadow: var(--shadow-light); padding: 28px; }
        .panel-header { font-size: 22px; font-weight: 600; margin-bottom: 20px; letter-spacing: -0.3px; display: flex; align-items: center; gap: 8px; }
        
        /* UI Elements */
        .control-group { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 16px; }
        input[type="text"], input[type="number"] { padding: 10px 16px; border: 1px solid var(--border-color); border-radius: var(--radius-md); outline: none; background: #f5f5f7; font-size: 14px; transition: var(--transition-smooth); }
        input[type="text"] { flex: 1; min-width: 280px; }
        input[type="text"]:focus, input[type="number"]:focus { border-color: var(--primary-color); background: #fff; box-shadow: 0 0 0 3px rgba(0, 113, 227, 0.2); }
        input[type="file"] { font-size: 14px; color: var(--text-secondary); }
        
        /* Buttons (Pill shape, scale on hover) */
        button { background: var(--primary-color); color: white; border: none; border-radius: var(--radius-pill); padding: 10px 20px; font-size: 14px; font-weight: 500; cursor: pointer; transition: var(--transition-smooth); display: inline-flex; align-items: center; justify-content: center; gap: 6px; }
        button:hover { transform: scale(1.03); background: var(--primary-hover); box-shadow: 0 4px 12px rgba(0, 113, 227, 0.3); }
        button:active { transform: scale(0.97); }
        button.secondary { background: #e5e5ea; color: var(--text-main); }
        button.secondary:hover { background: #d1d1d6; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); }
        button.yolo-btn { background: var(--success-color); }
        button.yolo-btn:hover { background: #2ebd4e; box-shadow: 0 4px 12px rgba(52, 199, 89, 0.3); }
        button.danger-btn { background: var(--danger-color); }
        button.danger-btn:hover { background: #e0352b; box-shadow: 0 4px 12px rgba(255, 59, 48, 0.3); }
        button:disabled { opacity: 0.5; cursor: not-allowed; transform: none !important; box-shadow: none !important; }

        /* Dashboard Widgets (iOS style) */
        .widget-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-top: 12px; }
        .widget { background: #fbfbfd; border-radius: var(--radius-lg); padding: 20px; text-align: center; border: 1px solid #e5e5ea; transition: var(--transition-smooth); }
        .widget:hover { transform: translateY(-2px); box-shadow: var(--shadow-light); }
        .widget-title { color: var(--text-secondary); font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
        .widget-value { font-size: 28px; font-weight: 700; color: var(--text-main); margin-top: 8px; }
        
        /* Sliders */
        .slider-container { display: flex; align-items: center; gap: 16px; font-size: 14px; background: #fbfbfd; padding: 12px 20px; border-radius: var(--radius-md); border: 1px solid #e5e5ea; width: 100%; box-sizing: border-box; }
        input[type=range] { -webkit-appearance: none; flex: 1; height: 6px; background: #e5e5ea; border-radius: 3px; outline: none; }
        input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 20px; height: 20px; border-radius: 50%; background: #fff; box-shadow: 0 2px 6px rgba(0,0,0,0.2); cursor: pointer; border: 1px solid #e5e5ea; transition: transform 0.1s; }
        input[type=range]::-webkit-slider-thumb:active { transform: scale(1.2); }

        /* Progress Bar */
        .progress-container { width: 100%; background-color: #e5e5ea; border-radius: var(--radius-pill); margin-top: 16px; display: none; height: 8px; overflow: hidden; }
        .progress-bar { height: 100%; background-color: var(--primary-color); width: 0%; transition: width 0.4s ease; border-radius: var(--radius-pill); }

        /* Gallery & Cards */
        .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 24px; margin-top: 24px; }
        .card { border-radius: var(--radius-md); overflow: hidden; background: var(--surface-color); border: 1px solid #e5e5ea; transition: var(--transition-smooth); display: flex; flex-direction: column; }
        .card:hover { transform: translateY(-6px); box-shadow: var(--shadow-hover); border-color: transparent; }
        .img-wrapper { position: relative; width: 100%; aspect-ratio: 16/9; background: #000; overflow: hidden; }
        .img-wrapper img { width: 100%; height: 100%; object-fit: cover; cursor: zoom-in; transition: transform 0.5s ease; }
        .img-wrapper:hover img { transform: scale(1.05); }
        .box-overlay { position: absolute; border: 2px solid var(--success-color); background: rgba(52, 199, 89, 0.15); pointer-events: none; border-radius: 4px; box-shadow: 0 0 0 1px rgba(0,0,0,0.1); }
        .box-label { position: absolute; top: -22px; left: -2px; background: var(--success-color); color: white; font-size: 11px; padding: 2px 6px; border-radius: 4px; white-space: nowrap; font-weight: bold; box-shadow: 0 2px 4px rgba(0,0,0,0.2); }
        
        .card-info { padding: 16px; flex: 1; display: flex; flex-direction: column; }
        .card-header-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
        .card-meta { display: flex; align-items: center; gap: 8px; }
        .select-checkbox { transform: scale(1.2); cursor: pointer; accent-color: var(--primary-color); margin: 0; }
        .badge { background: #f5f5f7; color: var(--text-secondary); padding: 4px 10px; border-radius: var(--radius-pill); font-size: 12px; font-weight: 600; }
        .filename-text { color: var(--text-secondary); font-size: 12px; margin-top: auto; word-break: break-all; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        
        /* Tags */
        .tags-container { display: flex; flex-wrap: wrap; gap: 6px; margin: 12px 0; }
        .tag-item { background: rgba(0, 113, 227, 0.1); color: var(--primary-color); padding: 4px 10px; border-radius: var(--radius-pill); font-size: 12px; font-weight: 500; display: flex; align-items: center; }
        .tag-remove { margin-left: 6px; cursor: pointer; font-weight: bold; opacity: 0.6; transition: opacity 0.2s; }
        .tag-remove:hover { opacity: 1; color: var(--danger-color); }
        .tag-add-btn { background: #f5f5f7; padding: 4px 10px; border-radius: var(--radius-pill); font-size: 12px; cursor: pointer; color: var(--text-secondary); font-weight: 500; transition: var(--transition-smooth); }
        .tag-add-btn:hover { background: #e5e5ea; color: var(--text-main); }

        /* Elegant Modal Animations */
        @keyframes fadeIn { from { opacity: 0; backdrop-filter: blur(0px); } to { opacity: 1; backdrop-filter: blur(10px); } }
        @keyframes scaleUp { from { transform: scale(0.9); opacity: 0; } to { transform: scale(1); opacity: 1; } }
        .modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.6); align-items: center; justify-content: center; }
        .modal.active { display: flex; animation: fadeIn 0.3s ease forwards; }
        .modal-content { position: relative; display: inline-block; max-width: 90vw; max-height: 90vh; animation: scaleUp 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards; box-shadow: 0 24px 48px rgba(0,0,0,0.2); border-radius: var(--radius-md); overflow: hidden; background: #000; }
        .modal-content img { display: block; max-width: 90vw; max-height: 90vh; object-fit: contain; }
        .close-btn { position: absolute; top: 16px; right: 16px; width: 36px; height: 36px; background: rgba(255,255,255,0.2); backdrop-filter: blur(4px); color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 24px; cursor: pointer; z-index: 1001; transition: background 0.2s; }
        .close-btn:hover { background: rgba(255,255,255,0.4); }

        /* Stats & Pagination Bar */
        .stats-bar { font-size: 14px; font-weight: 500; color: var(--text-main); background: #fbfbfd; padding: 16px 20px; border-radius: var(--radius-md); display: flex; align-items: center; justify-content: space-between; border: 1px solid #e5e5ea; margin-bottom: 16px; flex-wrap: wrap; gap: 12px; }
        .stats-number { font-size: 16px; font-weight: 700; padding: 0 2px; }
        
        .pagination-bar { display: none; background: #fff; padding: 12px 20px; border-radius: var(--radius-md); margin-bottom: 16px; align-items: center; justify-content: space-between; border: 1px solid #e5e5ea; }
        .pagination-controls { display: flex; gap: 8px; align-items: center; }
        .page-btn { background: #f5f5f7; color: var(--text-main); border: none; }
        
        .divider { height: 1px; background: #e5e5ea; margin: 24px 0; width: 100%; }
        .section-label { font-size: 13px; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; display: block; }
    </style>
</head>
<body>
    <header>
        <div class="header-title">🚘 Scene Mining Engine</div>
        <div class="header-actions">
            <button class="secondary" onclick="exportSearchResults()" style="background: rgba(0, 113, 227, 0.1); color: var(--primary-color);">􀈿 仅导出 CLIP 粗筛</button>
            <button onclick="exportData(true)">􀈿 导出勾选 JSON (含预标注)</button>
            <button class="secondary" onclick="exportData(false)">􀈿 导出未勾选</button>
        </div>
    </header>

    <div class="container">
        <!-- 模块 1 -->
        <div class="panel">
            <div class="panel-header">􀤆 物理底库大盘</div>
            <div class="control-group">
                <input type="file" id="fileInput" multiple accept="image/*">
                <button id="uploadBtn" onclick="uploadImagesInBatches()">网页增量入库</button>
                <span id="indexStatus" style="color: var(--text-secondary); font-size: 13px;">(海量初始化请使用离线脚本)</span>
            </div>
            <div id="progressContainer" class="progress-container">
                <div id="progressBar" class="progress-bar"></div>
            </div>
            
            <div class="widget-grid">
                <div class="widget">
                    <div class="widget-title">📁 硬盘原图总数</div>
                    <div id="dashRaw" class="widget-value">-</div>
                </div>
                <div class="widget">
                    <div class="widget-title" style="color: var(--success-color);">✅ 已向量化 (可检索)</div>
                    <div id="dashIndexed" class="widget-value" style="color: var(--success-color);">-</div>
                </div>
                <div class="widget">
                    <div class="widget-title" style="color: var(--warning-color);">⚠️ 待处理差额</div>
                    <div id="dashUnprocessed" class="widget-value" style="color: var(--warning-color);">-</div>
                </div>
            </div>
        </div>

        <!-- 模块 2 -->
        <div class="panel">
            <div class="panel-header">􀈕 冗余分析与降重</div>
            <div class="control-group">
                <span style="font-size: 14px; font-weight: 500;">特征相似度阈值:</span>
                <input type="number" id="dedupThreshold" value="0.95" step="0.01" min="0.5" max="1.0" style="width: 80px;">
                <button onclick="runDeduplicationAnalysis()" style="background: var(--text-main);">􀙬 特征扫描</button>
                <button id="exportRedundantBtn" class="secondary" onclick="exportRedundantList()" style="display: none;">􀈿 导出清单</button>
                <button id="deleteAndSyncBtn" class="danger-btn" onclick="deleteAndSync()" style="display: none;">􀈑 永久粉碎冗余并同步</button>
                <span id="dedupStatus" style="font-size: 14px; font-weight: 500; margin-left: 12px; color: var(--primary-color);"></span>
            </div>
            <div id="statsSummary" class="widget-grid" style="display: none;">
                <div class="widget"><div class="widget-title">底库总帧数</div><div id="statTotal" class="widget-value">0</div></div>
                <div class="widget"><div class="widget-title" style="color: var(--success-color);">唯一有效帧</div><div id="statUnique" class="widget-value" style="color: var(--success-color);">0</div></div>
                <div class="widget"><div class="widget-title" style="color: var(--danger-color);">冗余重复帧</div><div id="statDuplicates" class="widget-value" style="color: var(--danger-color);">0</div></div>
                <div class="widget"><div class="widget-title" style="color: var(--warning-color);">整体冗余率</div><div id="statRate" class="widget-value" style="color: var(--warning-color);">0%</div></div>
            </div>
        </div>

        <!-- 模块 3 -->
        <div class="panel">
            <div class="panel-header">􀊫 语义挖掘与预标注</div>
            
            <span class="section-label">Phase 1: 跨模态粗筛 (Retrieval)</span>
            <div class="control-group">
                <input type="text" id="queryInput" placeholder="输入自然语言场景，例如: overturned vehicle on the road...">
                <span style="font-size: 14px;">Top K:</span>
                <input type="number" id="topKInput" value="500" style="width: 80px;">
                <button onclick="searchScenes()">􀊫 文本搜图</button>
                <button class="secondary" onclick="startBrowseAll()">􀏚 全库浏览</button>
            </div>
            
            <div class="control-group">
                <input type="text" id="filenameInput" placeholder="精确打捞: 输入文件名片段 (如: 1781259270)">
                <button class="secondary" onclick="searchByFilename()">􀊫 文件名检索</button>
                
                <div style="width: 1px; height: 24px; background: var(--border-color); margin: 0 8px;"></div>
                
                <input type="file" id="externalImageInput" accept="image/*">
                <button class="secondary" onclick="searchByExternalImage()">􀎰 以图搜图</button>
            </div>
            
            <div class="divider"></div>
            
            <span class="section-label">Phase 2: 目标定位与预标 (Detection)</span>
            <div class="control-group">
                <input type="text" id="dinoPrompt" placeholder="DINO 开放词汇提示词，例如: traffic cone.">
                <button onclick="runBatchDetection('dino')">􀎡 执行 DINO 批量检测</button>
                <button class="yolo-btn" onclick="runBatchDetection('yolo')">􀎡 执行 YOLOv8x 兜底检测</button>
            </div>
            
            <div id="detectStatus" style="font-size: 14px; font-weight: 500; color: var(--warning-color); margin-bottom: 8px; display: none;">准备启动检测任务...</div>
            <div id="detectProgressContainer" class="progress-container" style="margin-bottom: 20px;">
                <div id="detectProgressBar" class="progress-bar" style="background-color: var(--warning-color);"></div>
            </div>
            
            <div class="control-group" style="align-items: stretch;">
                <div class="slider-container">
                    <span style="width: 160px;">CLIP 匹配度底线: <span id="clipThresholdVal" style="color:var(--primary-color); font-weight:bold;">0.20</span></span>
                    <input type="range" id="clipSlider" min="-1" max="1" step="0.01" value="0.20" oninput="updateClipThreshold(this.value)">
                </div>
                <div class="slider-container" style="background: rgba(52, 199, 89, 0.05); border-color: rgba(52, 199, 89, 0.2);">
                    <span style="width: 160px;">目标置信度要求: <span id="boxThresholdVal" style="color:var(--success-color); font-weight:bold;">0.50</span></span>
                    <input type="range" id="boxSlider" min="0" max="1" step="0.01" value="0.50" oninput="updateBoxThreshold(this.value)">
                </div>
            </div>
            
            <div class="stats-bar">
                <div style="display: flex; gap: 24px; align-items: center; flex-wrap: wrap;">
                    <span>本页展示: <span id="dispCount" class="stats-number" style="color: var(--primary-color);">0</span></span>
                    <span>本页含目标: <span id="targetCount" class="stats-number" style="color: var(--success-color);">0</span></span>
                    <span>全局已勾选: <span id="checkedCount" class="stats-number" style="color: var(--warning-color);">0</span></span>
                    <span>全局未勾选: <span id="uncheckedCount" class="stats-number" style="color: var(--text-secondary);">0</span></span>
                </div>
                <div style="font-size: 12px; color: var(--text-secondary);">(导出 JSON 严格基于全局勾选状态)</div>
            </div>
            
            <!-- 分页 -->
            <div id="paginationBar" class="pagination-bar">
                <div style="font-size: 13px; color: var(--text-secondary);">
                    当前第 <span id="pageCurrent" style="font-weight: 600; color: var(--text-main);">1</span> / <span id="pageTotal" style="font-weight: 600;">0</span> 页 
                    (任务总计 <span id="itemTotal">0</span> 帧)
                </div>
                <div class="pagination-controls">
                    <button id="prevPageBtn" class="page-btn" onclick="changePage(-1)">上一页</button>
                    <input type="number" id="jumpPageInput" min="1" style="width: 60px; text-align: center;" placeholder="页码">
                    <button class="page-btn" onclick="jumpToPage()">跳转</button>
                    <button id="nextPageBtn" class="page-btn" onclick="changePage(1)">下一页</button>
                </div>
            </div>

            <!-- 画廊 -->
            <div id="gallery" class="gallery"></div>
        </div>
    </div>

    <!-- 弹窗 -->
    <div id="imageModal" class="modal" onclick="closeModal(event)">
        <div class="modal-content" id="modalContentWrapper">
            <div class="close-btn" onclick="closeModal(event)">×</div>
            <img id="modalImg" src="">
            <div id="modalBoxContainer"></div>
        </div>
    </div>

    <script>
        const API_BASE = "http://localhost:8008"; 
        let allFrontendData = [];  
        let currentResults = [];   
        let currentMode = 'browse'; 
        let currentPage = 1;
        let totalPages = 0;
        const pageSize = 30; 
        let currentClipThreshold = 0.20; 
        let currentBoxThreshold = 0.50;  
        let redundantFilesList = [];

        async function fetchDbStats() {
            try {
                const res = await fetch(`${API_BASE}/api/db_stats`);
                const data = await res.json();
                document.getElementById('dashRaw').innerText = data.total_raw;
                document.getElementById('dashIndexed').innerText = data.indexed;
                const unprocEl = document.getElementById('dashUnprocessed');
                if (data.unprocessed > 0) {
                    unprocEl.innerText = `${data.unprocessed}`;
                    unprocEl.style.color = "var(--danger-color)";
                } else {
                    unprocEl.innerText = "0";
                    unprocEl.style.color = "var(--warning-color)";
                }
            } catch (e) { console.error("Stats Fetch Error", e); }
        }

        window.onload = function() { fetchDbStats(); startBrowseAll(); };

        function updateSelectionStats() {
            let list = currentMode === 'browse' ? currentResults : allFrontendData;
            if (currentMode === 'search') list = list.filter(item => item.score >= currentClipThreshold);
            let checkedCount = list.filter(item => item._selected).length;
            document.getElementById('checkedCount').innerText = checkedCount;
            document.getElementById('uncheckedCount').innerText = list.length - checkedCount;
        }

        window.toggleItemSelection = function(id, isChecked) {
            let list = currentMode === 'browse' ? currentResults : allFrontendData;
            let item = list.find(i => i.id === id);
            if(item) item._selected = isChecked;
            updateSelectionStats();
        };

        function removeTag(itemIndex, tagText) {
            if(currentResults[itemIndex].userTags) {
                currentResults[itemIndex].userTags = currentResults[itemIndex].userTags.filter(t => t !== tagText);
                renderGallery(); 
            }
        }

        function addTag(itemIndex) {
            const newTag = prompt("输入要添加的标签名称:");
            if(newTag && newTag.trim() !== "") {
                const item = currentResults[itemIndex];
                if(!item.userTags) item.userTags = [];
                if(!item.userTags.includes(newTag.trim())) {
                    item.userTags.push(newTag.trim());
                    item._selected = true; 
                    renderGallery();
                }
            }
        }

        async function uploadImagesInBatches() {
            const files = document.getElementById('fileInput').files;
            if (!files.length) return alert('请先选择图片');
            if (files.length > 30000) return alert('单次任务限制 30000 张。');
            const btn = document.getElementById('uploadBtn');
            const statusSpan = document.getElementById('indexStatus');
            const pContainer = document.getElementById('progressContainer');
            const pBar = document.getElementById('progressBar');
            btn.disabled = true; pContainer.style.display = 'block'; pBar.style.width = '0%'; statusSpan.style.color = 'var(--primary-color)';
            
            let totalIndexed = 0, processedSoFar = 0;
            for (let i = 0; i < files.length; i += 50) {
                const chunk = Array.from(files).slice(i, i + 50);
                const formData = new FormData();
                chunk.forEach(f => formData.append('files', f));
                statusSpan.innerText = `处理中 ${i + 1} / ${files.length}...`;
                try {
                    const res = await fetch(`${API_BASE}/api/upload_batch`, { method: 'POST', body: formData });
                    const data = await res.json();
                    if (!res.ok) { alert(`中断: ${data.error}`); break; }
                    totalIndexed = data.total_indexed; processedSoFar += chunk.length;
                    pBar.style.width = `${Math.round((processedSoFar / files.length) * 100)}%`;
                } catch (e) { alert('网络中断'); break; }
            }
            btn.disabled = false; fetchDbStats(); 
            if (processedSoFar === files.length) { statusSpan.innerText = '✅ 入库成功'; statusSpan.style.color = 'var(--success-color)'; startBrowseAll(); } 
            else { statusSpan.innerText = '⚠️ 任务未完结'; statusSpan.style.color = 'var(--danger-color)'; }
        }

        function openModal(index) {
            const item = currentResults[index];
            document.getElementById('modalImg').src = `${API_BASE}/api/image/${item.id}`;
            const boxContainer = document.getElementById('modalBoxContainer');
            boxContainer.innerHTML = ''; 
            if (item.detections && item.detections.width) {
                const oW = item.detections.width, oH = item.detections.height;
                let validBoxes = item.detections.scores ? item.detections.scores.map((s, i) => ({ score:s, label:item.detections.labels[i], box:item.detections.boxes[i] })).filter(b => b.score >= currentBoxThreshold) : [];
                let html = '';
                validBoxes.forEach(b => {
                    html += `<div class="box-overlay" style="left: ${(b.box[0]/oW)*100}%; top: ${(b.box[1]/oH)*100}%; width: ${((b.box[2]-b.box[0])/oW)*100}%; height: ${((b.box[3]-b.box[1])/oH)*100}%; border-width:3px;"><div class="box-label" style="top:-24px; font-size:14px;">${b.label} ${b.score.toFixed(2)}</div></div>`;
                });
                boxContainer.innerHTML = html;
            }
            document.getElementById('imageModal').classList.add('active');
        }

        function closeModal(e) { if(e.target.id === 'imageModal' || e.target.className === 'close-btn') document.getElementById('imageModal').classList.remove('active'); }

        function handleFrontendData(dataArray, mode) {
            currentMode = mode;
            allFrontendData = dataArray.map(item => ({ ...item, detections: null, userTags: null, _selected: false }));
            updateFrontendPagination(true);
        }

        function updateFrontendPagination(resetPage = false) {
            if (resetPage) currentPage = 1;
            let filtered = currentMode === 'search' ? allFrontendData.filter(i => i.score >= currentClipThreshold) : allFrontendData;
            totalPages = Math.ceil(filtered.length / pageSize) || 1;
            if (currentPage > totalPages) currentPage = totalPages;
            document.getElementById('pageCurrent').innerText = currentPage;
            document.getElementById('pageTotal').innerText = totalPages;
            document.getElementById('itemTotal').innerText = filtered.length;
            document.getElementById('prevPageBtn').disabled = (currentPage === 1);
            document.getElementById('nextPageBtn').disabled = (currentPage === totalPages);
            document.getElementById('paginationBar').style.display = 'flex';
            currentResults = filtered.slice((currentPage - 1) * pageSize, currentPage * pageSize);
            renderGallery();
        }

        function startBrowseAll() { currentMode = 'browse'; currentPage = 1; loadPageData(); }
        function changePage(delta) {
            if (currentPage + delta < 1 || currentPage + delta > totalPages) return;
            currentPage += delta;
            currentMode === 'browse' ? loadPageData() : updateFrontendPagination(false);
        }
        function jumpToPage() {
            const val = parseInt(document.getElementById('jumpPageInput').value);
            if (val >= 1 && val <= totalPages) { currentPage = val; currentMode === 'browse' ? loadPageData() : updateFrontendPagination(false); } 
            else alert('无效页码');
        }

        async function loadPageData() {
            document.body.style.cursor = 'wait'; document.getElementById('paginationBar').style.display = 'flex'; 
            try {
                const res = await fetch(`${API_BASE}/api/list_all?page=${currentPage}&size=${pageSize}`);
                const data = await res.json();
                if(data.total === 0) {
                    document.getElementById('paginationBar').style.display = 'none';
                    document.getElementById('gallery').innerHTML = '<div style="color:var(--text-secondary); width:100%; text-align:center; padding:40px;">数据池为空，请先入库。</div>';
                    document.getElementById('dispCount').innerText = 0; document.getElementById('targetCount').innerText = 0; updateSelectionStats();
                } else {
                    totalPages = data.total_pages; currentPage = data.current_page;
                    document.getElementById('pageCurrent').innerText = currentPage; document.getElementById('pageTotal').innerText = totalPages; document.getElementById('itemTotal').innerText = data.total;
                    document.getElementById('prevPageBtn').disabled = (currentPage === 1); document.getElementById('nextPageBtn').disabled = (currentPage === totalPages);
                    currentResults = data.results.map(item => ({ ...item, detections: null, userTags: null, _selected: false }));
                    renderGallery();
                }
            } catch (e) { console.error(e); }
            document.body.style.cursor = 'default';
        }

        async function searchScenes() {
            const query = document.getElementById('queryInput').value;
            if (!query) return alert('请输入检索词');
            const formData = new FormData(); formData.append('query', query); formData.append('top_k', document.getElementById('topKInput').value);
            document.body.style.cursor = 'wait';
            const res = await fetch(`${API_BASE}/api/search`, { method: 'POST', body: formData });
            handleFrontendData((await res.json()).results, 'search');
            document.body.style.cursor = 'default';
        }
        
        async function searchByFilename() {
            const query = document.getElementById('filenameInput').value.trim();
            if (!query) return alert('请输入关键字');
            const formData = new FormData(); formData.append('filename_query', query);
            document.body.style.cursor = 'wait';
            try {
                const res = await fetch(`${API_BASE}/api/search_by_filename`, { method: 'POST', body: formData });
                const data = await res.json();
                if (data.results.length === 0) alert('未找到匹配文件'); else handleFrontendData(data.results, 'filename');
            } catch (e) { alert("检索失败"); }
            document.body.style.cursor = 'default';
        }

        async function searchByExternalImage() {
            const fileInput = document.getElementById('externalImageInput');
            if (!fileInput.files.length) return alert('请选择图片');
            const formData = new FormData(); formData.append('file', fileInput.files[0]); formData.append('top_k', document.getElementById('topKInput').value || 24);
            document.body.style.cursor = 'wait';
            const res = await fetch(`${API_BASE}/api/search_by_external_image`, { method: 'POST', body: formData });
            handleFrontendData((await res.json()).results, 'search');
            document.body.style.cursor = 'default';
        }

        async function runDeduplicationAnalysis() {
            const statusText = document.getElementById('dedupStatus');
            statusText.innerText = "扫描中..."; statusText.style.color = "var(--primary-color)";
            const formData = new FormData(); formData.append('sim_threshold', document.getElementById('dedupThreshold').value || 0.95);
            redundantFilesList = []; document.getElementById('exportRedundantBtn').style.display = 'none'; document.getElementById('deleteAndSyncBtn').style.display = 'none';
            try {
                const data = await (await fetch(`${API_BASE}/api/dedup_stats`, { method: 'POST', body: formData })).json();
                document.getElementById('statsSummary').style.display = 'grid';
                document.getElementById('statTotal').innerText = data.total_images; document.getElementById('statUnique').innerText = data.unique_images;
                document.getElementById('statDuplicates').innerText = data.duplicate_count; document.getElementById('statRate').innerText = data.dedup_rate;
                statusText.innerText = `完成! 发现 ${data.duplicate_count} 张冗余`; statusText.style.color = "var(--success-color)";
                if (data.clusters.length > 0) {
                    let flat = [];
                    data.clusters.forEach(c => { for(let k=1; k<c.items.length; k++) redundantFilesList.push(c.items[k].id); flat = flat.concat(c.items); });
                    if(redundantFilesList.length > 0) { document.getElementById('exportRedundantBtn').style.display = 'inline-flex'; document.getElementById('deleteAndSyncBtn').style.display = 'inline-flex'; }
                    handleFrontendData(flat, 'dedup');
                }
            } catch(e) { statusText.innerText = "请求失败"; statusText.style.color = "var(--danger-color)"; }
        }
        
        function exportRedundantList() {
            if (redundantFilesList.length === 0) return;
            const a = document.createElement('a');
            a.href = URL.createObjectURL(new Blob([redundantFilesList.join('\n')], { type: 'text/plain;charset=utf-8' }));
            a.download = `AD_Redundant_IDs.txt`; document.body.appendChild(a); a.click(); document.body.removeChild(a);
        }

        async function deleteAndSync() {
            if (!confirm(`⚠️ 警告：这将在硬盘上永久粉碎冗余图片并重组特征库！确定继续吗？`)) return;
            document.body.style.cursor = 'wait';
            const formData = new FormData(); formData.append('image_ids', redundantFilesList.join(','));
            try {
                const data = await (await fetch(`${API_BASE}/api/delete_and_sync`, { method: 'POST', body: formData })).json();
                alert(`清理完成！物理删除: ${data.deleted_count} 帧`);
                document.getElementById('exportRedundantBtn').style.display = 'none'; document.getElementById('deleteAndSyncBtn').style.display = 'none';
                document.getElementById('dedupStatus').innerText = "底库净化完成"; fetchDbStats(); startBrowseAll();
            } catch (e) { alert("清理失败"); }
            document.body.style.cursor = 'default';
        }

        async function runBatchDetection(mode) {
            let targetList = currentMode === 'browse' ? currentResults : allFrontendData;
            if (currentMode === 'search') targetList = allFrontendData.filter(i => i.score >= currentClipThreshold);
            if (targetList.length === 0) return alert('当前没有待检测图片。');
            
            const prompt = document.getElementById('dinoPrompt').value;
            if (mode === 'dino' && !prompt) return alert('请填写 DINO 提示词');
            if (!confirm(`将后台处理 ${targetList.length} 张图片，确认开始？`)) return;

            const endpoint = mode === 'yolo' ? `${API_BASE}/api/yolo_detect` : `${API_BASE}/api/ground_detect`;
            const pContainer = document.getElementById('detectProgressContainer');
            const pBar = document.getElementById('detectProgressBar');
            const status = document.getElementById('detectStatus');
            
            status.style.display = 'block'; pContainer.style.display = 'block'; pBar.style.width = '0%';
            document.body.style.cursor = 'wait';
            
            for (let i = 0; i < targetList.length; i++) {
                const item = targetList[i];
                status.innerText = `检测中: ${i + 1} / ${targetList.length}...`;
                pBar.style.width = `${Math.round(((i + 1) / targetList.length) * 100)}%`;
                
                const fd = new FormData(); fd.append('image_id', item.id); if (mode === 'dino') fd.append('text_prompt', prompt);
                try {
                    const res = await fetch(endpoint, { method: 'POST', body: fd });
                    if (!res.ok) continue;
                    const data = await res.json();
                    item.detections = data;
                    
                    let hasTargets = false;
                    if (data.scores && data.labels) {
                        const counts = {};
                        data.scores.forEach((s, idx) => {
                            if (s >= currentBoxThreshold) { counts[data.labels[idx]] = (counts[data.labels[idx]] || 0) + 1; hasTargets = true; }
                        });
                        item.userTags = Object.keys(counts).map(k => `${k} ${counts[k]}`);
                    }
                    item._selected = hasTargets;
                    updateSelectionStats(); 
                    if (currentResults.find(curr => curr.id === item.id)) renderGallery(); 
                    await new Promise(r => setTimeout(r, 10));
                } catch (e) {}
            }
            
            status.innerText = `✅ 检测完成！生成报告...`; status.style.color = 'var(--success-color)';
            document.body.style.cursor = 'default';
            updateSelectionStats(); 
            
            setTimeout(() => {
                pContainer.style.display = 'none'; status.style.display = 'none'; status.style.color = 'var(--warning-color)';
                let checked = targetList.filter(i => i._selected).length;
                alert(`✅ 批量检测执行完毕\n\n总数：${targetList.length}\n检出目标：${checked}\n未检出：${targetList.length - checked}`);
            }, 300);
        }

        function updateBoxThreshold(val) {
            currentBoxThreshold = parseFloat(val);
            document.getElementById('boxThresholdVal').innerText = currentBoxThreshold.toFixed(2);
            let list = currentMode === 'browse' ? currentResults : allFrontendData;
            list.forEach(item => {
                let h = false;
                if (item.detections && item.detections.scores) if (item.detections.scores.some(s => s >= currentBoxThreshold)) h = true;
                if (item.userTags && item.userTags.length > 0) h = true;
                item._selected = h;
            });
            renderGallery(); 
        }

        function updateClipThreshold(val) {
            currentClipThreshold = parseFloat(val);
            document.getElementById('clipThresholdVal').innerText = currentClipThreshold.toFixed(2);
            currentMode === 'search' ? updateFrontendPagination(true) : renderGallery();
        }

        function renderGallery() {
            const container = document.getElementById('gallery'); container.innerHTML = '';
            let withTarget = 0;
            currentResults.forEach((item, index) => {
                let validBoxes = item.detections && item.detections.scores ? item.detections.scores.map((score, i) => ({ score, label: item.detections.labels[i], box: item.detections.boxes[i] })).filter(b => b.score >= currentBoxThreshold) : [];
                if (validBoxes.length > 0 || (item.userTags && item.userTags.length > 0)) withTarget++;

                const card = document.createElement('div'); card.className = 'card';
                let boxesHTML = '';
                if (item.detections && item.detections.width) {
                    const oW = item.detections.width, oH = item.detections.height;
                    validBoxes.forEach(b => { boxesHTML += `<div class="box-overlay" style="left: ${(b.box[0]/oW)*100}%; top: ${(b.box[1]/oH)*100}%; width: ${((b.box[2]-b.box[0])/oW)*100}%; height: ${((b.box[3]-b.box[1])/oH)*100}%;"><div class="box-label">${b.label} ${b.score.toFixed(2)}</div></div>`; });
                }
                
                let tagsHTML = (item.userTags || []).map(tag => `<div class="tag-item">${tag} <span class="tag-remove" onclick="removeTag(${index}, '${tag}')">×</span></div>`).join('');
                tagsHTML += `<div class="tag-add-btn" onclick="addTag(${index})">+ Add</div>`;

                card.innerHTML = `
                    <div class="img-wrapper">
                        <img src="${API_BASE}/api/image/${item.id}" onclick="openModal(${index})">
                        ${boxesHTML}
                    </div>
                    <div class="card-info">
                        <div class="card-header-row">
                            <div class="card-meta">
                                <input type="checkbox" class="select-checkbox" value="${item.id}" ${item._selected ? 'checked' : ''} onchange="toggleItemSelection(${item.id}, this.checked)">
                                <span class="badge">ID: ${item.id}</span> 
                            </div>
                            ${currentMode === 'search' ? `<span class="badge" style="background:#e0f2fe; color:var(--primary-color);">Match: ${item.score.toFixed(3)}</span>` : ''}
                        </div>
                        <div class="tags-container">${tagsHTML}</div>
                        <div class="filename-text" title="${item.filename}">${item.filename}</div>
                    </div>
                `;
                container.appendChild(card);
            });
            document.getElementById('dispCount').innerText = currentResults.length;
            document.getElementById('targetCount').innerText = withTarget;
            updateSelectionStats(); 
        }

        function exportSearchResults() {
            let list = currentMode === 'browse' ? currentResults : allFrontendData;
            if (currentMode === 'search') list = list.filter(i => i.score >= currentClipThreshold);
            if (list.length === 0) return alert('⚠️ 列表为空！');
            doExport(list, `AD_CLIP_RawSearch_${list.length}_Items`);
        }

        function exportData(exportChecked) {
            let list = currentMode === 'browse' ? currentResults : allFrontendData;
            if (currentMode === 'search') list = list.filter(i => i.score >= currentClipThreshold);
            let targetItems = list.filter(i => exportChecked ? !!i._selected : !i._selected);
            if (targetItems.length === 0) return alert(`⚠️ 任务池中无${exportChecked?'已勾选':'未勾选'}图片！`);
            doExport(targetItems, `${exportChecked ? 'AD_Selected' : 'AD_Unselected'}_Total_${targetItems.length}`);
        }

        function doExport(items, prefix) {
            const payload = items.map(item => {
                let fd = null;
                if (item.detections) {
                    const vi = item.detections.scores.map((s, i) => s >= currentBoxThreshold ? i : -1).filter(i => i !== -1);
                    fd = { boxes: vi.map(i => item.detections.boxes[i]), labels: vi.map(i => item.detections.labels[i]), scores: vi.map(i => item.detections.scores[i]) };
                }
                return { image_id: item.id, filename: item.filename, file_path: `./workspace/images/${item.filename}`, score: currentMode === 'search' ? item.score : 1.0, annotations: fd, final_tags: item.userTags || [] };
            });
            const a = document.createElement('a');
            a.href = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(payload, null, 2));
            a.download = `${prefix}_${Date.now()}.json`; a.click();
        }
    </script>
</body>
</html>