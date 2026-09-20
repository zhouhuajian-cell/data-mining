# -*- coding: utf-8 -*-
"""自动遍历新增数据守护（24h，systemd timer 每 5 分钟触发；flock 防重叠）。

流程：读配置 → 遍历各源目录收集视频 → 与清单做差集 → 新视频按目录分组提交抽帧
（/api/process_video，抽帧内自动流水线向量化）→ 管道空闲时自动帧检测 + 段级接地判定。

数据去重约定（用户 2026-09-19 拍板，见 AGENTS.md）：
- 混合目录（视频+图片）：只处理视频，图片跳过（同内容连拍）；
- 同车 2M/8M 双机位：内容相同，只遍历 2M，8M 跳过；
- 同一台车/同一文件夹只进一次（清单按视频路径去重）；
- 手动优先：已被更密手动抽帧覆盖的目录，自动不插手（帧名=身份，手动 5 抽 1 自动补齐 10 抽 1 空隙）。

闸门（24h 无人值守保命设计）：
- workspace/AUTO_ING_OFF 总开关（touch/rm 免重启）；
- 自适应退避：上一轮遍历 >150s（NAS 忙）本轮跳过；
- 可用内存 <4G 只发现不提交；每轮最多提交 N 个目录（限流）；
- 打标闸门：向量化积压 >100 帧先补向量化；全局无 TAG 在跑；VLM 必须 7B。

状态：workspace/ingest_status.json（最近一轮）+ ingest_manifest.json（已提交视频清单）。
"""
import os, sys, json, time, fcntl, traceback

sys.path.insert(0, "/opt/ad_mining")
os.chdir("/opt/ad_mining")
# ⚠️ 绝不 import app/db_service：那会在守护进程里把 82 万条 metadata + 3.8GB 索引
# 全量加载一遍（实测占 3.1GB 常驻），叠加服务进程 11.8GB + VLM 10.5GB 会逼近 OOM。
# 所有判断改成走 HTTP 接口（服务已经把状态暴露出来）。

WORKSPACE = "/opt/ad_mining/workspace"
CFG_PATH = os.path.join(WORKSPACE, "auto_ingest.json")
MANIFEST_PATH = os.path.join(WORKSPACE, "ingest_manifest.json")
STATUS_PATH = os.path.join(WORKSPACE, "ingest_status.json")
WALKTIME_PATH = os.path.join(WORKSPACE, "ingest_walk_time.json")
JUDGE_STATUS_PATH = os.path.join(WORKSPACE, "clip_judge_status.json")   # 段级判定进度（供任务总览）
JUDGE_STOP_PATH = os.path.join(WORKSPACE, "CLIP_JUDGE_STOP")            # 中止标志（内容=项目名）
OFF_FILE = os.path.join(WORKSPACE, "AUTO_ING_OFF")
API = "http://127.0.0.1:8009"
VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mkv", ".wmv", ".ts", ".flv", ".mpg", ".mpeg")
_last_walk_secs = 0.0   # 最近一次 NAS 目录遍历耗时（秒）—— 自适应退避依据


def log(m):
    print("[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), m), flush=True)


def mem_avail_mb():
    try:
        with open("/proc/meminfo") as f:
            d = dict(l.split(":", 1) for l in f if ":" in l)
        return int(d["MemAvailable"].split()[0]) // 1024
    except Exception:
        return None


def http_post(path, fields, timeout=120):
    import urllib.parse, urllib.request
    data = urllib.parse.urlencode(fields).encode("utf-8")
    with urllib.request.urlopen(API + path, data=data, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def http_get(path, timeout=60):
    import urllib.request
    with urllib.request.urlopen(API + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def urllib_parse_quote(s):
    import urllib.parse
    return urllib.parse.quote(str(s))


def load_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def is_video(fn):
    return fn.lower().endswith(VIDEO_EXTS)


_8M_2M_PAIRS = (("CAM_8M", "CAM_2M"), ("CAM8M", "CAM2M"), ("CAM-8M", "CAM-2M"),
                ("vin_device_8M", "vin_device_2M"), ("8M", "2M"))


def _has_2m_sibling(d, dirs_set):
    """同车 2M/8M 双机位：路径里 8M 写法换成 2M 后若该 2M 目录也存在 → 内容重复，8M 跳过。"""
    for a, b in _8M_2M_PAIRS:
        if a in d:
            return d.replace(a, b) in dirs_set
    return False


def walk_videos(root):
    """收集目录下全部视频（含子目录）。自带计时：耗时写全局 _last_walk_secs（退避依据）。"""
    global _last_walk_secs
    _t0 = time.time()
    out = []
    for r, _d, fs in os.walk(root):
        for f in fs:
            if is_video(f):
                p = os.path.join(r, f)
                out.append((p, r))
    _last_walk_secs = max(_last_walk_secs, time.time() - _t0)
    return out


def in_window(window):
    """tag_window 形如 "20-08"（20:00→次日08:00）。None/空 = 不限窗口。"""
    if not window:
        return True
    try:
        a, b = window.split("-")
        h = time.localtime().tm_hour
        if int(a) <= int(b):
            return int(a) <= h < int(b)
        return h >= int(a) or h < int(b)      # 跨午夜
    except Exception:
        return True


def _running_tag_jobs_any():
    """任意项目有 TAG（检测/段级判定）在跑 —— 全局串行依据（共用 YOLO 模型实例，不可并发）。
    纯 HTTP 查询（不加载 DB/项目数据，见文件头注释）。"""
    try:
        jobs = http_get("/api/pipeline/list?limit=50").get("jobs", [])
        return any(j.get("job_type") == "TAG" and j.get("status") in ("RUNNING", "PENDING")
                   for j in jobs)
    except Exception:
        return True   # 查不到就当忙，宁可不动


def pipeline_busy():
    """抽帧/向量化是否在跑（管道忙闲）。纯 HTTP（见文件头：不 import app）。"""
    try:
        d = http_get("/api/all_tasks_status")
        for v in d.values():
            if isinstance(v, dict) and v.get("is_running") and                     v.get("task_type") in ("video", "extract", "vectorize"):
                return True
    except Exception:
        return True   # 查不到就当忙
    return False


def _dir_manual_denser(project, d, auto_step):
    """目录是否已被【更密的手动抽帧】覆盖（历史任务 payload step < 自动 step）。
    手动优先：更密的目录由手动管理增量（5 抽 1 自动补齐 10 抽 1 的空隙），自动不再插手。"""
    segs = [x for x in d.replace("\\", "/").split("/") if x]
    if len(segs) < 2:
        return False
    key = "%" + "%".join(segs[-2:]) + "%"
    try:
        from models import Job, JobType, Project
        db = D.get_db_session()
        try:
            proj = db.query(Project).filter(Project.name == project).first()
            if proj is None:
                return False
            rows = db.query(Job.payload).filter(
                Job.project_id == proj.id,
                Job.job_type == JobType.EXTRACT,
                Job.payload.like(key)).limit(20).all()
        finally:
            db.close()
        for (payload,) in rows:
            try:
                if int((json.loads(payload or "{}")).get("step") or 999) < auto_step:
                    return True
            except Exception:
                continue
    except Exception:
        return False
    return False


def main():
    status = {"last_run": time.strftime("%Y-%m-%d %H:%M:%S"), "roots": [], "new_videos": 0,
              "submitted_dirs": 0, "tag_submitted": None, "errors": []}
    if os.path.exists(OFF_FILE):
        log("[自动遍历] 总开关关闭（AUTO_ING_OFF 存在），跳过")
        status["errors"].append("开关关闭")
        save_json(STATUS_PATH, status)
        return
    cfg = load_json(CFG_PATH, [])
    if not cfg:
        log("[自动遍历] 无配置（%s 不存在或为空），跳过" % CFG_PATH)
        save_json(STATUS_PATH, status)
        return
    manifest = load_json(MANIFEST_PATH, {"videos": {}})
    videos_m = manifest.setdefault("videos", {})
    first_run = len(videos_m) == 0
    manifest_dirty = False
    if first_run:
        log("[自动遍历] 首次运行：只登记存量视频，不提交（防历史全量重抽）")

    if _last_walk_secs > 150:   # 上一轮遍历超 150s = NAS 忙 → 退避（自适应，空闲自动恢复）
        log("[自动遍历] 上一轮遍历耗时 %.0fs（NAS 忙），本轮退避跳过" % _last_walk_secs)
        status["errors"].append("退避：上轮遍历 %.0fs" % _last_walk_secs)
        save_json(STATUS_PATH, status)
        return

    avail = mem_avail_mb()
    can_submit = (avail is None) or (avail >= 4096)
    if not can_submit:
        log("[自动遍历] 可用内存 %sMB < 4096，本轮只发现不提交" % avail)

    submitted_any = False
    for ent in cfg:
        if not ent.get("enabled", True):
            continue
        root, project = ent.get("path"), ent.get("project")
        if not project:
            continue
        st = {"path": root or "(仅检测/判定，不遍历抽帧)", "project": project,
              "found": 0, "new": 0, "submitted_dirs": 0}
        try:
            if root and os.path.isdir(root):
                vids = walk_videos(root)
                st["found"] = len(vids)
                # 🚚 同车 2M/8M 双机位（用户约定）：内容一样，只遍历 2M，8M 跳过
                _dirs = set(d for _p, d in vids)
                _kept, _skip8 = [], []
                for vp, d in vids:
                    if "8M" in d and _has_2m_sibling(d, _dirs):
                        _skip8.append((vp, d))
                    else:
                        _kept.append((vp, d))
                if _skip8:
                    log("[自动遍历] %s: 跳过 8M 目录视频 %d 个（同车 2M 已覆盖，内容相同）"
                        % (project, len(_skip8)))
                    ts8 = time.strftime("%Y-%m-%d %H:%M:%S")
                    for vp, _d in _skip8:
                        videos_m[vp] = {"ts": ts8, "job": "skip_8m"}
                    manifest_dirty = True
                new_by_dir = {}
                _auto_step = int(ent.get("step", 5))
                for vp, d in _kept:
                    if vp in videos_m:
                        continue
                    # 手动优先：已被更密手动抽帧覆盖的目录，自动不插手
                    if _dir_manual_denser(project, d, _auto_step):
                        videos_m[vp] = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "job": "manual_dir"}
                        manifest_dirty = True
                        continue
                    new_by_dir.setdefault(d, []).append(vp)
                st["new"] = sum(len(v) for v in new_by_dir.values())
                if first_run:
                    for vp, _d in _kept:
                        videos_m[vp] = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "job": "seed"}
                    manifest_dirty = True
                    log("[自动遍历] %s: 存量登记 %d 个视频（另跳过 8M %d 个，不提交）"
                        % (project, len(_kept), len(_skip8)))
                elif new_by_dir:
                    max_jobs = int(ent.get("max_jobs_per_cycle", 3))
                    submitted = 0
                    for d, vs in sorted(new_by_dir.items()):
                        if submitted >= max_jobs or not can_submit:
                            break
                        try:
                            r = http_post("/api/process_video", {
                                "project": project, "video_path": d,
                                "frame_step": ent.get("step", 5),
                                "unit": ent.get("unit", "count"),
                                "mode": ent.get("mode", "interval"),
                            })
                            submitted += 1
                            submitted_any = True
                            st["submitted_dirs"] = submitted
                            ts = time.strftime("%Y-%m-%d %H:%M:%S")
                            for vp in vs:
                                videos_m[vp] = {"ts": ts, "job": r.get("job_id")}
                            manifest_dirty = True
                            log("[自动遍历] 提交目录 %s → %s（新视频 %d 个）→ %s"
                                % (d, project, len(vs), json.dumps(r, ensure_ascii=False)[:140]))
                        except Exception as e:
                            status["errors"].append("提交失败 %s: %s" % (d, e))
                            log("[自动遍历] 提交失败 %s: %s" % (d, e))
                    if st["new"] and submitted < len(new_by_dir):
                        log("[自动遍历] %s: 本轮还有 %d 个目录待提交（限流/内存闸门），下轮继续"
                            % (project, len(new_by_dir) - submitted))
            # ---- 灵活打标（用户拍板）：管道空闲就打，忙就让路 ----
            # 闸门顺序：auto_tag → 管道空闲 → 向量化积压 → (可选)窗口 → 内存 → 无 TAG 在跑 → 7B 就绪
            _pend = 0
            try:
                _st = http_get("/api/db_stats?project=%s" % urllib_parse_quote(project), timeout=120)
                _pend = (_st or {}).get("pending_count") or 0
            except Exception:
                pass
            if ent.get("auto_tag") and _pend > 100 and not pipeline_busy() and can_submit:
                try:
                    http_post("/api/auto_vec/run", {"project": project}, timeout=120)
                    log("[自动打标] %s 向量化积压 %d 帧，先补向量化（本轮不打标）" % (project, _pend))
                except Exception as e:
                    log("[自动打标] 补向量化失败 %s: %s" % (project, e))
            if ent.get("auto_tag") and not pipeline_busy() and in_window(ent.get("tag_window")) \
                    and can_submit and _pend <= 100:
                try:
                    # 全局串行：任何项目的 TAG（检测/判定）在跑都等
                    alljobs = http_get("/api/pipeline/list?limit=50").get("jobs", [])
                    if any(j.get("job_type") == "TAG" and j.get("status") in ("RUNNING", "PENDING")
                           for j in alljobs):
                        log("[自动打标] 有 TAG 任务在跑（全局串行），%s 本轮跳过" % project)
                    else:
                        vlm = http_get("/api/vlm_status?warm=1")   # 主动触发 7B 加载；失败/非 7B 才跳过
                        if vlm.get("required_7b") and not vlm.get("is_7b"):
                            log("[自动打标] %s 跳过：VLM 非 7B（%s）" % (project, vlm.get("model")))
                        else:
                            # ① 帧级=检测：先把没有检测记录的帧检测掉
                            r = http_post("/api/pipeline/run_ai_batch", {
                                "project": project, "only_unprocessed": "1",
                                "preset": "balanced", "detect_first": "1",
                                "dino_enhance": "0", "clip_size": "30",
                            }, timeout=600)
                            pending = (r or {}).get("pending") or 0
                            if pending > 0:
                                log("[自动打标] %s 提交帧检测任务（%d 帧待检测），本轮结束；"
                                    "检测完成后下轮进入段级判定" % (project, pending))
                            else:
                                # ② 全部帧已有检测 → 组 Clip（幂等）→ 逐段接地 VLM 判定
                                scan = http_post("/api/clips/scan", {"project": project, "clip_size": 30}, timeout=600)
                                log("[自动打标] %s clips/scan: %s" % (project, json.dumps(scan, ensure_ascii=False)[:120]))
                                judged, page = 0, 1
                                budget = time.time() + 45 * 60
                                _tot = int((scan or {}).get("count") or 0)
                                # 进度写轻量状态文件：任务总览每 4 秒轮询一次，
                                # 直接读 136MB 的 clips.json 会拖垮服务（缓存也救不了 —— 判定期间每段都改它）
                                def _wr_judge(done_, total_, running_=True):
                                    try:
                                        save_json(JUDGE_STATUS_PATH, {
                                            "project": project, "running": running_,
                                            "done": int(done_), "total": int(total_),
                                            "ts": time.time()})
                                    except Exception:
                                        pass
                                _prog = {}
                                try:
                                    _prog = http_get("/api/clips/judge_progress?project=%s"
                                                     % urllib_parse_quote(project), timeout=60) or {}
                                except Exception:
                                    _prog = {}
                                _done0 = int(_prog.get("done") or 0)
                                _tot = int(_prog.get("total") or _tot or 0)
                                _wr_judge(_done0, _tot if _tot else 1)
                                while time.time() < budget:
                                    if os.path.exists(JUDGE_STOP_PATH):
                                        try:
                                            with open(JUDGE_STOP_PATH, encoding="utf-8") as _f:
                                                _sp = (_f.read() or "").strip()
                                        except Exception:
                                            _sp = project
                                        if (not _sp) or _sp == project:
                                            log("[自动打标] %s 收到中止信号，停止判定" % project)
                                            break
                                    if pipeline_busy():
                                        log("[自动打标] 管道变忙（新数据到达），本轮判定让路")
                                        break
                                    d = http_get("/api/clips/list?project=%s&page=%d&size=100&light=1"
                                                 % (urllib_parse_quote(project), page), timeout=120)
                                    items = (d or {}).get("items") or []
                                    if not items:
                                        break
                                    todo = [c for c in items if not c.get("decision")]
                                    for c in todo:
                                        if time.time() > budget or pipeline_busy():
                                            break
                                        try:
                                            _rr = http_post("/api/clips/analyze_single",
                                                            {"project": project, "clip_id": c["clip_id"]},
                                                            timeout=900)
                                            # HTTP 200 但业务 code 可能是 500（判定失败）—— 必须查 code
                                            if (_rr or {}).get("code") == 200:
                                                judged += 1
                                                _wr_judge(_done0 + judged, _tot if _tot else max(judged, 1))
                                            else:
                                                log("[自动打标] 段判定未成功 %s: %s"
                                                    % (c["clip_id"][:40], json.dumps(_rr, ensure_ascii=False)[:120]))
                                        except Exception as e:
                                            log("[自动打标] 段判定失败 %s: %s" % (c["clip_id"][:40], e))
                                    if len(items) < 100:
                                        break
                                    page += 1
                                try:
                                    save_json(JUDGE_STATUS_PATH, {
                                        "project": project, "running": False,
                                        "done": _done0 + judged, "total": _tot if _tot else judged,
                                        "ts": time.time()})
                                except Exception:
                                    pass
                                status["tag_submitted"] = {"project": project, "judged": judged}
                                log("[自动打标] %s 本轮段级判定 %d 段" % (project, judged))
                except Exception as e:
                    status["errors"].append("打标提交失败 %s: %s" % (project, e))
                    log("[自动打标] 提交失败 %s: %s" % (project, e))
        except Exception as e:
            status["errors"].append("%s: %s" % (root, e))
            log("[自动遍历] %s 处理异常: %s" % (root, e))
            log(traceback.format_exc()[:500])
        status["roots"].append(st)

    if manifest_dirty:
        save_json(MANIFEST_PATH, manifest)
    status["new_videos"] = sum(r.get("new", 0) for r in status["roots"])
    status["submitted_dirs"] = sum(r.get("submitted_dirs", 0) for r in status["roots"])
    save_json(WALKTIME_PATH, {"secs": round(_last_walk_secs, 1)})
    save_json(STATUS_PATH, status)
    log("[自动遍历] 本轮完成：新视频 %d，提交目录 %d，错误 %d"
        % (status["new_videos"], status["submitted_dirs"], len(status["errors"])))


if __name__ == "__main__":
    import fcntl
    _lf = open("/opt/ad_mining/workspace/auto_ingest.lock", "w")
    try:
        fcntl.flock(_lf, fcntl.LOCK_EX | fcntl.LOCK_NB)   # 防 timer 触发时上一轮未结束而重叠
    except OSError:
        print("[自动遍历] 上一轮还在运行，本轮跳过", flush=True)
        sys.exit(0)
    main()
