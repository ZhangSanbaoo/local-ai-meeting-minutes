"""
音频处理 API 路由
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import logging

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

logger = logging.getLogger("meeting_ai.api.routes.process")

from meeting_ai.api.schemas import (
    EnhanceMode,
    JobResponse,
    JobStatus,
    ProcessResultResponse,
    SegmentResponse,
    SegmentUpdateRequest,
    SegmentSplitRequest,
    SpeakerRenameRequest,
    SpeakerResponse,
    SummaryUpdateRequest,
)
from meeting_ai.config import get_settings

router = APIRouter()

# 任务存储（生产环境应使用 Redis）
_jobs: dict[str, dict] = {}


def _get_job(job_id: str) -> dict:
    """获取任务"""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"任务不存在: {job_id}")
    return _jobs[job_id]


@router.post("/process", response_model=JobResponse)
async def create_process_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    name: Optional[str] = Form(default=None),  # 自定义会议名称
    asr_model: Optional[str] = Form(default=None),  # 统一 ASR 模型（新）
    whisper_model: str = Form(default="medium"),     # 向后兼容
    llm_model: Optional[str] = Form(default=None),
    gender_model: Optional[str] = Form(default=None),
    enable_naming: bool = Form(default=True),
    enable_correction: bool = Form(default=True),
    enable_summary: bool = Form(default=True),
    enhance_mode: str = Form(default="none"),
):
    """
    上传音频文件并创建处理任务

    返回任务 ID，可通过 GET /api/jobs/{job_id} 查询进度
    """
    # 验证文件类型
    allowed_types = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"}
    file_ext = Path(file.filename or "").suffix.lower()
    if file_ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {file_ext}，支持: {', '.join(allowed_types)}"
        )

    # 创建任务
    job_id = str(uuid.uuid4())[:8]
    settings = get_settings()

    # 创建输出目录 - 使用自定义名称或文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if name and name.strip():
        folder_name = f"{name.strip()}_{timestamp}"
    else:
        file_stem = Path(file.filename or "audio").stem
        folder_name = f"{file_stem}_{timestamp}"
    output_dir = settings.paths.output_dir / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存上传的文件
    upload_path = output_dir / f"original{file_ext}"
    content = await file.read()
    upload_path.write_bytes(content)

    # 创建任务记录
    _jobs[job_id] = {
        "job_id": job_id,
        "status": JobStatus.PENDING,
        "progress": 0,
        "message": "等待处理",
        "created_at": datetime.now(),
        "completed_at": None,
        "output_dir": str(output_dir),
        "upload_path": str(upload_path),
        "options": {
            "asr_model": asr_model or whisper_model,  # 优先用 asr_model
            "whisper_model": whisper_model,            # 向后兼容
            "llm_model": llm_model,
            "gender_model": gender_model,
            "enable_naming": enable_naming,
            "enable_correction": enable_correction,
            "enable_summary": enable_summary,
            "enhance_mode": enhance_mode,
        },
        "result": None,
    }

    # 在后台执行处理
    background_tasks.add_task(_process_audio_task, job_id)

    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        progress=0,
        message="任务已创建",
        created_at=_jobs[job_id]["created_at"],
    )


async def _process_audio_task(job_id: str):
    """后台处理音频任务"""
    job = _jobs[job_id]
    job["status"] = JobStatus.PROCESSING

    try:
        # 在线程池中运行同步处理
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _sync_process_audio, job_id)

        job["result"] = result
        job["status"] = JobStatus.COMPLETED
        job["progress"] = 1.0
        job["message"] = "处理完成"
        job["completed_at"] = datetime.now()

    except Exception as e:
        import traceback
        traceback.print_exc()
        job["status"] = JobStatus.FAILED
        job["message"] = str(e)
        job["completed_at"] = datetime.now()


def _sync_process_audio(job_id: str) -> dict:
    """同步处理音频（在线程池中运行）"""
    job = _jobs[job_id]
    options = job["options"]
    output_dir = Path(job["output_dir"])
    upload_path = Path(job["upload_path"])

    def update_progress(progress: float, message: str):
        job["progress"] = progress
        job["message"] = message

    # 导入处理模块
    from meeting_ai.config import get_settings
    from meeting_ai.utils.audio import ensure_wav_16k_mono
    from meeting_ai.services import get_diarization_service, get_asr_engine
    from meeting_ai.services.alignment import (
        align_transcript_with_speakers,
        fix_unknown_speakers,
        merge_adjacent_segments,
    )
    from meeting_ai.services.gender import detect_all_genders
    from meeting_ai.services.naming import get_naming_service
    from meeting_ai.services.correction import correct_segments
    from meeting_ai.services.summary import summarize_meeting, format_summary_markdown
    from meeting_ai.models import SpeakerInfo

    settings = get_settings()

    # ============================================================
    # 日志捕获：将处理过程中的所有 meeting_ai.* 日志写入文件
    # ============================================================
    log_path = output_dir / "process.log"
    file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    root_log = logging.getLogger("meeting_ai")
    root_log.addHandler(file_handler)

    step_times: dict[str, float] = {}
    pipeline_start = time.time()

    def save_debug(name: str, data):
        """保存中间结果到 output 目录"""
        try:
            (output_dir / f"debug_{name}.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass

    try:
        # 配置模型
        asr_model_name = options.get("asr_model") or options.get("whisper_model")
        if not asr_model_name:
            raise ValueError("请选择 ASR 语音识别模型")

        use_llm = options["llm_model"] and options["llm_model"] != "disabled"
        if use_llm:
            settings.llm.enabled = True
            llm_model_name = options["llm_model"]
            if not llm_model_name.endswith(".gguf"):
                llm_model_name = f"{llm_model_name}.gguf"
            if not llm_model_name.startswith("llm/") and not llm_model_name.startswith("llm\\"):
                llm_model_name = f"llm/{llm_model_name}"
            settings.llm.model_path = Path(llm_model_name)
        else:
            settings.llm.enabled = False

        # 保存选项快照
        save_debug("0_pipeline_options", {
            "options": options,
            "models_used": {
                "asr": asr_model_name,
                "diarization": "pyannote-3.1",
                "gender": options.get("gender_model") or "f0",
                "llm": options.get("llm_model") or "disabled",
            },
        })

        # ── 1. 转换音频格式 ──
        update_progress(0.05, "转换音频格式...")
        t0 = time.time()
        wav_path_original = output_dir / "audio_16k.wav"
        ensure_wav_16k_mono(upload_path, wav_path_original)
        step_times["1_audio_convert"] = time.time() - t0

        # ── 2. 音频增强 ──
        wav_path_for_asr = wav_path_original  # 默认用原始音频
        enhance_mode = options.get("enhance_mode", "none")
        if enhance_mode and enhance_mode != "none":
            update_progress(0.10, "音频增强...")
            t0 = time.time()
            try:
                from meeting_ai.utils.enhance import enhance_audio
                enhanced_path = output_dir / "audio_enhanced.wav"
                enhance_audio(wav_path_original, enhanced_path, mode=enhance_mode)
                wav_path_for_asr = enhanced_path  # ASR 用增强后的音频
            except ImportError:
                pass
            step_times["2_enhance"] = time.time() - t0

        # ── 3. 说话人分离 ──
        # CRITICAL: diarization 必须用原始音频，保留声纹特征
        update_progress(0.15, "说话人分离...")
        t0 = time.time()
        diar_service = get_diarization_service("pyannote-3.1")
        diar_result = diar_service.diarize(wav_path_original)
        step_times["3_diarization"] = time.time() - t0

        diar_speakers = sorted(set(s.speaker for s in diar_result.segments if s.speaker))
        save_debug("1_diarization", {
            "speaker_count": len(diar_speakers),
            "speakers": diar_speakers,
            "segment_count": len(diar_result.segments),
            "segments": [
                {"start": round(s.start, 3), "end": round(s.end, 3), "speaker": s.speaker}
                for s in diar_result.segments
            ],
        })

        # ── 4. 语音转写 ──
        # ASR 用增强后的音频（如果有）可提升转写质量
        update_progress(0.40, "语音转写...")
        t0 = time.time()
        asr_engine = get_asr_engine(asr_model_name)
        asr_result = asr_engine.transcribe(wav_path_for_asr)
        step_times["4_asr"] = time.time() - t0

        save_debug("2_asr", {
            "duration": asr_result.duration,
            "segment_count": len(asr_result.segments),
            "segments": [
                {"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text}
                for s in asr_result.segments
            ],
        })

        # ── 5. 对齐 ──
        update_progress(0.55, "对齐说话人...")
        t0 = time.time()
        aligned_segments = align_transcript_with_speakers(asr_result, diar_result)
        step_times["5a_alignment"] = time.time() - t0

        save_debug("3_aligned", [
            {"id": s.id, "start": round(s.start, 3), "end": round(s.end, 3),
             "text": s.text, "speaker": s.speaker}
            for s in aligned_segments
        ])

        t0 = time.time()
        fixed_segments = fix_unknown_speakers(aligned_segments)
        step_times["5b_fix_unknown"] = time.time() - t0

        save_debug("4_fixed", [
            {"id": s.id, "start": round(s.start, 3), "end": round(s.end, 3),
             "text": s.text, "speaker": s.speaker}
            for s in fixed_segments
        ])

        t0 = time.time()
        final_segments = merge_adjacent_segments(fixed_segments, max_gap=0.3)
        step_times["5c_merge"] = time.time() - t0

        save_debug("5_merged", [
            {"id": s.id, "start": round(s.start, 3), "end": round(s.end, 3),
             "text": s.text, "speaker": s.speaker}
            for s in final_segments
        ])

        # ── 6. 错别字校正 ──
        if options["enable_correction"] and use_llm:
            update_progress(0.65, "错别字校正...")
            t0 = time.time()
            try:
                final_segments = correct_segments(final_segments)
            except Exception as e:
                logger.warning(f"错别字校正失败: {e}")
            step_times["6_correction"] = time.time() - t0

        # ── 7. 智能命名 ──
        speakers = {}
        gender_map = {}
        if options["enable_naming"]:
            update_progress(0.75, "智能命名...")

            # 7a. 性别检测
            t0 = time.time()
            try:
                gender_engine = options.get("gender_model") or "f0"
                # 性别检测用原始音频，保留声纹特征
                gender_map = detect_all_genders(wav_path_original, final_segments, engine_name=gender_engine)
            except Exception as e:
                logger.warning(f"性别检测失败（回退空性别信息）: {e}")
            step_times["7a_gender"] = time.time() - t0

            save_debug("6_gender", {
                spk: {"gender": g.value, "metric": round(m, 4)}
                for spk, (g, m) in gender_map.items()
            })

            # 7b. 智能命名
            t0 = time.time()
            try:
                naming_service = get_naming_service()
                speakers = naming_service.name_speakers(final_segments, gender_map)
            except Exception as e:
                logger.warning(f"智能命名失败: {e}")
            step_times["7b_naming"] = time.time() - t0

            save_debug("7_naming", {
                spk: info.model_dump() for spk, info in speakers.items()
            } if speakers else {})

        # 默认说话人信息
        if not speakers:
            speaker_ids = set(seg.speaker for seg in final_segments if seg.speaker)
            for spk_id in speaker_ids:
                total_dur = sum(seg.duration for seg in final_segments if seg.speaker == spk_id)
                seg_count = sum(1 for seg in final_segments if seg.speaker == spk_id)
                speakers[spk_id] = SpeakerInfo(
                    id=spk_id,
                    display_name=spk_id,
                    total_duration=total_dur,
                    segment_count=seg_count,
                )

        # ── 8. 会议总结 ──
        summary_md = ""
        if options["enable_summary"] and use_llm:
            update_progress(0.85, "生成总结...")
            t0 = time.time()
            try:
                summary_obj = summarize_meeting(final_segments, speakers, duration=asr_result.duration)
                if summary_obj:
                    summary_md = format_summary_markdown(summary_obj, speakers, duration=asr_result.duration)
            except Exception as e:
                logger.warning(f"会议总结生成失败: {e}")
                summary_md = "*总结生成失败*"
            step_times["8_summary"] = time.time() - t0

        # ── 9. 保存结果 ──
        update_progress(0.95, "保存结果...")
        segments_data = [s.model_dump() for s in final_segments]
        speakers_data = {k: v.model_dump() for k, v in speakers.items()}

        result_data = {
            "speakers": speakers_data,
            "segments": segments_data,
            "speaker_count": len(speakers),
        }
        (output_dir / "result.json").write_text(
            json.dumps(result_data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        # 保存文本
        lines = []
        for seg in segments_data:
            sid = seg.get("speaker", "UNKNOWN")
            name = speakers_data.get(sid, {}).get("display_name", sid)
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            text = seg.get("text", "")
            lines.append(f"[{int(start//60):02d}:{int(start%60):02d}-{int(end//60):02d}:{int(end%60):02d}] {name}: {text}")
        (output_dir / "result.txt").write_text("\n".join(lines), encoding="utf-8")

        # 保存总结
        if summary_md:
            (output_dir / "summary.md").write_text(summary_md, encoding="utf-8")

        # ── 保存管线统计 ──
        step_times["total"] = time.time() - pipeline_start
        save_debug("0_pipeline_info", {
            "options": options,
            "models_used": {
                "asr": asr_model_name,
                "diarization": "pyannote-3.1",
                "gender": options.get("gender_model") or "f0",
                "llm": options.get("llm_model") or "disabled",
                "enhance": enhance_mode,
            },
            "timing_seconds": {k: round(v, 2) for k, v in step_times.items()},
            "result_stats": {
                "diarization_speakers": len(diar_speakers),
                "asr_segments": len(asr_result.segments),
                "aligned_segments": len(aligned_segments),
                "fixed_segments": len(fixed_segments),
                "final_segments": len(final_segments),
                "speakers_named": len(speakers),
                "duration": asr_result.duration,
            },
        })

        # 返回结果
        return {
            "segments": segments_data,
            "speakers": speakers_data,
            "summary": summary_md,
            "duration": asr_result.duration,
            "audio_path": str(wav_path_for_asr),  # 返回 ASR 使用的音频路径
        }

    finally:
        # 确保日志 handler 被清理
        root_log.removeHandler(file_handler)
        file_handler.close()


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """获取任务状态"""
    job = _get_job(job_id)
    return JobResponse(
        job_id=job["job_id"],
        status=job["status"],
        progress=job["progress"],
        message=job["message"],
        created_at=job["created_at"],
        completed_at=job.get("completed_at"),
    )


@router.get("/jobs/{job_id}/result", response_model=ProcessResultResponse)
async def get_job_result(job_id: str):
    """获取任务处理结果"""
    job = _get_job(job_id)

    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"任务尚未完成，当前状态: {job['status']}"
        )

    result = job["result"]
    output_dir = Path(job["output_dir"])

    # 构建音频 URL
    settings = get_settings()
    relative_path = output_dir.relative_to(settings.paths.output_dir)
    audio_original_url = f"/files/{relative_path}/audio_16k.wav"
    if (output_dir / "audio_enhanced.wav").exists():
        audio_url = f"/files/{relative_path}/audio_enhanced.wav"
    else:
        audio_url = audio_original_url

    # 构建响应
    segments = [
        SegmentResponse(
            id=i,
            start=seg["start"],
            end=seg["end"],
            text=seg["text"],
            speaker=seg["speaker"],
            speaker_name=result["speakers"].get(seg["speaker"], {}).get("display_name", seg["speaker"]),
        )
        for i, seg in enumerate(result["segments"])
    ]

    speakers = {
        k: SpeakerResponse(
            id=k,
            display_name=v.get("display_name", k),
            gender=v.get("gender"),
            total_duration=v.get("total_duration", 0),
            segment_count=v.get("segment_count", 0),
        )
        for k, v in result["speakers"].items()
    }

    return ProcessResultResponse(
        job_id=job_id,
        status=job["status"],
        segments=segments,
        speakers=speakers,
        summary=result.get("summary", ""),
        audio_url=audio_url,
        audio_original_url=audio_original_url,
        duration=result.get("duration", 0),
        output_dir=str(output_dir.name),
    )


@router.put("/jobs/{job_id}/segments/{segment_id}")
async def update_segment(job_id: str, segment_id: int, request: SegmentUpdateRequest):
    """更新对话片段"""
    job = _get_job(job_id)

    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="任务未完成")

    result = job["result"]
    if segment_id < 0 or segment_id >= len(result["segments"]):
        raise HTTPException(status_code=404, detail="片段不存在")

    segment = result["segments"][segment_id]

    if request.text is not None:
        segment["text"] = request.text

    if request.speaker_name is not None:
        speaker_id = segment["speaker"]
        if speaker_id in result["speakers"]:
            result["speakers"][speaker_id]["display_name"] = request.speaker_name

    # 保存更新
    _save_result(job)

    return {"status": "ok"}


@router.post("/jobs/{job_id}/segments/{segment_id}/split")
async def split_segment(job_id: str, segment_id: int, request: SegmentSplitRequest):
    """
    分割对话片段

    将一个片段在指定位置分割成两个片段，时间按文本比例分配
    """
    job = _get_job(job_id)

    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="任务未完成")

    result = job["result"]
    if segment_id < 0 or segment_id >= len(result["segments"]):
        raise HTTPException(status_code=404, detail="片段不存在")

    segment = result["segments"][segment_id]
    text = segment.get("text", "")

    # 验证分割位置
    if request.split_position <= 0 or request.split_position >= len(text):
        raise HTTPException(status_code=400, detail="分割位置无效")

    # 计算时间分割点（按文本比例）
    start_time = segment["start"]
    end_time = segment["end"]
    duration = end_time - start_time
    split_ratio = request.split_position / len(text)
    split_time = start_time + duration * split_ratio

    # 分割文本
    text1 = text[:request.split_position].strip()
    text2 = text[request.split_position:].strip()

    # 确定说话人
    speaker1 = segment["speaker"]
    speaker2 = request.new_speaker if request.new_speaker else speaker1

    # 创建两个新片段
    new_segment1 = {
        "id": segment_id,
        "start": start_time,
        "end": split_time,
        "text": text1,
        "speaker": speaker1,
    }

    new_segment2 = {
        "id": segment_id + 1,  # 临时 ID，后面会重新编号
        "start": split_time,
        "end": end_time,
        "text": text2,
        "speaker": speaker2,
    }

    # 替换原片段为两个新片段
    result["segments"] = (
        result["segments"][:segment_id] +
        [new_segment1, new_segment2] +
        result["segments"][segment_id + 1:]
    )

    # 重新编号所有片段
    for i, seg in enumerate(result["segments"]):
        seg["id"] = i

    # 保存更新
    _save_result(job)

    return {
        "status": "ok",
        "segments": [new_segment1, new_segment2],
    }


@router.put("/jobs/{job_id}/speakers")
async def rename_speaker(job_id: str, request: SpeakerRenameRequest):
    """全局重命名说话人"""
    job = _get_job(job_id)

    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="任务未完成")

    result = job["result"]
    if request.speaker_id not in result["speakers"]:
        raise HTTPException(status_code=404, detail="说话人不存在")

    result["speakers"][request.speaker_id]["display_name"] = request.new_name

    # 保存更新
    _save_result(job)

    return {"status": "ok"}


@router.put("/jobs/{job_id}/summary")
async def update_summary(job_id: str, request: SummaryUpdateRequest):
    """更新会议总结"""
    job = _get_job(job_id)

    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="任务未完成")

    job["result"]["summary"] = request.summary

    # 保存更新
    _save_result(job)

    return {"status": "ok"}


def _save_result(job: dict):
    """保存结果到文件"""
    output_dir = Path(job["output_dir"])
    result = job["result"]

    # 保存 JSON
    result_data = {
        "speakers": result["speakers"],
        "segments": result["segments"],
        "speaker_count": len(result["speakers"]),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result_data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # 保存文本
    lines = []
    for seg in result["segments"]:
        sid = seg.get("speaker", "UNKNOWN")
        name = result["speakers"].get(sid, {}).get("display_name", sid)
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        text = seg.get("text", "")
        lines.append(f"[{int(start//60):02d}:{int(start%60):02d}-{int(end//60):02d}:{int(end%60):02d}] {name}: {text}")
    (output_dir / "result.txt").write_text("\n".join(lines), encoding="utf-8")

    # 保存总结
    if result.get("summary"):
        (output_dir / "summary.md").write_text(result["summary"], encoding="utf-8")
