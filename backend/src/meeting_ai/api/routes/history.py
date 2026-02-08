"""
历史记录 API 路由
"""

import json
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from meeting_ai.api.schemas import (
    HistoryItemResponse,
    HistoryListResponse,
    ProcessResultResponse,
    SegmentResponse,
    SegmentUpdateRequest,
    SegmentSplitRequest,
    SegmentMergeRequest,
    SpeakerRenameRequest,
    SpeakerResponse,
    SummaryUpdateRequest,
    JobStatus,
)
from meeting_ai.config import get_settings

router = APIRouter()


@router.get("/history", response_model=HistoryListResponse)
async def list_history(limit: int = 50, offset: int = 0):
    """获取历史记录列表"""
    settings = get_settings()
    output_dir = settings.paths.output_dir

    if not output_dir.exists():
        return HistoryListResponse(items=[], total=0)

    items = []
    for d in sorted(output_dir.iterdir(), reverse=True):
        if not d.is_dir():
            continue

        result_file = d / "result.json"
        if not result_file.exists():
            continue

        try:
            data = json.loads(result_file.read_text(encoding="utf-8"))
            segments = data.get("segments", [])

            # 计算时长
            duration = 0
            if segments:
                duration = max(seg.get("end", 0) for seg in segments)

            # 检查是否有总结
            has_summary = (d / "summary.md").exists()

            # 获取创建时间
            created_at = datetime.fromtimestamp(result_file.stat().st_mtime)

            items.append(HistoryItemResponse(
                id=d.name,
                name=d.name,
                created_at=created_at,
                duration=duration,
                segment_count=len(segments),
                has_summary=has_summary,
            ))
        except Exception:
            continue

    total = len(items)
    items = items[offset:offset + limit]

    return HistoryListResponse(items=items, total=total)


@router.get("/history/{history_id}", response_model=ProcessResultResponse)
async def get_history_item(history_id: str):
    """获取历史记录详情"""
    settings = get_settings()
    output_dir = settings.paths.output_dir / history_id

    if not output_dir.exists():
        raise HTTPException(status_code=404, detail="历史记录不存在")

    result_file = output_dir / "result.json"
    if not result_file.exists():
        raise HTTPException(status_code=404, detail="结果文件不存在")

    try:
        data = json.loads(result_file.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取结果文件失败: {e}")

    segments_data = data.get("segments", [])
    speakers_data = data.get("speakers", {})

    # 读取总结
    summary = ""
    summary_file = output_dir / "summary.md"
    if summary_file.exists():
        summary = summary_file.read_text(encoding="utf-8")

    # 计算时长
    duration = 0
    if segments_data:
        duration = max(seg.get("end", 0) for seg in segments_data)

    # 构建音频 URL - 检查多种可能的文件名
    # 原始音频
    if (output_dir / "audio_16k.wav").exists():
        audio_original_url = f"/files/{history_id}/audio_16k.wav"
    elif (output_dir / "recording.wav").exists():
        audio_original_url = f"/files/{history_id}/recording.wav"
    else:
        audio_original_url = f"/files/{history_id}/audio_16k.wav"
    # 增强音频（如果有）
    if (output_dir / "audio_enhanced.wav").exists():
        audio_url = f"/files/{history_id}/audio_enhanced.wav"
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
            speaker_name=speakers_data.get(seg["speaker"], {}).get("display_name", seg["speaker"]),
        )
        for i, seg in enumerate(segments_data)
    ]

    speakers = {
        k: SpeakerResponse(
            id=k,
            display_name=v.get("display_name", k),
            gender=v.get("gender"),
            total_duration=v.get("total_duration", 0),
            segment_count=v.get("segment_count", 0),
        )
        for k, v in speakers_data.items()
    }

    return ProcessResultResponse(
        job_id=history_id,
        status=JobStatus.COMPLETED,
        segments=segments,
        speakers=speakers,
        summary=summary,
        audio_url=audio_url,
        audio_original_url=audio_original_url,
        duration=duration,
        output_dir=history_id,
    )


@router.delete("/history/{history_id}")
async def delete_history_item(history_id: str):
    """删除历史记录"""
    settings = get_settings()
    output_dir = settings.paths.output_dir / history_id

    if not output_dir.exists():
        raise HTTPException(status_code=404, detail="历史记录不存在")

    try:
        import shutil
        shutil.rmtree(output_dir)
        return {"status": "ok", "message": f"已删除: {history_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {e}")


@router.put("/history/{history_id}/rename")
async def rename_history_item(history_id: str, new_name: str):
    """重命名历史记录"""
    settings = get_settings()
    old_dir = settings.paths.output_dir / history_id

    if not old_dir.exists():
        raise HTTPException(status_code=404, detail="历史记录不存在")

    # 提取原始时间戳（如果有的话）
    # 格式: name_YYYYMMDD_HHMMSS
    parts = history_id.rsplit("_", 2)
    if len(parts) >= 3 and len(parts[-1]) == 6 and len(parts[-2]) == 8:
        # 有时间戳，保留它
        timestamp = f"{parts[-2]}_{parts[-1]}"
        new_folder_name = f"{new_name.strip()}_{timestamp}"
    else:
        # 没有时间戳，添加当前时间
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_folder_name = f"{new_name.strip()}_{timestamp}"

    new_dir = settings.paths.output_dir / new_folder_name

    if new_dir.exists():
        raise HTTPException(status_code=400, detail="目标名称已存在")

    try:
        old_dir.rename(new_dir)
        return {
            "status": "ok",
            "old_id": history_id,
            "new_id": new_folder_name,
            "message": f"已重命名为: {new_folder_name}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重命名失败: {e}")


@router.get("/history/{history_id}/export/{format}")
async def export_history(history_id: str, format: str):
    """导出历史记录"""
    settings = get_settings()
    output_dir = settings.paths.output_dir / history_id

    if not output_dir.exists():
        raise HTTPException(status_code=404, detail="历史记录不存在")

    file_map = {
        "txt": "result.txt",
        "json": "result.json",
        "md": "summary.md",
    }

    if format not in file_map:
        raise HTTPException(status_code=400, detail=f"不支持的格式: {format}")

    file_path = output_dir / file_map[format]
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"文件不存在: {file_map[format]}")

    return FileResponse(
        path=str(file_path),
        filename=f"{history_id}.{format}",
        media_type="application/octet-stream",
    )


def _load_result(history_id: str) -> tuple[Path, dict]:
    """加载历史记录的结果数据"""
    settings = get_settings()
    output_dir = settings.paths.output_dir / history_id

    if not output_dir.exists():
        raise HTTPException(status_code=404, detail="历史记录不存在")

    result_file = output_dir / "result.json"
    if not result_file.exists():
        raise HTTPException(status_code=404, detail="结果文件不存在")

    try:
        data = json.loads(result_file.read_text(encoding="utf-8"))
        return output_dir, data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取结果文件失败: {e}")


def _save_result(output_dir: Path, data: dict, summary: str = None):
    """保存结果到文件"""
    # 保存 JSON
    result_data = {
        "speakers": data.get("speakers", {}),
        "segments": data.get("segments", []),
        "speaker_count": len(data.get("speakers", {})),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result_data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # 保存文本
    lines = []
    speakers_data = data.get("speakers", {})
    for seg in data.get("segments", []):
        sid = seg.get("speaker", "UNKNOWN")
        name = speakers_data.get(sid, {}).get("display_name", sid)
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        text = seg.get("text", "")
        lines.append(f"[{int(start//60):02d}:{int(start%60):02d}-{int(end//60):02d}:{int(end%60):02d}] {name}: {text}")
    (output_dir / "result.txt").write_text("\n".join(lines), encoding="utf-8")

    # 保存总结
    if summary is not None:
        (output_dir / "summary.md").write_text(summary, encoding="utf-8")


@router.put("/history/{history_id}/segments/{segment_id}")
async def update_history_segment(history_id: str, segment_id: int, request: SegmentUpdateRequest):
    """更新历史记录中的对话片段"""
    output_dir, data = _load_result(history_id)
    segments = data.get("segments", [])

    if segment_id < 0 or segment_id >= len(segments):
        raise HTTPException(status_code=404, detail="片段不存在")

    segment = segments[segment_id]

    if request.text is not None:
        segment["text"] = request.text

    if request.speaker_name is not None:
        speaker_id = segment["speaker"]
        if speaker_id in data.get("speakers", {}):
            data["speakers"][speaker_id]["display_name"] = request.speaker_name

    _save_result(output_dir, data)
    return {"status": "ok"}


@router.post("/history/{history_id}/segments/{segment_id}/split")
async def split_history_segment(history_id: str, segment_id: int, request: SegmentSplitRequest):
    """
    分割历史记录中的对话片段

    将一个片段在指定位置分割成两个片段，时间按文本比例分配
    """
    output_dir, data = _load_result(history_id)
    segments = data.get("segments", [])

    if segment_id < 0 or segment_id >= len(segments):
        raise HTTPException(status_code=404, detail="片段不存在")

    segment = segments[segment_id]
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
        "id": segment_id + 1,
        "start": split_time,
        "end": end_time,
        "text": text2,
        "speaker": speaker2,
    }

    # 替换原片段为两个新片段
    data["segments"] = segments[:segment_id] + [new_segment1, new_segment2] + segments[segment_id + 1:]

    # 重新编号所有片段
    for i, seg in enumerate(data["segments"]):
        seg["id"] = i

    _save_result(output_dir, data)

    return {
        "status": "ok",
        "segments": [new_segment1, new_segment2],
    }


@router.put("/history/{history_id}/segments/{segment_id}/speaker")
async def change_segment_speaker(history_id: str, segment_id: int, speaker_id: str):
    """更改片段的说话人"""
    output_dir, data = _load_result(history_id)
    segments = data.get("segments", [])

    if segment_id < 0 or segment_id >= len(segments):
        raise HTTPException(status_code=404, detail="片段不存在")

    # 验证说话人存在
    if speaker_id not in data.get("speakers", {}):
        raise HTTPException(status_code=404, detail="说话人不存在")

    segments[segment_id]["speaker"] = speaker_id
    _save_result(output_dir, data)

    return {"status": "ok"}


@router.put("/history/{history_id}/speakers")
async def rename_history_speaker(history_id: str, request: SpeakerRenameRequest):
    """全局重命名历史记录中的说话人"""
    output_dir, data = _load_result(history_id)

    if request.speaker_id not in data.get("speakers", {}):
        raise HTTPException(status_code=404, detail="说话人不存在")

    data["speakers"][request.speaker_id]["display_name"] = request.new_name
    _save_result(output_dir, data)

    return {"status": "ok"}


@router.put("/history/{history_id}/summary")
async def update_history_summary(history_id: str, request: SummaryUpdateRequest):
    """更新历史记录的会议总结"""
    output_dir, data = _load_result(history_id)
    _save_result(output_dir, data, summary=request.summary)
    return {"status": "ok"}


@router.post("/history/{history_id}/summary/regenerate")
async def regenerate_history_summary(history_id: str, request: dict | None = None):
    """
    重新生成历史记录的会议总结

    使用当前的对话片段和说话人信息重新生成会议总结。
    可通过 request body 传入 llm_model 指定使用的 LLM 模型。
    """
    import asyncio
    from pathlib import Path
    from meeting_ai.services.summary import summarize_meeting, format_summary_markdown
    from meeting_ai.services.llm import reset_llm
    from meeting_ai.models import Segment, SpeakerInfo
    from meeting_ai.config import get_settings

    settings = get_settings()

    # 根据前端选择的 LLM 模型配置
    body = request or {}
    llm_model = body.get("llm_model")
    if llm_model and llm_model != "disabled":
        settings.llm.enabled = True
        llm_model_name = llm_model
        if not llm_model_name.endswith(".gguf"):
            llm_model_name = f"{llm_model_name}.gguf"
        if not llm_model_name.startswith("llm/") and not llm_model_name.startswith("llm\\"):
            llm_model_name = f"llm/{llm_model_name}"
        settings.llm.model_path = Path(llm_model_name)
        # 重置 LLM 单例，让它重新按新路径加载
        reset_llm()
    elif llm_model == "disabled":
        raise HTTPException(status_code=400, detail="未选择 LLM 模型")

    # 检查 LLM 是否可用
    if not settings.llm.enabled:
        raise HTTPException(status_code=400, detail="LLM 未启用，无法生成总结")

    output_dir, data = _load_result(history_id)
    segments_data = data.get("segments", [])
    speakers_data = data.get("speakers", {})

    if not segments_data:
        raise HTTPException(status_code=400, detail="没有对话片段，无法生成总结")

    # 转换为 Segment 对象
    segments = [
        Segment(
            id=seg["id"],
            start=seg["start"],
            end=seg["end"],
            text=seg["text"],
            speaker=seg["speaker"],
        )
        for seg in segments_data
    ]

    # 转换为 SpeakerInfo 对象
    speakers = {
        sid: SpeakerInfo(
            id=sid,
            display_name=info.get("display_name", sid),
            gender=info.get("gender"),
            total_duration=info.get("total_duration", 0),
            segment_count=info.get("segment_count", 0),
        )
        for sid, info in speakers_data.items()
    }

    # 计算总时长
    total_duration = segments[-1].end if segments else 0

    # 在线程池中运行 LLM 生成（避免阻塞）
    loop = asyncio.get_event_loop()
    try:
        summary_obj = await loop.run_in_executor(
            None,
            lambda: summarize_meeting(segments, speakers, duration=total_duration)
        )

        if summary_obj:
            summary_md = format_summary_markdown(summary_obj, speakers, duration=total_duration)
        else:
            summary_md = "*总结生成失败*"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成总结失败: {e}")

    # 保存新的总结
    _save_result(output_dir, data, summary=summary_md)

    return {
        "status": "ok",
        "summary": summary_md,
    }


@router.post("/history/{history_id}/segments/merge")
async def merge_history_segments(history_id: str, request: SegmentMergeRequest):
    """
    合并历史记录中的多个连续片段

    将多个连续的片段合并为一个，时间范围为第一个片段的开始到最后一个片段的结束
    """
    output_dir, data = _load_result(history_id)
    segments = data.get("segments", [])

    # 验证 segment_ids
    if len(request.segment_ids) < 2:
        raise HTTPException(status_code=400, detail="至少需要选择两个片段进行合并")

    # 排序并验证连续性
    sorted_ids = sorted(request.segment_ids)
    for i in range(len(sorted_ids) - 1):
        if sorted_ids[i + 1] - sorted_ids[i] != 1:
            raise HTTPException(status_code=400, detail="只能合并连续的片段")

    # 验证所有 ID 在有效范围内
    if sorted_ids[0] < 0 or sorted_ids[-1] >= len(segments):
        raise HTTPException(status_code=404, detail="片段 ID 超出范围")

    # 获取要合并的片段
    merge_segments = [segments[i] for i in sorted_ids]

    # 创建合并后的片段
    merged_segment = {
        "id": sorted_ids[0],
        "start": merge_segments[0]["start"],
        "end": merge_segments[-1]["end"],
        "text": " ".join(seg["text"] for seg in merge_segments),
        "speaker": merge_segments[0]["speaker"],  # 使用第一个片段的说话人
    }

    # 重建片段列表
    new_segments = segments[:sorted_ids[0]] + [merged_segment] + segments[sorted_ids[-1] + 1:]

    # 重新编号所有片段
    for i, seg in enumerate(new_segments):
        seg["id"] = i

    data["segments"] = new_segments
    _save_result(output_dir, data)

    return {
        "status": "ok",
        "merged_segment": merged_segment,
        "new_segment_count": len(new_segments),
    }
