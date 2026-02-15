"""
模型管理 API 路由
"""

import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from meeting_ai.api.schemas import (
    ModelInfoResponse,
    ModelsListResponse,
    StreamingEngineResponse,
    StreamingEnginesListResponse,
    SystemInfoResponse,
)
from meeting_ai.config import get_settings
from meeting_ai.services.asr import detect_engine_type
from meeting_ai.services.streaming_asr import list_available_engines

# 已知性别检测模型的元数据
GENDER_MODEL_META: dict[str, dict[str, str]] = {
    "f0": {"metric": "≈90%", "vram": "CPU", "label": "基频分析 (内置)"},
    "ecapa-gender": {"metric": "≈97%", "vram": "≈0.3G", "label": "ECAPA-TDNN"},
    "wav2vec2-gender": {"metric": "≈95%", "vram": "≈1.5G", "label": "Wav2Vec2"},
    "audeering-gender": {"metric": "≈98%", "vram": "≈1.5G", "label": "Wav2Vec2-Robust (Age+Gender)"},
}

# 已知 ASR 模型的元数据（CER 取自 AISHELL-1 基准，VRAM 为 float16 推理估算）
ASR_MODEL_META: dict[str, dict[str, str]] = {
    # faster-whisper (name 是去掉 faster-whisper- 前缀后的)
    "tiny": {"cer": "—", "vram": "≈1G", "label": "Whisper tiny"},
    "base": {"cer": "—", "vram": "≈1G", "label": "Whisper base"},
    "small": {"cer": "—", "vram": "≈1G", "label": "Whisper small"},
    "medium": {"cer": "5.1%", "vram": "≈2G", "label": "Whisper medium"},
    "large-v2": {"cer": "5.1%", "vram": "≈4G", "label": "Whisper large-v2"},
    "large-v3": {"cer": "5.1%", "vram": "≈4G", "label": "Whisper large-v3"},
    # FunASR
    "sensevoice-small": {"cer": "3.0%", "vram": "≈1G", "label": "SenseVoice-Small"},
    "sensevoice-large": {"cer": "2.1%", "vram": "≈4G", "label": "SenseVoice-Large"},
    "paraformer-large": {"cer": "1.7%", "vram": "≈1G", "label": "Paraformer-Large"},
    # FireRedASR
    "fireredasr-aed": {"cer": "0.6%", "vram": "≈3G", "label": "FireRedASR-AED"},
}


router = APIRouter()


@router.get("/models", response_model=ModelsListResponse)
async def list_models():
    """获取可用模型列表"""
    settings = get_settings()
    asr_models: list[ModelInfoResponse] = []
    llm_models = []

    def _build_asr_display_name(name: str, size_mb: float) -> str:
        """构建 ASR display_name: 紧凑格式 Label (精度 显存)"""
        meta = ASR_MODEL_META.get(name)
        if meta:
            return f"{meta['label']} ({meta['cer']} {meta['vram']})"
        return f"{name} ({size_mb:.0f}MB)"

    def _build_gender_display_name(name: str, size_mb: float) -> str:
        """构建性别检测 display_name"""
        meta = GENDER_MODEL_META.get(name)
        if meta:
            return f"{meta['label']} ({meta['metric']} {meta['vram']})"
        return f"{name} ({size_mb:.0f}MB)"

        return f"{name} ({size_mb:.0f}MB)"

    # 扫描 Whisper 模型 (models/whisper/)
    whisper_dir = settings.paths.models_dir / "whisper"
    if whisper_dir.exists():
        for d in sorted(whisper_dir.iterdir()):
            if d.is_dir() and (d.name.startswith("faster-whisper-") or (d / "config.json").exists()):
                model_name = d.name.replace("faster-whisper-", "")
                size_mb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / (1024 * 1024)
                asr_models.append(ModelInfoResponse(
                    name=model_name,
                    display_name=_build_asr_display_name(model_name, size_mb),
                    path=str(d),
                    size_mb=round(size_mb, 1),
                    engine="faster-whisper",
                ))

    # 扫描非 Whisper ASR 模型 (models/asr/)
    asr_dir = settings.paths.models_dir / "asr"
    if asr_dir.exists():
        for d in sorted(asr_dir.iterdir()):
            if not d.is_dir():
                continue
            engine_type = detect_engine_type(d)
            if engine_type is None:
                continue
            size_mb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / (1024 * 1024)
            asr_models.append(ModelInfoResponse(
                name=d.name,
                display_name=_build_asr_display_name(d.name, size_mb),
                path=str(d),
                size_mb=round(size_mb, 1),
                engine=engine_type,
            ))

    # 向后兼容: whisper_models = asr_models 中 engine=="faster-whisper" 的子集
    whisper_models = [m for m in asr_models if m.engine == "faster-whisper"]

    # 扫描 LLM 模型
    llm_models.append(ModelInfoResponse(
        name="disabled",
        display_name="不使用 LLM",
        path="",
        size_mb=None,
    ))

    llm_dir = settings.paths.models_dir / "llm"
    if llm_dir.exists():
        for f in sorted(llm_dir.glob("*.gguf")):
            display_name = f.stem
            if len(display_name) > 35:
                display_name = display_name[:32] + "..."
            size_mb = f.stat().st_size / (1024 * 1024)
            llm_models.append(ModelInfoResponse(
                name=f.stem,
                display_name=display_name,
                path=str(f),
                size_mb=round(size_mb, 1),
            ))

    # 扫描性别检测模型
    gender_models = [
        ModelInfoResponse(
            name="f0",
            display_name=_build_gender_display_name("f0", 0),
            path="",
            size_mb=None,
        )
    ]
    gender_dir = settings.paths.models_dir / "gender"
    if gender_dir.exists():
        for d in sorted(gender_dir.iterdir()):
            if d.is_dir() and (
                (d / "config.json").exists()
                or (d / "pytorch_model.bin").exists()
                or (d / "model.safetensors").exists()
                or any(d.glob("*.onnx"))
            ):
                size_mb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / (1024 * 1024)
                gender_models.append(ModelInfoResponse(
                    name=d.name,
                    display_name=_build_gender_display_name(d.name, size_mb),
                    path=str(d),
                    size_mb=round(size_mb, 1),
                ))

    return ModelsListResponse(
        asr_models=asr_models,
        whisper_models=whisper_models,
        llm_models=llm_models,
        gender_models=gender_models,
    )


@router.post("/models/upload/whisper")
async def upload_whisper_model(file: UploadFile = File(...)):
    """
    上传 Whisper 模型 (压缩包)

    支持: .zip, .tar.gz
    解压后放入 models/whisper/ 目录
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="缺少文件名")

    # 验证格式
    if not (file.filename.endswith(".zip") or file.filename.endswith(".tar.gz")):
        raise HTTPException(status_code=400, detail="只支持 .zip 或 .tar.gz 格式")

    settings = get_settings()
    whisper_dir = settings.paths.models_dir / "whisper"
    whisper_dir.mkdir(parents=True, exist_ok=True)

    # 保存临时文件
    temp_path = whisper_dir / file.filename
    try:
        content = await file.read()
        temp_path.write_bytes(content)

        # 解压
        model_name = file.filename.replace(".tar.gz", "").replace(".zip", "")
        extract_dir = whisper_dir / model_name

        if file.filename.endswith(".zip"):
            import zipfile
            with zipfile.ZipFile(temp_path, 'r') as zf:
                zf.extractall(extract_dir)
        else:
            import tarfile
            with tarfile.open(temp_path, 'r:gz') as tf:
                tf.extractall(extract_dir)

        temp_path.unlink()

        return {"status": "ok", "model_name": model_name, "path": str(extract_dir)}

    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise HTTPException(status_code=500, detail=f"上传失败: {e}")


@router.post("/models/upload/asr")
async def upload_asr_model(file: UploadFile = File(...)):
    """
    上传 ASR 模型 (压缩包)

    支持: .zip, .tar.gz
    解压后放入 models/asr/ 目录
    适用于 SenseVoice / Paraformer / FireRedASR 等非 Whisper 模型
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="缺少文件名")

    if not (file.filename.endswith(".zip") or file.filename.endswith(".tar.gz")):
        raise HTTPException(status_code=400, detail="只支持 .zip 或 .tar.gz 格式")

    settings = get_settings()
    asr_dir = settings.paths.models_dir / "asr"
    asr_dir.mkdir(parents=True, exist_ok=True)

    temp_path = asr_dir / file.filename
    try:
        content = await file.read()
        temp_path.write_bytes(content)

        model_name = file.filename.replace(".tar.gz", "").replace(".zip", "")
        extract_dir = asr_dir / model_name

        if file.filename.endswith(".zip"):
            import zipfile
            with zipfile.ZipFile(temp_path, 'r') as zf:
                zf.extractall(extract_dir)
        else:
            import tarfile
            with tarfile.open(temp_path, 'r:gz') as tf:
                tf.extractall(extract_dir)

        temp_path.unlink()
        return {"status": "ok", "model_name": model_name, "path": str(extract_dir)}

    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise HTTPException(status_code=500, detail=f"上传失败: {e}")


@router.delete("/models/asr/{model_name}")
async def delete_asr_model(model_name: str):
    """删除 ASR 模型（models/asr/ 目录下）"""
    settings = get_settings()
    model_path = settings.paths.models_dir / "asr" / model_name

    if not model_path.exists():
        raise HTTPException(status_code=404, detail="模型不存在")

    shutil.rmtree(model_path)
    return {"status": "ok", "message": f"已删除: {model_name}"}


@router.post("/models/upload/llm")
async def upload_llm_model(file: UploadFile = File(...)):
    """
    上传 LLM 模型 (GGUF 格式)

    直接保存到 models/llm/ 目录
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="缺少文件名")

    if not file.filename.endswith(".gguf"):
        raise HTTPException(status_code=400, detail="只支持 .gguf 格式")

    settings = get_settings()
    llm_dir = settings.paths.models_dir / "llm"
    llm_dir.mkdir(parents=True, exist_ok=True)

    model_path = llm_dir / file.filename
    try:
        content = await file.read()
        model_path.write_bytes(content)

        return {"status": "ok", "model_name": Path(file.filename).stem, "path": str(model_path)}

    except Exception as e:
        if model_path.exists():
            model_path.unlink()
        raise HTTPException(status_code=500, detail=f"上传失败: {e}")


@router.delete("/models/whisper/{model_name}")
async def delete_whisper_model(model_name: str):
    """删除 Whisper 模型"""
    settings = get_settings()

    # 尝试两种命名方式
    model_path = settings.paths.models_dir / "whisper" / model_name
    if not model_path.exists():
        model_path = settings.paths.models_dir / "whisper" / f"faster-whisper-{model_name}"

    if not model_path.exists():
        raise HTTPException(status_code=404, detail="模型不存在")

    shutil.rmtree(model_path)
    return {"status": "ok", "message": f"已删除: {model_name}"}


@router.delete("/models/llm/{model_name}")
async def delete_llm_model(model_name: str):
    """删除 LLM 模型"""
    settings = get_settings()
    model_path = settings.paths.models_dir / "llm" / f"{model_name}.gguf"

    if not model_path.exists():
        raise HTTPException(status_code=404, detail="模型不存在")

    model_path.unlink()
    return {"status": "ok", "message": f"已删除: {model_name}"}


@router.post("/models/upload/diarization")
async def upload_diarization_model(file: UploadFile = File(...)):
    """
    上传说话人分离模型 (压缩包)

    支持: .zip, .tar.gz
    解压后放入 models/diarization/ 目录
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="缺少文件名")

    if not (file.filename.endswith(".zip") or file.filename.endswith(".tar.gz")):
        raise HTTPException(status_code=400, detail="只支持 .zip 或 .tar.gz 格式")

    settings = get_settings()
    diar_dir = settings.paths.models_dir / "diarization"
    diar_dir.mkdir(parents=True, exist_ok=True)

    temp_path = diar_dir / file.filename
    try:
        content = await file.read()
        temp_path.write_bytes(content)

        model_name = file.filename.replace(".tar.gz", "").replace(".zip", "")
        extract_dir = diar_dir / model_name

        if file.filename.endswith(".zip"):
            import zipfile
            with zipfile.ZipFile(temp_path, 'r') as zf:
                zf.extractall(extract_dir)
        else:
            import tarfile
            with tarfile.open(temp_path, 'r:gz') as tf:
                tf.extractall(extract_dir)

        temp_path.unlink()
        return {"status": "ok", "model_name": model_name, "path": str(extract_dir)}

    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise HTTPException(status_code=500, detail=f"上传失败: {e}")


@router.post("/models/upload/gender")
async def upload_gender_model(file: UploadFile = File(...)):
    """
    上传性别检测模型 (压缩包)

    支持: .zip, .tar.gz
    解压后放入 models/gender/ 目录
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="缺少文件名")

    if not (file.filename.endswith(".zip") or file.filename.endswith(".tar.gz")):
        raise HTTPException(status_code=400, detail="只支持 .zip 或 .tar.gz 格式")

    settings = get_settings()
    gender_dir = settings.paths.models_dir / "gender"
    gender_dir.mkdir(parents=True, exist_ok=True)

    temp_path = gender_dir / file.filename
    try:
        content = await file.read()
        temp_path.write_bytes(content)

        model_name = file.filename.replace(".tar.gz", "").replace(".zip", "")
        extract_dir = gender_dir / model_name

        if file.filename.endswith(".zip"):
            import zipfile
            with zipfile.ZipFile(temp_path, 'r') as zf:
                zf.extractall(extract_dir)
        else:
            import tarfile
            with tarfile.open(temp_path, 'r:gz') as tf:
                tf.extractall(extract_dir)

        temp_path.unlink()
        return {"status": "ok", "model_name": model_name, "path": str(extract_dir)}

    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise HTTPException(status_code=500, detail=f"上传失败: {e}")


@router.delete("/models/diarization/{model_name}")
async def delete_diarization_model(model_name: str):
    """删除说话人分离模型"""
    settings = get_settings()
    model_path = settings.paths.models_dir / "diarization" / model_name

    if not model_path.exists():
        raise HTTPException(status_code=404, detail="模型不存在")

    shutil.rmtree(model_path)
    return {"status": "ok", "message": f"已删除: {model_name}"}


@router.delete("/models/gender/{model_name}")
async def delete_gender_model(model_name: str):
    """删除性别检测模型"""
    if model_name == "f0":
        raise HTTPException(status_code=400, detail="内置模型不可删除")

    settings = get_settings()
    model_path = settings.paths.models_dir / "gender" / model_name

    if not model_path.exists():
        raise HTTPException(status_code=404, detail="模型不存在")

    shutil.rmtree(model_path)
    return {"status": "ok", "message": f"已删除: {model_name}"}


@router.get("/system", response_model=SystemInfoResponse)
async def get_system_info():
    """获取系统信息"""
    settings = get_settings()

    cuda_available = False
    cuda_version = None
    gpu_name = None

    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_version = torch.version.cuda
            gpu_name = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    return SystemInfoResponse(
        version="0.6.0",
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        gpu_name=gpu_name,
        models_dir=str(settings.paths.models_dir),
        output_dir=str(settings.paths.output_dir),
    )


@router.get("/streaming-engines", response_model=StreamingEnginesListResponse)
async def get_streaming_engines():
    """获取可用的流式 ASR 引擎列表"""
    engines_data = list_available_engines()
    engines = [StreamingEngineResponse(**e) for e in engines_data]
    current = get_settings().streaming.asr_engine
    return StreamingEnginesListResponse(engines=engines, current=current)


@router.get("/audio-devices")
async def list_audio_devices():
    """
    获取可用的音频设备列表

    返回:
    - input_devices: 输入设备（麦克风）
    - loopback_devices: Loopback 设备（施工中，暂不可用）

    设备 ID 使用 sounddevice 的 ID
    """
    input_devices = []
    loopback_devices = []  # 施工中，暂时为空
    default_input = None
    error_message = None

    # 排除这些关键词的设备（虚拟设备、输出设备的虚拟输入等）
    EXCLUDE_KEYWORDS = [
        "loopback", "立体声混音", "stereo mix", "what u hear",
        "wave out", "映射器", "mapper", "microsoft sound mapper",
        # 输出设备相关
        "speakers", "扬声器", "headphones", "耳机", "hdmi", "displayport",
        "realtek digital output", "spdif", "数字音频",
    ]

    try:
        import sounddevice as sd

        # 获取所有设备和 host API 信息
        devices = sd.query_devices()
        host_apis = sd.query_hostapis()
        default_input_id = sd.default.device[0]  # 默认输入设备 ID

        for i, device in enumerate(devices):
            # 只获取有输入通道的设备
            if device["max_input_channels"] <= 0:
                continue

            device_name = device["name"].lower()

            # 排除包含特定关键词的设备
            if any(kw in device_name for kw in EXCLUDE_KEYWORDS):
                continue

            # 获取 host API 名称
            host_api_name = host_apis[device["hostapi"]]["name"].lower()

            # 只使用 MME 或 Windows WASAPI 的设备（排除 Windows DirectSound 等可能有问题的 API）
            # MME 是最兼容的，WASAPI 是推荐的
            if "mme" not in host_api_name and "wasapi" not in host_api_name:
                continue

            # 优先显示 WASAPI 设备，因为它们延迟更低
            # 但同一个物理设备可能同时出现在 MME 和 WASAPI 中，需要去重
            # 简单处理：直接都添加，让用户选择

            device_info = {
                "id": i,
                "name": device["name"],
                "channels": device["max_input_channels"],
                "sample_rate": int(device["default_samplerate"]),
                "is_loopback": False,
                "host_api": host_apis[device["hostapi"]]["name"],
            }
            input_devices.append(device_info)

            if i == default_input_id:
                default_input = device["name"]

    except ImportError:
        error_message = "sounddevice 未安装，请运行: pip install sounddevice"
        print(f"警告: {error_message}")

    except Exception as e:
        error_message = f"获取音频设备失败: {e}"
        print(f"sounddevice error: {e}")

    return {
        "input_devices": input_devices,
        "loopback_devices": loopback_devices,
        "default_input": default_input,
        "default_output": None,
        "loopback_available": False,  # 施工中
        "error": error_message,
    }
