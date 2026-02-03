"""
命令行接口

使用 typer 构建，提供：
- 子命令结构（transcribe, diarize, process, serve）
- 丰富的帮助信息
- 进度显示
- 配置覆盖
"""

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.panel import Panel
from rich.table import Table

from .config import get_settings
from .logger import console, print_error, print_info, print_step, print_success, print_warning, setup_logging

# 创建主应用
app = typer.Typer(
    name="meeting-ai",
    help="🎙️ Meeting AI - 离线会议纪要工具",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


# ============================================================
# 回调函数（全局选项）
# ============================================================
@app.callback()
def main(
    debug: Annotated[
        bool,
        typer.Option("--debug", "-d", help="启用调试模式"),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="安静模式，只输出结果"),
    ] = False,
) -> None:
    """
    🎙️ Meeting AI - 本地离线会议纪要工具

    功能：
    - 语音转写（ASR）
    - 说话人分离（Diarization）
    - 智能命名
    - 会议总结

    使用示例：
        meeting-ai process audio.mp3
        meeting-ai transcribe audio.mp3 --model large-v3
        meeting-ai serve --port 8080
    """
    settings = get_settings()

    if debug:
        settings.debug = True
        settings.log_level = "DEBUG"

    if not quiet:
        setup_logging(
            level=settings.log_level,
            show_path=debug,
        )


# ============================================================
# info 命令
# ============================================================
@app.command()
def info() -> None:
    """显示系统信息和配置"""
    settings = get_settings()

    # 标题面板
    console.print(
        Panel.fit(
            f"[bold blue]{settings.app_name}[/bold blue] v{settings.version}",
            subtitle="本地离线会议纪要工具",
        )
    )

    # 配置表格
    table = Table(title="当前配置", show_header=True)
    table.add_column("配置项", style="cyan")
    table.add_column("值", style="green")

    # 路径配置
    table.add_row("数据目录", str(settings.paths.data_dir))
    table.add_row("模型目录", str(settings.paths.models_dir))
    table.add_row("输出目录", str(settings.paths.output_dir))

    # ASR 配置
    table.add_row("ASR 模型", settings.asr.model_name)
    table.add_row("ASR 设备", settings.asr.device)
    table.add_row("计算精度", settings.asr.compute_type)

    # Diarization 配置
    table.add_row("分离模型", settings.diarization.model_name)
    hf_status = "✓ 已配置" if settings.diarization.hf_token else "✗ 未配置"
    table.add_row("HF Token", hf_status)

    # LLM 配置
    llm_status = "✓ 启用" if settings.llm.enabled else "✗ 禁用"
    table.add_row("LLM 命名", llm_status)
    if settings.llm.model_path:
        table.add_row("LLM 模型", str(settings.llm.model_path))

    console.print(table)

    # 检查依赖
    console.print("\n[bold]依赖检查:[/bold]")
    _check_dependencies()


def _check_dependencies() -> None:
    """检查关键依赖是否可用"""
    checks = [
        ("faster-whisper", "faster_whisper"),
        ("pyannote-audio", "pyannote.audio"),
        ("torch", "torch"),
        ("llama-cpp-python", "llama_cpp"),
    ]

    for name, module in checks:
        try:
            __import__(module)
            print_success(f"{name}")
        except ImportError:
            print_error(f"{name} [dim](未安装)[/dim]")


# ============================================================
# transcribe 命令（阶段 2）
# ============================================================
@app.command()
def transcribe(
    audio_file: Annotated[
        Path,
        typer.Argument(
            help="音频文件路径",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Whisper 模型 (tiny/base/small/medium/large-v3)"),
    ] = "small",
    language: Annotated[
        Optional[str],
        typer.Option("--language", "-l", help="指定语言（如 zh, en），默认自动检测"),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="输出目录"),
    ] = None,
) -> None:
    """
    转写音频文件

    将音频文件转换为文字（不包含说话人信息）。

    示例：
        meeting-ai transcribe meeting.mp3
        meeting-ai transcribe meeting.mp3 --model large-v3 --language zh
    """
    import json
    from datetime import datetime

    from rich.progress import Progress, SpinnerColumn, TextColumn

    from .services import get_asr_service
    from .utils.audio import ensure_wav_16k_mono, get_audio_info

    settings = get_settings()
    settings.ensure_dirs()

    # 临时修改 ASR 模型设置
    settings.asr.model_name = model

    # 确定输出目录
    if output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = settings.paths.output_dir / f"{audio_file.stem}_{timestamp}"
    output.mkdir(parents=True, exist_ok=True)

    print_info(f"输入文件: {audio_file}")
    print_info(f"输出目录: {output}")
    print_info(f"模型: {model}")

    # 获取音频信息
    audio_info = get_audio_info(audio_file)
    print_info(f"音频时长: {audio_info['duration']:.1f} 秒")

    # 转换音频格式
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("转换音频格式...", total=None)
        wav_path = ensure_wav_16k_mono(audio_file, output / "audio_16k.wav")

    # 运行转写
    print_step(1, 2, "加载模型...")
    service = get_asr_service()

    print_step(2, 2, "转写音频...")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("转写中（可能需要几分钟）...", total=None)
        result = service.transcribe(wav_path, language=language)

    # 保存结果
    # 1. JSON 格式
    (output / "transcript.json").write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )

    # 2. 纯文本
    (output / "transcript.txt").write_text(
        result.full_text,
        encoding="utf-8",
    )

    # 3. 带时间戳的文本
    lines = []
    for seg in result.segments:
        lines.append(f"[{seg.format_time()}] {seg.text}")
    (output / "transcript_with_time.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    # 显示结果
    print_success("转写完成！")
    console.print()
    print_info(f"语言: {result.language} (置信度: {result.language_probability:.2%})")
    print_info(f"片段数: {len(result.segments)}")
    print_info(f"结果已保存到: {output}")


# ============================================================
# diarize 命令（阶段 1-2）
# ============================================================
@app.command()
def diarize(
    audio_file: Annotated[
        Path,
        typer.Argument(
            help="音频文件路径",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="输出目录（默认: outputs/<文件名>/）"),
    ] = None,
    min_speakers: Annotated[
        Optional[int],
        typer.Option("--min-speakers", help="最少说话人数"),
    ] = None,
    max_speakers: Annotated[
        Optional[int],
        typer.Option("--max-speakers", help="最多说话人数"),
    ] = None,
) -> None:
    """
    说话人分离

    分析音频中的不同说话人，输出每个人说话的时间段。

    示例：
        meeting-ai diarize meeting.mp3
        meeting-ai diarize meeting.wav --max-speakers 3
    """
    import json
    from datetime import datetime

    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table

    from .services import get_diarization_service
    from .utils.audio import ensure_wav_16k_mono, get_audio_info

    settings = get_settings()
    settings.ensure_dirs()

    # 确定输出目录
    if output is None:
        # 默认: outputs/<文件名_时间戳>/
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = settings.paths.output_dir / f"{audio_file.stem}_{timestamp}"
    output.mkdir(parents=True, exist_ok=True)

    print_info(f"输入文件: {audio_file}")
    print_info(f"输出目录: {output}")

    # 获取音频信息
    audio_info = get_audio_info(audio_file)
    print_info(f"音频时长: {audio_info['duration']:.1f} 秒")

    # 转换音频格式（pyannote 需要 16kHz 单声道）
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("转换音频格式...", total=None)
        wav_path = ensure_wav_16k_mono(audio_file, output / "audio_16k.wav")

    # 运行说话人分离
    print_step(1, 2, "加载模型...")
    service = get_diarization_service()

    print_step(2, 2, "分析说话人...")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("说话人分离中（可能需要几分钟）...", total=None)
        result = service.diarize(
            wav_path,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

    # 保存结果
    # 1. 保存完整的 JSON
    result_json = output / "diarization.json"
    result_json.write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )

    # 2. 保存易读的文本格式
    result_txt = output / "diarization.txt"
    lines = [
        f"说话人分离结果",
        f"=" * 40,
        f"音频文件: {audio_file.name}",
        f"检测到 {result.speaker_count} 个说话人",
        f"共 {len(result.segments)} 个片段",
        f"",
        f"时间轴:",
        f"-" * 40,
    ]
    for seg in result.segments:
        lines.append(f"[{seg.format_time()}] {seg.speaker}")
    
    lines.extend([
        f"",
        f"说话人统计:",
        f"-" * 40,
    ])
    for speaker_id, info in result.speakers.items():
        lines.append(
            f"{speaker_id}: {info.segment_count} 个片段, "
            f"共 {info.total_duration:.1f} 秒"
        )
    
    result_txt.write_text("\n".join(lines), encoding="utf-8")

    # 显示结果
    print_success(f"说话人分离完成！")
    console.print()

    # 显示说话人统计表格
    table = Table(title="说话人统计")
    table.add_column("说话人", style="cyan")
    table.add_column("片段数", justify="right")
    table.add_column("总时长", justify="right")
    table.add_column("占比", justify="right")

    total_duration = sum(s.total_duration for s in result.speakers.values())
    for speaker_id, info in sorted(result.speakers.items()):
        percentage = (info.total_duration / total_duration * 100) if total_duration > 0 else 0
        table.add_row(
            speaker_id,
            str(info.segment_count),
            f"{info.total_duration:.1f}s",
            f"{percentage:.1f}%",
        )

    console.print(table)
    console.print()
    print_info(f"结果已保存到: {output}")


# ============================================================
# process 命令（完整流程）
# ============================================================
@app.command()
def process(
    audio_file: Annotated[
        Path,
        typer.Argument(
            help="音频文件路径",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="输出目录"),
    ] = None,
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Whisper 模型"),
    ] = "small",
    language: Annotated[
        Optional[str],
        typer.Option("--language", "-l", help="指定语言"),
    ] = None,
    no_diarize: Annotated[
        bool,
        typer.Option("--no-diarize", help="跳过说话人分离"),
    ] = False,
    no_naming: Annotated[
        bool,
        typer.Option("--no-naming", help="跳过智能命名"),
    ] = False,
    no_correct: Annotated[
        bool,
        typer.Option("--no-correct", help="跳过错别字校正"),
    ] = False,
    no_summary: Annotated[
        bool,
        typer.Option("--no-summary", help="跳过会议总结"),
    ] = False,
    enhance: Annotated[
        bool,
        typer.Option("--enhance", "-e", help="启用音频增强（降噪+归一化）"),
    ] = False,
    deep_enhance: Annotated[
        bool,
        typer.Option("--deep-enhance", help="深度音频增强（效果更好但更慢）"),
    ] = False,
    separate_voice: Annotated[
        bool,
        typer.Option("--separate-voice", help="分离人声（去除背景音乐/音效）"),
    ] = False,
    max_speakers: Annotated[
        Optional[int],
        typer.Option("--max-speakers", help="最多说话人数"),
    ] = None,
) -> None:
    """
    完整处理流程

    执行完整的会议纪要生成流程：
    0. 音频增强（可选）- 降噪、去混响、归一化
    1. 说话人分离（可选）
    2. 语音转写
    3. 错别字校正（可选）
    4. 对齐说话人和文字
    5. 智能命名（可选）
    6. 会议总结（可选）

    示例：
        meeting-ai process meeting.mp3
        meeting-ai process meeting.mp3 --model large-v3 --max-speakers 3
        meeting-ai process meeting.mp3 --no-diarize  # 只转写，不分离说话人
        meeting-ai process meeting.mp3 --no-naming   # 跳过智能命名
        meeting-ai process meeting.mp3 --no-correct  # 跳过错别字校正
        meeting-ai process meeting.mp3 --no-summary  # 跳过会议总结
        meeting-ai process meeting.mp3 --enhance     # 启用音频增强
        meeting-ai process meeting.mp3 --deep-enhance  # 深度增强（更慢但更好）
        meeting-ai process meeting.mp3 --separate-voice  # 分离人声（适合有背景音乐的视频）
    """
    import json
    from datetime import datetime

    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table

    from .services import (
        align_transcript_with_speakers,
        correct_segments,
        detect_all_genders,
        fix_unknown_speakers,
        format_summary_markdown,
        get_asr_service,
        get_diarization_service,
        get_naming_service,
        merge_adjacent_segments,
        summarize_meeting,
    )
    from .models import DiarizationResult, SpeakerInfo
    from .utils.audio import ensure_wav_16k_mono, get_audio_info

    settings = get_settings()
    settings.ensure_dirs()
    settings.asr.model_name = model

    # 计算总步骤数
    total_steps = 2  # 转换 + 转写
    if enhance or deep_enhance or separate_voice:
        total_steps += 1  # 增强
    if not no_correct and settings.llm.enabled:
        total_steps += 1  # 校正
    if not no_diarize:
        total_steps += 2  # 分离 + 对齐
    if not no_diarize and not no_naming:
        total_steps += 1  # 命名
    if not no_summary and settings.llm.enabled:
        total_steps += 1  # 总结
    current_step = 0

    # 确定输出目录
    if output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = settings.paths.output_dir / f"{audio_file.stem}_{timestamp}"
    output.mkdir(parents=True, exist_ok=True)

    print_info(f"输入文件: {audio_file}")
    print_info(f"输出目录: {output}")

    # 获取音频信息
    audio_info = get_audio_info(audio_file)
    print_info(f"音频时长: {audio_info['duration']:.1f} 秒")

    # Step 1: 转换音频格式
    console.print()
    current_step += 1
    print_step(current_step, total_steps, "转换音频格式...")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("转换中...", total=None)
        wav_path = ensure_wav_16k_mono(audio_file, output / "audio_16k.wav")

    # Step 1.5: 音频增强（可选）
    if enhance or deep_enhance or separate_voice:
        current_step += 1
        print_step(current_step, total_steps, "音频增强...")
        
        try:
            from .utils.enhance import enhance_audio
            
            enhanced_path = output / "audio_enhanced.wav"
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task_desc = []
                if separate_voice:
                    task_desc.append("分离人声")
                if deep_enhance:
                    task_desc.append("深度降噪")
                elif enhance:
                    task_desc.append("降噪")
                task_desc.append("归一化")
                progress.add_task(f"处理中（{'+'.join(task_desc)}）...", total=None)
                
                wav_path = enhance_audio(
                    wav_path,
                    enhanced_path,
                    denoise=enhance or deep_enhance,
                    normalize=True,
                    separate_voice=separate_voice,
                    deep_denoise=deep_enhance,
                )
            print_info("音频增强完成")
        except ImportError as e:
            print_warning(f"音频增强需要额外依赖: {e}")
            print_warning("请运行: pip install noisereduce")
            print_warning("如需深度降噪: pip install deepfilternet")
            print_warning("如需人声分离: pip install demucs")

    # Step 2: 说话人分离（可选）
    diarization_result = None
    if not no_diarize:
        current_step += 1
        print_step(current_step, total_steps, "说话人分离...")
        diar_service = get_diarization_service()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("分离中（可能需要几分钟）...", total=None)
            diarization_result = diar_service.diarize(
                wav_path,
                max_speakers=max_speakers,
            )
        print_info(f"检测到 {diarization_result.speaker_count} 个说话人")

    # Step 3: 语音转写
    current_step += 1
    print_step(current_step, total_steps, "语音转写...")
    asr_service = get_asr_service()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("转写中（可能需要几分钟）...", total=None)
        transcript_result = asr_service.transcribe(wav_path, language=language)
    print_info(f"语言: {transcript_result.language}, 片段数: {len(transcript_result.segments)}")

    # Step 3.5: 错别字校正（可选）
    if not no_correct and settings.llm.enabled:
        current_step += 1
        print_step(current_step, total_steps, "错别字校正...")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("校正中...", total=None)
            corrected_segments = correct_segments(transcript_result.segments)
            # 更新 transcript_result 的片段
            transcript_result.segments = corrected_segments
        print_info("文本校正完成")

    # Step 4: 对齐（如果有说话人分离结果）
    final_speakers = {}
    if diarization_result is not None:
        current_step += 1
        print_step(current_step, total_steps, "对齐说话人...")
        aligned_segments = align_transcript_with_speakers(
            transcript_result,
            diarization_result,
        )
        # 修复 UNKNOWN 说话人
        fixed_segments = fix_unknown_speakers(aligned_segments)
        # 合并相邻同一说话人的片段
        final_segments = merge_adjacent_segments(fixed_segments)
        
        # Step 5: 智能命名（如果启用）
        if not no_naming:
            current_step += 1
            print_step(current_step, total_steps, "智能命名...")
            
            # 性别检测
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("分析说话人...", total=None)
                gender_map = detect_all_genders(wav_path, final_segments)
            
            # LLM 命名
            naming_service = get_naming_service()
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("推断说话人身份...", total=None)
                final_speakers = naming_service.name_speakers(final_segments, gender_map)
            
            # 更新片段中的说话人显示名
            speaker_id_to_name = {
                spk_id: info.display_name 
                for spk_id, info in final_speakers.items()
            }
        else:
            # 不命名，只统计
            speaker_stats: dict[str, dict] = {}
            for seg in final_segments:
                spk = seg.speaker or "UNKNOWN"
                if spk not in speaker_stats:
                    speaker_stats[spk] = {"total_duration": 0.0, "segment_count": 0}
                speaker_stats[spk]["total_duration"] += seg.duration
                speaker_stats[spk]["segment_count"] += 1
            
            for spk_id, stats in speaker_stats.items():
                final_speakers[spk_id] = SpeakerInfo(
                    id=spk_id,
                    display_name=spk_id,
                    total_duration=stats["total_duration"],
                    segment_count=stats["segment_count"],
                )
            speaker_id_to_name = {spk_id: spk_id for spk_id in final_speakers}
        
        final_diarization = DiarizationResult(
            speakers=final_speakers,
            segments=final_segments,
        )
    else:
        final_segments = transcript_result.segments
        final_diarization = None
        speaker_id_to_name = {}
        final_speakers = {}

    # Step 6: 会议总结（如果启用）
    meeting_summary = None
    if not no_summary and settings.llm.enabled and final_diarization is not None:
        current_step += 1
        print_step(current_step, total_steps, "生成会议总结...")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("总结中...", total=None)
            meeting_summary = summarize_meeting(
                final_segments,
                final_speakers,
                duration=audio_info['duration'],
            )
        if meeting_summary:
            print_info(f"会议主题: {meeting_summary.title}")

    # 保存结果
    console.print()
    print_info("保存结果...")

    # 1. 转写结果
    (output / "transcript.json").write_text(
        transcript_result.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (output / "transcript.txt").write_text(
        transcript_result.full_text,
        encoding="utf-8",
    )

    # 2. 最终对齐结果
    if final_diarization is not None:
        (output / "result.json").write_text(
            final_diarization.model_dump_json(indent=2),
            encoding="utf-8",
        )
        
        # 生成易读的对话格式（使用显示名）
        lines = []
        for seg in final_segments:
            speaker_id = seg.speaker or "UNKNOWN"
            display_name = speaker_id_to_name.get(speaker_id, speaker_id)
            lines.append(f"[{seg.format_time()}] {display_name}: {seg.text}")
        (output / "result.txt").write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

    # 3. 会议总结
    if meeting_summary is not None:
        # JSON 格式
        (output / "summary.json").write_text(
            meeting_summary.model_dump_json(indent=2),
            encoding="utf-8",
        )
        # Markdown 格式
        summary_md = format_summary_markdown(
            meeting_summary,
            final_speakers,
            duration=audio_info['duration'],
        )
        (output / "summary.md").write_text(summary_md, encoding="utf-8")

    # 显示结果
    console.print()
    print_success("处理完成！")
    console.print()

    # 显示统计表格
    if final_diarization is not None:
        table = Table(title="说话人统计")
        table.add_column("ID", style="dim")
        table.add_column("名称", style="cyan")
        table.add_column("片段数", justify="right")
        table.add_column("总时长", justify="right")
        table.add_column("占比", justify="right")

        total_duration = sum(s.total_duration for s in final_diarization.speakers.values())
        for speaker_id, info in sorted(final_diarization.speakers.items()):
            percentage = (info.total_duration / total_duration * 100) if total_duration > 0 else 0
            table.add_row(
                speaker_id,
                info.display_name,
                str(info.segment_count),
                f"{info.total_duration:.1f}s",
                f"{percentage:.1f}%",
            )
        console.print(table)
        console.print()

    # 显示前几个片段预览
    console.print("[bold]对话预览（前5条）:[/bold]")
    for seg in final_segments[:5]:
        speaker_id = seg.speaker or "UNKNOWN"
        display_name = speaker_id_to_name.get(speaker_id, speaker_id)
        text_preview = seg.text[:50] + "..." if len(seg.text) > 50 else seg.text
        console.print(f"  [cyan]{display_name}[/cyan]: {text_preview}")

    # 显示会议总结预览
    if meeting_summary is not None:
        console.print()
        console.print("[bold]会议总结:[/bold]")
        console.print(f"  [yellow]主题:[/yellow] {meeting_summary.title}")
        if meeting_summary.summary:
            summary_preview = meeting_summary.summary[:100]
            if len(meeting_summary.summary) > 100:
                summary_preview += "..."
            console.print(f"  [yellow]摘要:[/yellow] {summary_preview}")
        if meeting_summary.key_points:
            console.print(f"  [yellow]要点:[/yellow] {len(meeting_summary.key_points)} 条")

    console.print()
    print_info(f"结果已保存到: {output}")


# ============================================================
# serve 命令（阶段 5）
# ============================================================
@app.command()
def serve(
    host: Annotated[
        str,
        typer.Option("--host", "-h", help="服务地址"),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="服务端口"),
    ] = 8000,
    reload: Annotated[
        bool,
        typer.Option("--reload", "-r", help="开发模式（自动重载）"),
    ] = False,
) -> None:
    """
    启动 Web 服务

    启动本地 Web 界面，可通过浏览器访问。

    示例：
        meeting-ai serve
        meeting-ai serve --port 8080 --reload
    """
    print_info(f"启动服务: http://{host}:{port}")

    # TODO: 阶段 5 实现
    print_error("此功能尚未实现，将在阶段 5 完成")
    raise typer.Exit(1)


# ============================================================
# 入口点
# ============================================================
def main_entry() -> None:
    """包入口点"""
    app()


if __name__ == "__main__":
    main_entry()
