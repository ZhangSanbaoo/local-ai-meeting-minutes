"""
会议纪要 AI - Flet 桌面应用

适配 Flet 0.80+ 版本 API

使用方式:
    flet run src/meeting_ai/gui.py

或直接运行:
    python src/meeting_ai/gui.py
"""

import asyncio
import json
import threading
from pathlib import Path
from datetime import datetime
from typing import Callable

import flet as ft

try:
    import flet_audio as fta
    HAS_FLET_AUDIO = True
except ImportError:
    HAS_FLET_AUDIO = False

# 使用 zenity 文件对话框（WSL 兼容，UI 更好，中文支持好）
import subprocess

# 导入辅助函数：支持作为模块运行和直接运行两种方式
def _import_module(relative_name: str):
    """动态导入模块，支持相对和绝对导入"""
    try:
        import importlib
        return importlib.import_module(f".{relative_name}", "meeting_ai")
    except ImportError:
        import importlib
        return importlib.import_module(f"meeting_ai.{relative_name}")

def _zenity_pick_file(title: str, file_filter: str = "") -> str | None:
    """使用 zenity 选择文件"""
    cmd = ["zenity", "--file-selection", f"--title={title}"]
    if file_filter:
        cmd.append(f"--file-filter={file_filter}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None

def _zenity_save_file(title: str, default_name: str, file_filter: str = "") -> str | None:
    """使用 zenity 保存文件对话框"""
    cmd = ["zenity", "--file-selection", "--save", "--confirm-overwrite",
           f"--title={title}", f"--filename={default_name}"]
    if file_filter:
        cmd.append(f"--file-filter={file_filter}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


# ============================================================================
# 常量
# ============================================================================

WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
PRIMARY_COLOR = ft.Colors.BLUE_600
SECONDARY_COLOR = ft.Colors.BLUE_400


# ============================================================================
# 可复用组件
# ============================================================================

class SegmentCard(ft.Container):
    """单条对话卡片"""

    def __init__(
        self,
        segment_id: int,
        time_str: str,
        speaker: str,
        speaker_id: str,
        text: str,
        on_click: Callable | None = None,
        on_edit: Callable | None = None,
        on_speaker_click: Callable | None = None,
        is_playing: bool = False,
    ):
        self.segment_id = segment_id
        self.speaker = speaker
        self.speaker_id = speaker_id
        self.text = text
        self.on_click_callback = on_click
        self.on_edit_callback = on_edit
        self.on_speaker_click_callback = on_speaker_click
        self.is_playing = is_playing  # 保存播放状态

        # 播放指示器
        self.play_icon = ft.Icon(
            ft.Icons.PLAY_ARROW,
            size=16,
            color=PRIMARY_COLOR if is_playing else ft.Colors.TRANSPARENT,
        )

        # 时间标签
        self.time_label = ft.Text(
            time_str,
            size=11,
            color=ft.Colors.GREY_500,
            width=55,
        )

        # 说话人标签（可点击重命名）
        self.speaker_label = ft.GestureDetector(
            content=ft.Container(
                content=ft.Text(
                    speaker,
                    size=12,
                    weight=ft.FontWeight.BOLD,
                    color=PRIMARY_COLOR,
                ),
                width=80,
                tooltip="点击重命名说话人",
            ),
            on_tap=self._on_speaker_tap,
            mouse_cursor=ft.MouseCursor.CLICK,
        )

        # 内容
        self.text_label = ft.Text(
            text,
            size=13,
            expand=True,
        )

        # 编辑按钮（始终可见）
        self.edit_btn = ft.IconButton(
            ft.Icons.EDIT,
            icon_size=16,
            icon_color=ft.Colors.GREY_400,
            tooltip="编辑对话内容",
            on_click=self._on_edit_click,
        )

        super().__init__(
            content=ft.Row([
                self.play_icon,
                self.time_label,
                self.speaker_label,
                self.text_label,
                self.edit_btn,
            ], spacing=10),
            padding=ft.Padding.symmetric(vertical=8, horizontal=12),
            bgcolor=ft.Colors.BLUE_100 if is_playing else ft.Colors.GREY_50,
            border_radius=8,
            on_click=self._on_click,
            on_hover=self._on_hover,
        )

    def _on_click(self, e):
        if self.on_click_callback:
            self.on_click_callback(self.segment_id)

    def _on_edit_click(self, e):
        if self.on_edit_callback:
            self.on_edit_callback(self.segment_id, self.speaker, self.text)

    def _on_speaker_tap(self, e):
        """点击说话人名字"""
        if self.on_speaker_click_callback:
            self.on_speaker_click_callback(self.speaker_id, self.speaker)

    def _on_hover(self, e):
        # 悬停时高亮背景（但播放中的保持蓝色）
        if self.is_playing:
            self.bgcolor = ft.Colors.BLUE_100  # 播放中更深的蓝色
        elif e.data == "true":
            self.bgcolor = ft.Colors.BLUE_50
        else:
            self.bgcolor = ft.Colors.GREY_50
        self.update()

    def set_playing(self, playing: bool):
        self.is_playing = playing
        self.play_icon.color = PRIMARY_COLOR if playing else ft.Colors.TRANSPARENT
        self.bgcolor = ft.Colors.BLUE_100 if playing else ft.Colors.GREY_50
        self.update()


class AudioPlayer(ft.Container):
    """音频播放器组件（使用 flet-audio 包）"""

    def __init__(self, on_seek: Callable | None = None, on_position_change: Callable | None = None):
        self.on_seek_callback = on_seek
        self.on_position_change = on_position_change
        self.duration_ms = 0
        self.current_position_ms = 0
        self.is_playing = False
        self.audio_path = None
        self._audio = None
        self._update_thread = None
        self._stop_flag = False

        # 播放/暂停按钮
        self.play_btn = ft.IconButton(
            ft.Icons.PLAY_ARROW,
            icon_size=24,
            icon_color=PRIMARY_COLOR,
            on_click=self._toggle_play,
            disabled=True,
        )

        # 进度条
        self.progress_slider = ft.Slider(
            min=0,
            max=100,
            value=0,
            expand=True,
            on_change_end=self._on_slider_change,
            active_color=PRIMARY_COLOR,
            disabled=True,
        )

        # 时间显示
        self.time_label = ft.Text(
            "00:00/00:00",
            size=11,
            color=ft.Colors.GREY_600,
            width=75,
        )

        # 状态标签
        status = "fta" if HAS_FLET_AUDIO else "无"
        self.status_label = ft.Text(
            f"[{status}]",
            size=10,
            color=ft.Colors.GREY_400 if HAS_FLET_AUDIO else ft.Colors.RED_400,
            width=35,
        )

        # 紧凑单行布局
        super().__init__(
            content=ft.Row([
                self.play_btn,
                self.progress_slider,
                self.time_label,
                self.status_label,
            ], spacing=5, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            padding=ft.Padding.symmetric(horizontal=8, vertical=4),
            border=ft.Border.all(1, ft.Colors.GREY_300),
            border_radius=6,
            height=40,
        )

    def load(self, audio_path: str | Path):
        """加载音频文件"""
        if not HAS_FLET_AUDIO:
            self.status_label.value = "需安装 flet-audio"
            self.status_label.color = ft.Colors.RED_400
            return

        self.stop()
        self.audio_path = str(audio_path)

        # 获取音频时长
        try:
            import soundfile as sf
            info = sf.info(self.audio_path)
            self.duration_ms = int(info.duration * 1000)
        except:
            self.duration_ms = 0

        self.current_position_ms = 0
        self.play_btn.icon = ft.Icons.PLAY_ARROW
        self.progress_slider.value = 0
        self._update_time_display()

        # 创建 flet_audio.Audio 组件 (Flet 0.80+ Audio 是 Service，不需要添加到 overlay)
        try:
            # 先释放旧的 audio
            if self._audio:
                try:
                    asyncio.create_task(self._audio.release())
                except:
                    pass
                self._audio = None

            self._audio = fta.Audio(
                src=self.audio_path,
                autoplay=False,
                volume=1.0,
            )

            self.play_btn.disabled = False
            self.progress_slider.disabled = False
            self.status_label.value = "▶"
            self.status_label.color = ft.Colors.GREEN_400
        except Exception as e:
            self._audio = None
            self.play_btn.disabled = True
            self.progress_slider.disabled = True
            self.status_label.value = "播放暂不可用"
            self.status_label.color = ft.Colors.ORANGE_400
            print(f"⚠ 音频播放器初始化失败: {e}")

    async def _async_play(self):
        """异步播放"""
        if self._audio:
            await self._audio.play()

    async def _async_pause(self):
        """异步暂停"""
        if self._audio:
            await self._audio.pause()

    async def _async_seek(self, position_ms: int):
        """异步跳转"""
        if self._audio:
            await self._audio.seek(position_ms)

    async def _async_get_position(self) -> int:
        """异步获取当前位置"""
        if self._audio:
            try:
                pos = await self._audio.get_current_position()
                return int(pos) if pos else 0
            except:
                return 0
        return 0

    def seek(self, position_ms: int):
        """跳转到指定位置（毫秒）"""
        self.current_position_ms = max(0, min(position_ms, self.duration_ms))
        if self._audio and self.page:
            self.page.run_task(self._async_seek, self.current_position_ms)
        self._update_time_display()

    def play(self):
        """播放"""
        if self._audio and self.audio_path and self.page:
            self.is_playing = True
            self.play_btn.icon = ft.Icons.PAUSE
            self._stop_flag = False
            self.page.run_task(self._async_play)
            self._start_update_loop()
            self.page.update()

    def pause(self):
        """暂停"""
        if self._audio and self.page:
            self.is_playing = False
            self.play_btn.icon = ft.Icons.PLAY_ARROW
            self._stop_flag = True
            self.page.run_task(self._async_pause)
            self.update()

    def stop(self):
        """停止"""
        self._stop_flag = True
        self.is_playing = False
        self.play_btn.icon = ft.Icons.PLAY_ARROW
        if self._audio and self.page:
            self.page.run_task(self._async_pause)
            self.page.run_task(self._async_seek, 0)
        self.current_position_ms = 0
        self._update_time_display()

    def _toggle_play(self, e):
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def _start_update_loop(self):
        """启动进度更新循环（Flet 0.80+ 使用异步方式）"""
        if self.page:
            self.page.run_task(self._async_update_loop)

    async def _async_update_loop(self):
        """异步进度更新循环"""
        while self.is_playing and not self._stop_flag:
            # 基于时间估算位置
            self.current_position_ms = min(
                self.current_position_ms + 200,
                self.duration_ms
            )

            if self.current_position_ms >= self.duration_ms:
                self.is_playing = False
                self.play_btn.icon = ft.Icons.PLAY_ARROW
                self.current_position_ms = 0
                self._update_time_display()
                break

            self._update_time_display()

            if self.on_position_change:
                self.on_position_change(self.current_position_ms)

            await asyncio.sleep(0.2)

    def _on_slider_change(self, e):
        if self.duration_ms > 0:
            position = int((e.control.value / 100) * self.duration_ms)
            self.seek(position)
            if self.on_seek_callback:
                self.on_seek_callback(position)

    def _update_time_display(self):
        if self.duration_ms > 0:
            self.progress_slider.value = (self.current_position_ms / self.duration_ms) * 100
        current = self._format_time(self.current_position_ms)
        total = self._format_time(self.duration_ms)
        self.time_label.value = f"{current}/{total}"
        # 使用 page.update() 确保后台线程能触发 UI 刷新
        if self.page:
            self.page.update()

    @staticmethod
    def _format_time(ms: int) -> str:
        seconds = ms // 1000
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes:02d}:{secs:02d}"


class PanelDivider(ft.Container):
    """面板分隔线（仅视觉，不可拖动）"""

    def __init__(self):
        super().__init__(
            width=6,
            bgcolor=ft.Colors.GREY_200,
            border_radius=3,
        )


class ResizableDivider(ft.GestureDetector):
    """可拖动的分隔线"""

    def __init__(self, on_resize: Callable[[float], None]):
        self.on_resize_callback = on_resize
        super().__init__(
            content=ft.Container(
                width=8,
                bgcolor=ft.Colors.GREY_200,
                border_radius=4,
            ),
            on_pan_update=self._on_drag,
            on_enter=self._on_enter,
            on_exit=self._on_exit,
            mouse_cursor=ft.MouseCursor.RESIZE_COLUMN,
        )

    def _on_drag(self, e: ft.DragUpdateEvent):
        # 调用回调，传递水平移动距离
        if self.on_resize_callback:
            delta_x = e.delta_x if hasattr(e, 'delta_x') else e.local_delta.x
            self.on_resize_callback(delta_x)

    def _on_enter(self, e):
        self.content.bgcolor = ft.Colors.BLUE_200
        self.update()

    def _on_exit(self, e):
        self.content.bgcolor = ft.Colors.GREY_200
        self.update()


# ============================================================================
# 主应用
# ============================================================================

class MeetingAIApp:
    """会议纪要 AI 桌面应用"""

    def __init__(self, page: ft.Page):
        self.page = page
        self.selected_file: Path | None = None
        self.processing = False
        self.result = None
        self.segments = []
        self.speakers = {}
        self.current_segment_id = -1
        self.edit_mode = False

        # 面板引用
        self.left_panel = None
        self.right_panel = None

        # 配置页面
        self._setup_page()

        # 创建组件
        self._create_components()

        # 构建 UI
        self._build_ui()

        # 加载历史记录
        self._load_history()

    def _setup_page(self):
        """配置页面"""
        self.page.title = "会议纪要 AI"
        self.page.window.width = WINDOW_WIDTH
        self.page.window.height = WINDOW_HEIGHT
        self.page.window.min_width = 800
        self.page.window.min_height = 600
        self.page.padding = 0
        self.page.theme_mode = ft.ThemeMode.LIGHT

        # 尝试加载 emoji 字体
        emoji_fonts = {}
        emoji_font_paths = [
            "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
            "/usr/share/fonts/google-noto-emoji/NotoColorEmoji.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # 有基础 emoji
        ]
        for path in emoji_font_paths:
            if Path(path).exists():
                emoji_fonts["EmojiFont"] = path
                break

        if emoji_fonts:
            self.page.fonts = emoji_fonts

        # 设置支持中文和 emoji 的字体
        self.page.theme = ft.Theme(
            color_scheme_seed=PRIMARY_COLOR,
            font_family="Noto Sans SC, EmojiFont, DejaVu Sans, Microsoft YaHei, SimHei, sans-serif",
        )

    def _get_current_whisper_model(self) -> str:
        """获取当前配置的 Whisper 模型"""
        try:
            config = _import_module("config")
            return config.get_settings().asr.model_name
        except:
            return "medium"

    def _scan_whisper_models(self) -> list:
        """扫描已安装的 Whisper 模型"""
        options = []
        try:
            config = _import_module("config")
            whisper_dir = config.get_settings().paths.models_dir / "whisper"
            if whisper_dir.exists():
                for d in sorted(whisper_dir.iterdir()):
                    if d.is_dir() and d.name.startswith("faster-whisper-"):
                        model_name = d.name.replace("faster-whisper-", "")
                        display = model_name
                        if model_name == "small":
                            display = "small (快速)"
                        elif model_name == "medium":
                            display = "medium (推荐)"
                        elif "large" in model_name:
                            display = f"{model_name} (高精度)"
                        options.append(ft.dropdown.Option(model_name, display))
        except Exception:
            # 如果扫描失败，提供默认选项
            options = [
                ft.dropdown.Option("small", "small"),
                ft.dropdown.Option("medium", "medium (推荐)"),
            ]
        return options if options else [ft.dropdown.Option("medium", "medium")]

    def _scan_llm_models(self) -> list:
        """扫描已安装的 LLM 模型"""
        options = [ft.dropdown.Option("disabled", "不使用 LLM")]
        try:
            config = _import_module("config")
            llm_dir = config.get_settings().paths.models_dir / "llm"
            if llm_dir.exists():
                for f in sorted(llm_dir.glob("*.gguf")):
                    # 简化显示名称
                    display_name = f.stem
                    if len(display_name) > 35:
                        display_name = display_name[:32] + "..."
                    options.append(ft.dropdown.Option(str(f), display_name))
        except Exception:
            pass
        return options

    def _create_components(self):
        """创建 UI 组件"""

        # 文件选择使用 tkinter（WSL 兼容），无需 Flet FilePicker

        # ===== 音频来源选择 =====
        self.source_radio = ft.RadioGroup(
            content=ft.Row([
                ft.Radio(value="new", label="选择新文件"),
                ft.Radio(value="history", label="历史记录"),
            ]),
            value="new",
            on_change=self._on_source_change,
        )

        # 文件选择按钮和显示
        self.file_btn = ft.Button(
            "浏览...",
            icon=ft.Icons.FOLDER_OPEN,
            on_click=self._pick_file_clicked,
        )
        self.file_text = ft.Text("未选择文件", size=13, color=ft.Colors.GREY_600)
        self.file_row = ft.Row([self.file_btn, self.file_text], spacing=15)

        # 历史记录下拉
        self.history_dropdown = ft.Dropdown(
            label="选择历史记录",
            width=400,
            on_select=self._on_history_change,
            visible=False,
        )

        # ===== 模型选择（自动检测已安装模型）=====
        self.whisper_dropdown = ft.Dropdown(
            label="Whisper 模型",
            width=180,
            options=self._scan_whisper_models(),
            value=self._get_current_whisper_model(),
        )

        self.llm_dropdown = ft.Dropdown(
            label="LLM 模型",
            width=280,
            options=self._scan_llm_models(),
        )
        # 设置默认值
        llm_options = self._scan_llm_models()
        if llm_options:
            self.llm_dropdown.value = llm_options[0].key

        # ===== 处理选项 =====
        self.cb_naming = ft.Checkbox(label="智能命名", value=True)
        self.cb_correction = ft.Checkbox(label="错别字校正", value=True)
        self.cb_summary = ft.Checkbox(label="会议总结", value=True)

        # 音频增强选项（下拉选择）
        self.enhance_dropdown = ft.Dropdown(
            label="音频增强",
            width=180,
            options=[
                ft.dropdown.Option("none", "不增强"),
                ft.dropdown.Option("simple", "普通降噪"),
                ft.dropdown.Option("deep", "深度降噪"),
                ft.dropdown.Option("ai", "AI人声分离"),
                ft.dropdown.Option("deep_ai", "深度降噪+AI"),
            ],
            value="none",
        )

        # ===== 进度 =====
        self.progress_bar = ft.ProgressBar(value=0, visible=False, expand=True)
        self.progress_text = ft.Text("", size=12, color=ft.Colors.GREY_600)

        # 开始按钮
        self.start_btn = ft.Button(
            "开始处理",
            icon=ft.Icons.PLAY_ARROW,
            on_click=self._on_start_clicked,
            style=ft.ButtonStyle(bgcolor=PRIMARY_COLOR, color=ft.Colors.WHITE),
            height=42,
        )

        # ===== 对话列表 =====
        self.segment_list = ft.ListView(expand=True, spacing=4, padding=10)

        # 编辑模式按钮
        self.edit_mode_btn = ft.IconButton(
            ft.Icons.EDIT,
            icon_color=ft.Colors.GREY_600,
            tooltip="编辑模式",
            on_click=self._toggle_edit_mode,
        )

        # ===== 音频播放器 =====
        self.audio_player = AudioPlayer(on_seek=self._on_audio_seek, on_position_change=self._on_audio_position_change)

        # ===== 总结显示 =====
        self.summary_text = ft.Markdown(
            "\n\n\n*处理音频后将在此显示会议总结...*\n\n\n",
            selectable=True,
            expand=True,
            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
        )

        # ===== 导出按钮 =====
        self.export_btns = ft.Row([
            ft.OutlinedButton("导出 TXT", icon=ft.Icons.DESCRIPTION,
                             on_click=lambda _: self._export("txt"), disabled=True),
            ft.OutlinedButton("导出 JSON", icon=ft.Icons.CODE,
                             on_click=lambda _: self._export("json"), disabled=True),
            ft.OutlinedButton("导出 Markdown", icon=ft.Icons.ARTICLE,
                             on_click=lambda _: self._export("md"), disabled=True),
        ], alignment=ft.MainAxisAlignment.CENTER, spacing=15)

        # ===== 编辑对话框 =====
        self.edit_dialog = self._create_edit_dialog()
        self.rename_dialog = self._create_rename_dialog()
        self.summary_edit_dialog = self._create_summary_edit_dialog()

    def _create_edit_dialog(self) -> ft.AlertDialog:
        """创建编辑对话框"""
        self.edit_speaker_field = ft.TextField(label="说话人", width=200)
        self.edit_text_field = ft.TextField(label="内容", multiline=True, min_lines=3, max_lines=6, width=400)
        self.edit_segment_id = -1

        return ft.AlertDialog(
            title=ft.Text("编辑对话"),
            content=ft.Column([
                self.edit_speaker_field,
                self.edit_text_field,
            ], tight=True, spacing=15),
            actions=[
                ft.TextButton("取消", on_click=self._close_edit_dialog),
                ft.Button("保存", on_click=self._save_edit),
            ],
        )

    def _create_rename_dialog(self) -> ft.AlertDialog:
        """创建全局重命名说话人对话框"""
        self.rename_speaker_id = ""
        self.rename_old_name = ft.Text("", size=14, color=ft.Colors.GREY_600)
        self.rename_field = ft.TextField(label="新名字", width=250)
        self.rename_count = ft.Text("", size=12, color=ft.Colors.GREY_500)

        return ft.AlertDialog(
            title=ft.Text("重命名说话人"),
            content=ft.Column([
                ft.Row([ft.Text("当前名字:"), self.rename_old_name]),
                self.rename_field,
                self.rename_count,
                ft.Text("修改后将替换所有该说话人的显示名", size=11, color=ft.Colors.ORANGE_700),
            ], tight=True, spacing=10),
            actions=[
                ft.TextButton("取消", on_click=self._close_rename_dialog),
                ft.Button("确认修改", on_click=self._apply_rename),
            ],
        )

    def _create_summary_edit_dialog(self) -> ft.AlertDialog:
        """创建会议总结编辑对话框"""
        self.summary_edit_field = ft.TextField(
            label="会议总结",
            multiline=True,
            min_lines=15,
            max_lines=20,
            width=500,
        )

        return ft.AlertDialog(
            title=ft.Text("编辑会议总结"),
            content=self.summary_edit_field,
            actions=[
                ft.TextButton("取消", on_click=self._close_summary_edit),
                ft.Button("保存", on_click=self._save_summary_edit),
            ],
        )

    def _pick_file_clicked(self, e):
        """处理文件选择（使用 zenity，WSL 兼容）"""
        file_path = _zenity_pick_file(
            "选择音频文件",
            "音频文件 | *.mp3 *.wav *.m4a *.flac *.ogg"
        )
        if file_path:
            self.selected_file = Path(file_path)
            self.file_text.value = f"[已选择] {self.selected_file.name}"
            self.file_text.color = ft.Colors.BLACK
        else:
            self.selected_file = None
            self.file_text.value = "未选择文件"
            self.file_text.color = ft.Colors.GREY_600
        self.page.update()

    def _save_file_clicked(self, fmt: str):
        """处理文件保存（使用 zenity，WSL 兼容）"""
        if not self.result:
            return

        ext_map = {"txt": "文本文件", "json": "JSON 文件", "md": "Markdown 文件"}
        file_filter = f"{ext_map.get(fmt, fmt.upper())} | *.{fmt}"
        save_path = _zenity_save_file(f"导出为 {fmt.upper()}", f"meeting_result.{fmt}", file_filter)

        if save_path:
            src = self.result["output_dir"] / {"txt": "result.txt", "json": "result.json", "md": "summary.md"}[fmt]
            try:
                if src.exists():
                    import shutil
                    shutil.copy(src, save_path)
                    self._show_snackbar(f"已导出: {save_path}")
                else:
                    self._show_snackbar("文件不存在", error=True)
            except Exception as ex:
                self._show_snackbar(f"导出失败: {ex}", error=True)

    def _build_ui(self):
        """构建 UI 布局"""

        # Tab 按钮（使用 Icon 替代 emoji）
        self.tab_realtime = ft.Container(
            content=ft.TextButton(
                content=ft.Row([
                    ft.Icon(ft.Icons.MIC, size=18),
                    ft.Text("实时录音"),
                ], spacing=5),
                on_click=lambda _: self._switch_tab(0),
            ),
            padding=ft.Padding.symmetric(horizontal=15, vertical=8),
            border_radius=8,
        )
        self.tab_file = ft.Container(
            content=ft.TextButton(
                content=ft.Row([
                    ft.Icon(ft.Icons.FOLDER_OPEN, size=18),
                    ft.Text("音频文件"),
                ], spacing=5),
                on_click=lambda _: self._switch_tab(1),
            ),
            padding=ft.Padding.symmetric(horizontal=15, vertical=8),
            bgcolor=ft.Colors.BLUE_50,
            border_radius=8,
        )

        # 当前 Tab 索引
        self.current_tab = 1

        # 内容区域
        self.content_area = ft.Container(expand=True)

        # 构建页面
        self.page.add(
            ft.Column([
                # 标题栏
                ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.RECORD_VOICE_OVER, size=28, color=PRIMARY_COLOR),
                        ft.Text("会议纪要 AI", size=22, weight=ft.FontWeight.BOLD),
                    ]),
                    padding=ft.Padding.symmetric(horizontal=20, vertical=15),
                    bgcolor=ft.Colors.WHITE,
                ),
                # Tab 切换栏
                ft.Container(
                    content=ft.Row([
                        self.tab_realtime,
                        self.tab_file,
                    ]),
                    padding=ft.Padding.symmetric(horizontal=15, vertical=5),
                    bgcolor=ft.Colors.GREY_100,
                ),
                # 内容区
                self.content_area,
            ], spacing=0, expand=True),
        )

        # 显示默认 Tab
        self._switch_tab(1)

    def _switch_tab(self, index: int):
        """切换 Tab"""
        self.current_tab = index

        # 更新 Tab 样式
        if index == 0:
            self.tab_realtime.bgcolor = ft.Colors.BLUE_50
            self.tab_file.bgcolor = ft.Colors.TRANSPARENT
        else:
            self.tab_realtime.bgcolor = ft.Colors.TRANSPARENT
            self.tab_file.bgcolor = ft.Colors.BLUE_50

        # 更新内容
        if index == 0:
            self.content_area.content = self._build_realtime_tab()
        else:
            self.content_area.content = self._build_file_tab()

        self.page.update()

    def _build_realtime_tab(self):
        """构建实时录音 Tab（预留）"""
        return ft.Container(
            content=ft.Column([
                ft.Container(height=80),
                ft.Icon(ft.Icons.MIC, size=80, color=ft.Colors.GREY_300),
                ft.Text("实时录音功能", size=28, color=ft.Colors.GREY_500, weight=ft.FontWeight.BOLD),
                ft.Text("即将推出，敬请期待...", size=16, color=ft.Colors.GREY_400),
                ft.Container(height=20),
                ft.OutlinedButton("返回音频文件", on_click=lambda _: self._switch_tab(1)),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            expand=True,
        )

    def _build_file_tab(self):
        """构建音频文件 Tab - 对话/总结在上，选项在下（紧凑布局）"""

        # ===== 上方：对话记录 + 会议总结 (占主要空间) =====
        # 左侧：对话记录 + 音频播放器（固定比例 6）
        self.left_panel = ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.CHAT, color=PRIMARY_COLOR, size=18),
                    ft.Text("对话记录", size=14, weight=ft.FontWeight.W_500),
                    ft.Container(expand=True),
                    self.edit_mode_btn,
                ], spacing=5),
                ft.Container(
                    content=self.segment_list,
                    border=ft.Border.all(1, ft.Colors.GREY_300),
                    border_radius=6,
                    expand=True,
                ),
                self.audio_player,
            ], spacing=6, expand=True),
            padding=8,
            expand=6,  # 固定比例 6:4
        )

        # 右侧：会议总结（固定比例 4）
        self.right_panel = ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.SUMMARIZE, color=ft.Colors.GREEN_600, size=18),
                    ft.Text("会议总结", size=14, weight=ft.FontWeight.W_500),
                    ft.Container(expand=True),
                    ft.IconButton(
                        ft.Icons.EDIT,
                        icon_size=16,
                        icon_color=ft.Colors.GREY_600,
                        tooltip="编辑总结",
                        on_click=self._edit_summary,
                    ),
                ], spacing=5),
                ft.Container(
                    content=ft.ListView([self.summary_text], expand=True, padding=0),
                    border=ft.Border.all(1, ft.Colors.GREY_300),
                    border_radius=6,
                    padding=10,
                    expand=True,
                    bgcolor=ft.Colors.WHITE,
                ),
            ], spacing=6, expand=True),
            padding=8,
            expand=4,  # 固定比例 6:4
        )

        # 简单分隔线（不可拖动）
        divider = ft.Container(width=6, bgcolor=ft.Colors.GREY_200, border_radius=3)

        # 上方主内容区
        self.panels_row = ft.Row([
            self.left_panel,
            divider,
            self.right_panel,
        ], expand=True, spacing=0)

        # ===== 下方：选项和控制区域（紧凑 2 行布局）=====
        bottom_section = ft.Container(
            content=ft.Column([
                # 第一行：文件选择 + 模型选择 + 开始按钮 + 进度
                ft.Row([
                    self.file_btn,
                    self.file_text,
                    self.history_dropdown,
                    ft.Container(width=10),
                    self.whisper_dropdown,
                    self.llm_dropdown,
                    ft.Container(expand=True),
                    self.start_btn,
                    self.progress_bar,
                    self.progress_text,
                ], spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER),

                # 第二行：选项 + 增强 + 导出
                ft.Row([
                    self.source_radio,
                    ft.Container(width=5),
                    self.cb_naming,
                    self.cb_correction,
                    self.cb_summary,
                    self.enhance_dropdown,
                    ft.Container(expand=True),
                    self.export_btns,
                ], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ], spacing=12),
            padding=ft.Padding.symmetric(horizontal=10, vertical=8),
            bgcolor=ft.Colors.GREY_100,
        )

        # 组装：上方内容 + 下方控制
        return ft.Column([
            self.panels_row,
            ft.Divider(height=1),
            bottom_section,
        ], expand=True, spacing=0)

    def _on_source_change(self, e):
        """音频来源切换"""
        is_history = e.control.value == "history"
        self.file_btn.visible = not is_history
        self.file_text.visible = not is_history
        self.history_dropdown.visible = is_history
        self.page.update()

    def _load_history(self):
        """加载历史记录"""
        try:
            config = _import_module("config")
            output_dir = config.get_settings().paths.output_dir

            if not output_dir.exists():
                return

            history = []
            for d in sorted(output_dir.iterdir(), reverse=True):
                if d.is_dir() and (d / "result.json").exists():
                    history.append(d.name)

            self.history_dropdown.options = [
                ft.dropdown.Option(h) for h in history[:20]
            ]
            self.page.update()
        except Exception:
            pass

    def _on_history_change(self, e):
        """历史记录选择变化"""
        if e.control.value:
            self._load_history_result(e.control.value)

    def _load_history_result(self, dir_name: str):
        """加载历史结果"""
        try:
            config = _import_module("config")
            output_dir = config.get_settings().paths.output_dir / dir_name

            result_file = output_dir / "result.json"
            if not result_file.exists():
                self._show_snackbar("结果文件不存在", error=True)
                return

            data = json.loads(result_file.read_text(encoding="utf-8"))

            self.segments = data.get("segments", [])
            self.speakers = data.get("speakers", {})
            self.result = {
                "segments": self.segments,
                "speakers": self.speakers,
                "output_dir": output_dir,
            }

            # 加载音频
            for af in ["audio_enhanced.wav", "audio_16k.wav"]:
                audio_path = output_dir / af
                if audio_path.exists():
                    self.audio_player.load(audio_path)
                    break

            self._display_segments()

            # 加载总结
            summary_file = output_dir / "summary.md"
            self.summary_text.value = summary_file.read_text(encoding="utf-8") if summary_file.exists() else "*暂无总结*"

            self._enable_exports()
            self._show_snackbar(f"已加载: {dir_name}")

        except Exception as ex:
            self._show_snackbar(f"加载失败: {ex}", error=True)

    def _on_start_clicked(self, e):
        """开始处理"""
        if self.processing:
            return

        if self.source_radio.value == "new":
            if not self.selected_file or not self.selected_file.exists():
                self._show_snackbar("请先选择音频文件", error=True)
                return
        else:
            if self.history_dropdown.value:
                return  # 已加载历史记录
            self._show_snackbar("请选择历史记录", error=True)
            return

        self.processing = True
        self.start_btn.disabled = True
        self.progress_bar.visible = True
        self.segment_list.controls.clear()
        self.summary_text.value = ""
        self._disable_exports()
        self.page.update()

        thread = threading.Thread(target=self._process_audio, daemon=True)
        thread.start()

    def _process_audio(self):
        """处理音频（后台线程）"""
        try:
            # 动态导入模块
            config = _import_module("config")
            audio_utils = _import_module("utils.audio")
            services = _import_module("services")
            alignment = _import_module("services.alignment")
            gender = _import_module("services.gender")
            naming = _import_module("services.naming")
            correction = _import_module("services.correction")
            summary = _import_module("services.summary")
            models = _import_module("models")

            get_settings = config.get_settings
            ensure_wav_16k_mono = audio_utils.ensure_wav_16k_mono
            get_diarization_service = services.get_diarization_service
            get_asr_service = services.get_asr_service
            align_transcript_with_speakers = alignment.align_transcript_with_speakers
            fix_unknown_speakers = alignment.fix_unknown_speakers
            merge_adjacent_segments = alignment.merge_adjacent_segments
            detect_all_genders = gender.detect_all_genders
            get_naming_service = naming.get_naming_service
            correct_segments = correction.correct_segments
            summarize_meeting = summary.summarize_meeting
            format_summary_markdown = summary.format_summary_markdown
            SpeakerInfo = models.SpeakerInfo

            settings = get_settings()

            # 使用用户选择的模型设置
            selected_whisper = self.whisper_dropdown.value
            selected_llm = self.llm_dropdown.value
            if selected_whisper:
                settings.asr.model_name = selected_whisper

            # 判断是否使用 LLM
            use_llm = selected_llm and selected_llm != "disabled"
            if use_llm:
                settings.llm.enabled = True
                settings.llm.model_path = Path(selected_llm)
            else:
                settings.llm.enabled = False

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = settings.paths.output_dir / f"{self.selected_file.stem}_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)

            self._update_progress(0.05, "转换音频格式...")
            wav_path = output_dir / "audio_16k.wav"
            ensure_wav_16k_mono(self.selected_file, wav_path)

            # 音频增强处理
            enhance_mode = self.enhance_dropdown.value
            if enhance_mode and enhance_mode != "none":
                self._update_progress(0.10, "音频增强...")
                try:
                    enhance_mod = _import_module("utils.enhance")
                    enhanced_path = output_dir / "audio_enhanced.wav"

                    # 根据选项设置增强参数
                    denoise = enhance_mode in ["simple", "deep", "deep_ai"]
                    deep_denoise = enhance_mode in ["deep", "deep_ai"]
                    separate_voice = enhance_mode in ["ai", "deep_ai"]

                    enhance_mod.enhance_audio(
                        wav_path, enhanced_path,
                        denoise=denoise,
                        normalize=True,
                        deep_denoise=deep_denoise,
                        separate_voice=separate_voice,
                    )
                    wav_path = enhanced_path
                except (ImportError, ModuleNotFoundError) as e:
                    print(f"音频增强模块加载失败: {e}")

            self._update_progress(0.15, "说话人分离...")
            diar_service = get_diarization_service()
            diar_result = diar_service.diarize(wav_path)

            self._update_progress(0.40, "语音转写...")
            asr_service = get_asr_service()
            asr_result = asr_service.transcribe(wav_path)

            self._update_progress(0.55, "对齐...")
            aligned_segments = align_transcript_with_speakers(asr_result, diar_result)
            fixed_segments = fix_unknown_speakers(aligned_segments)
            final_segments = merge_adjacent_segments(fixed_segments)

            if self.cb_correction.value and use_llm:
                self._update_progress(0.65, "错别字校正...")
                try:
                    final_segments = correct_segments(final_segments)
                except Exception:
                    pass

            speakers = {}
            if self.cb_naming.value:
                self._update_progress(0.75, "智能命名...")
                try:
                    gender_map = detect_all_genders(wav_path, final_segments)
                    naming_service = get_naming_service()
                    speakers = naming_service.name_speakers(final_segments, gender_map)
                except Exception:
                    pass

            # 如果没有命名，使用默认
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

            summary_md = ""
            if self.cb_summary.value and use_llm:
                self._update_progress(0.85, "生成总结...")
                try:
                    summary_obj = summarize_meeting(final_segments, speakers, duration=asr_result.duration)
                    if summary_obj:
                        summary_md = format_summary_markdown(summary_obj, speakers, duration=asr_result.duration)
                except Exception:
                    summary_md = "*总结生成失败*"

            self._update_progress(0.95, "保存结果...")

            self.segments = [s.model_dump() if hasattr(s, 'model_dump') else s for s in final_segments]
            self.speakers = {k: v.model_dump() if hasattr(v, 'model_dump') else v for k, v in speakers.items()}

            self.result = {
                "segments": self.segments,
                "speakers": self.speakers,
                "summary": summary_md,
                "output_dir": output_dir,
            }

            self._save_results(output_dir, self.segments, self.speakers, summary_md)
            self.audio_player.load(wav_path)

            self._update_progress(1.0, "完成!")
            self._display_segments()
            self.summary_text.value = summary_md if summary_md else "*未生成总结*"
            self._enable_exports()
            self._load_history()
            self._show_snackbar("处理完成！")

        except Exception as ex:
            import traceback
            traceback.print_exc()
            self._show_snackbar(f"处理失败: {ex}", error=True)
        finally:
            self.processing = False
            self.start_btn.disabled = False
            self.progress_bar.visible = False
            self.page.update()

    def _save_results(self, output_dir: Path, segments: list, speakers: dict, summary: str):
        """保存结果"""
        result_data = {"speakers": speakers, "segments": segments, "speaker_count": len(speakers)}
        (output_dir / "result.json").write_text(json.dumps(result_data, ensure_ascii=False, indent=2), encoding="utf-8")

        lines = []
        for seg in segments:
            sid = seg.get("speaker", "UNKNOWN")
            name = speakers.get(sid, {}).get("display_name", sid) if speakers else sid
            lines.append(f"[{self._format_time(seg.get('start', 0))}-{self._format_time(seg.get('end', 0))}] {name}: {seg.get('text', '')}")
        (output_dir / "result.txt").write_text("\n".join(lines), encoding="utf-8")

        if summary:
            (output_dir / "summary.md").write_text(summary, encoding="utf-8")

    def _display_segments(self):
        """显示对话列表"""
        self.segment_list.controls.clear()

        for i, seg in enumerate(self.segments):
            sid = seg.get("speaker", "UNKNOWN")
            name = self.speakers.get(sid, {}).get("display_name", sid) if self.speakers else sid

            self.segment_list.controls.append(SegmentCard(
                segment_id=i,
                time_str=self._format_time(seg.get("start", 0)),
                speaker=name,
                speaker_id=sid,
                text=seg.get("text", ""),
                on_click=self._on_segment_click,
                on_edit=self._on_segment_edit,
                on_speaker_click=self._on_speaker_click,
                is_playing=(i == self.current_segment_id),
            ))

        self.page.update()

    def _on_segment_click(self, segment_id: int):
        """点击对话条目"""
        if 0 <= segment_id < len(self.segments):
            self.current_segment_id = segment_id
            start_ms = int(self.segments[segment_id].get("start", 0) * 1000)
            self.audio_player.seek(start_ms)
            self.audio_player.play()
            self._display_segments()

    def _on_segment_edit(self, segment_id: int, speaker: str, text: str):
        """编辑对话"""
        self.edit_segment_id = segment_id
        self.edit_speaker_field.value = speaker
        self.edit_text_field.value = text
        self.page.show_dialog(self.edit_dialog)  # Flet 0.80.5 API

    def _close_edit_dialog(self, e):
        self.page.pop_dialog()  # Flet 0.80.5 API

    def _save_edit(self, e):
        """保存编辑"""
        if 0 <= self.edit_segment_id < len(self.segments):
            seg = self.segments[self.edit_segment_id]
            sid = seg.get("speaker", "UNKNOWN")
            if sid in self.speakers:
                self.speakers[sid]["display_name"] = self.edit_speaker_field.value
            seg["text"] = self.edit_text_field.value

            self.page.pop_dialog()  # Flet 0.80.5 API
            self._display_segments()

            if self.result and "output_dir" in self.result:
                self._save_results(self.result["output_dir"], self.segments, self.speakers, self.result.get("summary", ""))

            self._show_snackbar("已保存")

    def _toggle_edit_mode(self, e):
        self.edit_mode = not self.edit_mode
        self.edit_mode_btn.icon_color = PRIMARY_COLOR if self.edit_mode else ft.Colors.GREY_600
        self._show_snackbar("编辑模式: " + ("开启" if self.edit_mode else "关闭"))
        self.page.update()

    def _on_speaker_click(self, speaker_id: str, display_name: str):
        """点击说话人名字，弹出全局重命名对话框"""
        self.rename_speaker_id = speaker_id
        self.rename_old_name.value = display_name
        self.rename_field.value = display_name

        # 统计该说话人的出现次数
        count = sum(1 for seg in self.segments if seg.get("speaker") == speaker_id)
        self.rename_count.value = f"将影响 {count} 条对话记录"

        self.page.show_dialog(self.rename_dialog)

    def _close_rename_dialog(self, e):
        self.page.pop_dialog()

    def _apply_rename(self, e):
        """应用全局重命名"""
        new_name = self.rename_field.value.strip()
        if not new_name or not self.rename_speaker_id:
            return

        # 更新说话人信息
        if self.rename_speaker_id in self.speakers:
            self.speakers[self.rename_speaker_id]["display_name"] = new_name

        self.page.pop_dialog()
        self._display_segments()

        # 自动保存
        if self.result and "output_dir" in self.result:
            self._save_results(self.result["output_dir"], self.segments, self.speakers, self.result.get("summary", ""))

        self._show_snackbar(f"已重命名为 '{new_name}'")

    def _edit_summary(self, e):
        """编辑会议总结"""
        if not self.result:
            self._show_snackbar("暂无总结可编辑", error=True)
            return

        self.summary_edit_field.value = self.summary_text.value or ""
        self.page.show_dialog(self.summary_edit_dialog)

    def _close_summary_edit(self, e):
        self.page.pop_dialog()

    def _save_summary_edit(self, e):
        """保存会议总结编辑"""
        new_summary = self.summary_edit_field.value
        self.summary_text.value = new_summary

        if self.result:
            self.result["summary"] = new_summary

        self.page.pop_dialog()
        self.page.update()

        # 自动保存
        if self.result and "output_dir" in self.result:
            self._save_results(self.result["output_dir"], self.segments, self.speakers, new_summary)

        self._show_snackbar("总结已保存")

    def _on_audio_seek(self, position_ms: int):
        """音频跳转回调"""
        pos_sec = position_ms / 1000
        for i, seg in enumerate(self.segments):
            if seg.get("start", 0) <= pos_sec < seg.get("end", 0):
                if self.current_segment_id != i:
                    self.current_segment_id = i
                    self._display_segments()
                break

    def _on_audio_position_change(self, position_ms: int):
        """音频播放位置变化回调"""
        pos_sec = position_ms / 1000
        for i, seg in enumerate(self.segments):
            if seg.get("start", 0) <= pos_sec < seg.get("end", 0):
                if self.current_segment_id != i:
                    self.current_segment_id = i
                    # 只更新高亮，不重建整个列表
                    for j, card in enumerate(self.segment_list.controls):
                        if isinstance(card, SegmentCard):
                            card.set_playing(j == i)
                    self.page.update()
                break

    def _update_progress(self, value: float, text: str):
        self.progress_bar.value = value
        self.progress_text.value = text
        self.page.update()

    def _enable_exports(self):
        for btn in self.export_btns.controls:
            btn.disabled = False
        self.page.update()

    def _disable_exports(self):
        for btn in self.export_btns.controls:
            btn.disabled = True
        self.page.update()

    def _export(self, fmt: str):
        if not self.result:
            return
        self._save_file_clicked(fmt)

    def _show_snackbar(self, message: str, error: bool = False):
        snackbar = ft.SnackBar(
            content=ft.Text(message),
            bgcolor=ft.Colors.RED_400 if error else ft.Colors.GREEN_400,
        )
        self.page.show_dialog(snackbar)  # Flet 0.80.5 API

    @staticmethod
    def _format_time(seconds: float) -> str:
        return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


async def main(page: ft.Page):
    # 先设置窗口大小
    page.title = "会议纪要 AI"
    page.window.width = WINDOW_WIDTH
    page.window.height = WINDOW_HEIGHT
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.bgcolor = ft.Colors.WHITE

    # 显示加载界面
    loading = ft.Column([
        ft.ProgressRing(width=50, height=50, stroke_width=4),
        ft.Container(height=20),
        ft.Text("正在加载...", size=16, color=ft.Colors.GREY_600),
    ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)
    page.add(loading)

    # 等待 UI 渲染
    await asyncio.sleep(0.1)

    # 窗口居中
    try:
        await page.window.center()
    except Exception:
        pass

    # 创建主应用
    page.controls.clear()
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.horizontal_alignment = ft.CrossAxisAlignment.START
    MeetingAIApp(page)


if __name__ == "__main__":
    # 初始化日志系统，输出到终端
    try:
        logger_mod = _import_module("logger")
        logger_mod.setup_logging(level="INFO", show_path=False)
        print("✓ 日志系统已初始化，处理过程将输出到终端")
    except Exception as e:
        print(f"⚠ 日志初始化失败: {e}")

    ft.run(main)  # Flet 0.80+ 使用 run()
