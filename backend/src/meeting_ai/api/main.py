"""
Meeting AI - FastAPI 主应用

启动方式:
    cd backend
    uvicorn meeting_ai.api.main:app --reload --host 0.0.0.0 --port 8000

或者:
    python -m meeting_ai.api.main
"""

# 在所有导入之前抑制不影响功能的警告
import warnings
import logging
import os

# Python warnings
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", message=".*triton.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# 抑制 torch 内部日志
logging.getLogger("torch").setLevel(logging.ERROR)

# 设置环境变量抑制底层警告
os.environ.setdefault("TORCH_LOGS", "-all")

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# 添加 src 到路径
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from meeting_ai.config import get_settings
from meeting_ai.logger import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    setup_logging(level="INFO", show_path=False)
    settings = get_settings()

    # 确保输出目录存在
    settings.paths.output_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.data_dir.mkdir(parents=True, exist_ok=True)

    # 挂载静态文件目录（提供音频文件访问）
    if settings.paths.output_dir.exists():
        app.mount("/files", StaticFiles(directory=str(settings.paths.output_dir)), name="files")
        print(f"✓ 静态文件目录已挂载: {settings.paths.output_dir}")

    print(f"✓ Meeting AI API 启动")
    print(f"  - 模型目录: {settings.paths.models_dir}")
    print(f"  - 输出目录: {settings.paths.output_dir}")

    yield

    # 关闭时
    print("✓ Meeting AI API 关闭")


# 创建 FastAPI 应用
app = FastAPI(
    title="Meeting AI API",
    description="本地离线会议纪要 AI 后端服务",
    version="0.6.0",
    lifespan=lifespan,
)

# CORS 配置 - 允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite 开发服务器
        "http://localhost:3000",  # 其他前端端口
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 导入并注册路由
from meeting_ai.api.routes import process, history, models, realtime

app.include_router(process.router, prefix="/api", tags=["处理"])
app.include_router(history.router, prefix="/api", tags=["历史记录"])
app.include_router(models.router, prefix="/api", tags=["模型"])
# WebSocket 路由不需要 prefix（路径已在 endpoint 中定义）
app.include_router(realtime.router, tags=["实时录音"])


# 根路由
@app.get("/")
async def root():
    """API 根路由"""
    return {
        "name": "Meeting AI API",
        "version": "0.6.0",
        "docs": "/docs",
        "redoc": "/redoc",
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok"}


# 直接运行
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "meeting_ai.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
