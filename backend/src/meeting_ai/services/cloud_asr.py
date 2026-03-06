"""
云端 ASR 引擎 — 国内主流云端语音识别 API

支持的服务商:
  - cloud-tencent    腾讯云语音识别（SentenceRecognition，微信同款技术）
  - cloud-baidu      百度语音识别（Pro API，简单 OAuth 认证）
  - cloud-iflytek    讯飞语音识别（录音文件转写 lfasr v2，异步轮询）
  - cloud-bytedance  字节跳动/火山引擎语音识别（同步 REST）
  - cloud-ali        阿里云 DashScope 语音识别（paraformer-v2，异步任务）

所有引擎:
  - 无本地 GPU 依赖，API Key 存储在 backend/.env
  - 共用 Silero VAD 预分段（同本地引擎）
  - 返回标准 TranscriptResult（与本地引擎接口完全兼容）
  - 无时间戳时自动线性插值兜底

配置方式（backend/.env）:
  MEETING_AI_CLOUD_ASR__TENCENT_SECRET_ID=AKIDxxx
  MEETING_AI_CLOUD_ASR__TENCENT_SECRET_KEY=xxx
  MEETING_AI_CLOUD_ASR__BAIDU_API_KEY=xxx
  MEETING_AI_CLOUD_ASR__BAIDU_SECRET_KEY=xxx
  MEETING_AI_CLOUD_ASR__IFLYTEK_APP_ID=xxx
  MEETING_AI_CLOUD_ASR__IFLYTEK_API_KEY=xxx
  MEETING_AI_CLOUD_ASR__IFLYTEK_API_SECRET=xxx
  MEETING_AI_CLOUD_ASR__BYTEDANCE_APP_ID=xxx
  MEETING_AI_CLOUD_ASR__BYTEDANCE_ACCESS_TOKEN=xxx
  MEETING_AI_CLOUD_ASR__ALI_API_KEY=sk-xxx
"""

from __future__ import annotations

import base64
import gc
import hashlib
import hmac
import io
import json
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

import requests

from ..config import get_settings
from ..logger import get_logger
from ..models import CharTimestamp, Segment, TranscriptResult
from .asr import ASREngine, _run_vad

logger = get_logger("services.cloud_asr")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_wav_bytes_16k_mono(audio_array, sr: int) -> bytes:
    """将 numpy float32 音频转为 16kHz 单声道 16bit WAV bytes"""
    import numpy as np
    import soundfile as sf

    # 重采样到 16kHz
    if sr != 16000:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(16000, sr)
        audio_array = resample_poly(audio_array, 16000 // g, sr // g)
        sr = 16000

    # 转单声道
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)

    # 转 16bit PCM
    pcm16 = (np.clip(audio_array, -1.0, 1.0) * 32767).astype(np.int16)

    buf = io.BytesIO()
    sf.write(buf, pcm16, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _linear_char_timestamps(
    text: str, start: float, end: float,
) -> list[CharTimestamp]:
    """用线性插值生成字级时间戳（无 API 时间戳时的兜底）"""
    if not text:
        return []
    chars = list(text)
    dur = end - start
    char_dur = dur / len(chars) if chars else 0
    return [
        CharTimestamp(
            char=ch,
            start=round(start + i * char_dur, 3),
            end=round(start + (i + 1) * char_dur, 3),
        )
        for i, ch in enumerate(chars)
    ]


def _word_list_to_char_ts(
    word_list: list[dict],
    start_key: str,
    end_key: str,
    word_key: str,
    time_unit: str = "ms",  # "ms" or "s"
    time_offset: float = 0.0,
) -> list[CharTimestamp]:
    """把 API 返回的 word_list 转为字级时间戳"""
    char_ts: list[CharTimestamp] = []
    for item in word_list:
        word = item.get(word_key, "")
        t_start = float(item.get(start_key, 0))
        t_end = float(item.get(end_key, 0))
        if time_unit == "ms":
            t_start /= 1000.0
            t_end /= 1000.0
        t_start += time_offset
        t_end += time_offset

        chars = list(word)
        if not chars:
            continue
        char_dur = (t_end - t_start) / len(chars)
        for i, ch in enumerate(chars):
            char_ts.append(CharTimestamp(
                char=ch,
                start=round(t_start + i * char_dur, 3),
                end=round(t_start + (i + 1) * char_dur, 3),
            ))
    return char_ts


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class CloudASREngine(ASREngine, ABC):
    """云端 ASR 引擎基类 - 不依赖本地模型，通过 API 转写"""

    # 子类设置此属性以在日志中显示
    PROVIDER_NAME: str = "云端 ASR"

    # 伪 _model 属性，避免工厂函数误判为"引擎已卸载"
    _model: object = True

    def load(self, model_dir: Path) -> None:
        """云端引擎不加载本地模型"""
        pass

    def unload(self) -> None:
        """云端引擎无需释放"""
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """检查 API 凭据是否已配置"""
        ...

    def transcribe(
        self,
        audio_path: Path | str,
        language: str | None = None,
        *,
        skip_vad: bool = False,
    ) -> TranscriptResult:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        if not self.is_configured():
            raise RuntimeError(
                f"{self.PROVIDER_NAME} API 未配置，请在设置页面填写 API 密钥"
            )

        import soundfile as sf
        info = sf.info(str(audio_path))
        duration = info.frames / info.samplerate
        logger.info(f"[{self.PROVIDER_NAME}] 开始转写: {audio_path.name} ({duration:.1f}s)")

        return self._do_transcribe(audio_path, duration, language)

    @abstractmethod
    def _do_transcribe(
        self, audio_path: Path, duration: float, language: str | None,
    ) -> TranscriptResult:
        ...


# ---------------------------------------------------------------------------
# Segment-based base: VAD split → call API per segment
# ---------------------------------------------------------------------------

class _SegmentCloudEngine(CloudASREngine, ABC):
    """每个 VAD 片段独立调用 API 的基类"""

    def _do_transcribe(
        self, audio_path: Path, duration: float, language: str | None,
    ) -> TranscriptResult:
        import soundfile as sf
        import numpy as np

        # VAD 分段
        vad_segs = _run_vad(audio_path)
        if not vad_segs:
            vad_segs = [(0.0, min(duration, 58.0))]

        audio, sr = sf.read(str(audio_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        segments: list[Segment] = []
        all_char_ts: list[list[CharTimestamp]] = []
        seg_id = 0

        for vad_start, vad_end in vad_segs:
            s = int(vad_start * sr)
            e = int(vad_end * sr)
            chunk = audio[s:e]
            if len(chunk) < sr * 0.1:
                continue

            wav_bytes = _to_wav_bytes_16k_mono(chunk, sr)
            pcm_b64 = base64.b64encode(wav_bytes).decode()
            duration_ms = int((vad_end - vad_start) * 1000)

            try:
                text, char_ts = self._transcribe_chunk(
                    pcm_b64, wav_bytes, duration_ms, vad_start, vad_end, language,
                )
            except Exception as e:
                logger.warning(f"[{self.PROVIDER_NAME}] 段 {seg_id} 失败: {e}")
                continue

            if not text:
                continue

            segments.append(Segment(
                id=seg_id,
                start=round(vad_start, 3),
                end=round(vad_end, 3),
                text=text,
                speaker=None,
            ))
            all_char_ts.append(char_ts or _linear_char_timestamps(text, vad_start, vad_end))
            seg_id += 1

        logger.info(f"[{self.PROVIDER_NAME}] 转写完成: {len(segments)} 段")
        return TranscriptResult(
            segments=segments,
            char_timestamps=all_char_ts,
            duration=duration,
        )

    @abstractmethod
    def _transcribe_chunk(
        self,
        pcm_b64: str,
        wav_bytes: bytes,
        duration_ms: int,
        vad_start: float,
        vad_end: float,
        language: str | None,
    ) -> tuple[str, list[CharTimestamp]]:
        """转写单个片段，返回 (text, char_timestamps)"""
        ...


# ---------------------------------------------------------------------------
# Engine 1: 腾讯云 SentenceRecognition（微信同款，词级时间戳）
# ---------------------------------------------------------------------------

def _tencent_sign(
    secret_id: str,
    secret_key: str,
    payload_json: str,
    service: str = "asr",
    action: str = "SentenceRecognition",
    version: str = "2019-06-14",
    region: str = "ap-beijing",
) -> dict[str, str]:
    """生成腾讯云 TC3-HMAC-SHA256 请求 Header"""
    host = f"{service}.tencentcloudapi.com"
    timestamp = int(time.time())
    from datetime import datetime, timezone
    date = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")

    payload_bytes = payload_json.encode("utf-8")

    # 1. 规范请求
    ct = "application/json; charset=utf-8"
    canonical_headers = f"content-type:{ct}\nhost:{host}\nx-tc-action:{action.lower()}\n"
    signed_headers = "content-type;host;x-tc-action"
    hashed_payload = hashlib.sha256(payload_bytes).hexdigest()
    canonical_request = "\n".join([
        "POST", "/", "",
        canonical_headers, signed_headers, hashed_payload,
    ])

    # 2. 待签名字符串
    credential_scope = f"{date}/{service}/tc3_request"
    hashed_cr = hashlib.sha256(canonical_request.encode()).hexdigest()
    string_to_sign = f"TC3-HMAC-SHA256\n{timestamp}\n{credential_scope}\n{hashed_cr}"

    # 3. 计算签名
    def _hmac256(key: bytes, msg: str) -> bytes:
        return hmac.new(key, msg.encode(), hashlib.sha256).digest()

    secret_date = _hmac256(f"TC3{secret_key}".encode(), date)
    secret_svc = _hmac256(secret_date, service)
    secret_signing = _hmac256(secret_svc, "tc3_request")
    signature = _hmac256(secret_signing, string_to_sign).hex()

    # 4. Authorization
    authorization = (
        f"TC3-HMAC-SHA256 "
        f"Credential={secret_id}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, "
        f"Signature={signature}"
    )
    return {
        "Authorization": authorization,
        "Content-Type": ct,
        "Host": host,
        "X-TC-Action": action,
        "X-TC-Version": version,
        "X-TC-Timestamp": str(timestamp),
        "X-TC-Region": region,
    }


class TencentASREngine(_SegmentCloudEngine):
    """腾讯云语音识别 — SentenceRecognition（极速版，支持词级时间戳）"""

    PROVIDER_NAME = "腾讯云 ASR"

    def is_configured(self) -> bool:
        cfg = get_settings().cloud_asr
        return bool(cfg.tencent_secret_id and cfg.tencent_secret_key)

    def _transcribe_chunk(
        self,
        pcm_b64: str,
        wav_bytes: bytes,
        duration_ms: int,
        vad_start: float,
        vad_end: float,
        language: str | None,
    ) -> tuple[str, list[CharTimestamp]]:
        cfg = get_settings().cloud_asr
        # 引擎类型：中文16k / 中文方言 / 英文等
        eng_type = "16k_zh"
        if language == "en":
            eng_type = "16k_en"
        elif language == "yue":
            eng_type = "16k_zh-dialect"

        payload = {
            "ProjectId": 0,
            "SubServiceType": 2,
            "EngSerViceType": eng_type,
            "SourceType": 1,
            "VoiceFormat": "wav",
            "UsrAudioKey": uuid.uuid4().hex,
            "Data": pcm_b64,
            "DataLen": len(wav_bytes),
        }
        payload_json = json.dumps(payload)
        headers = _tencent_sign(
            cfg.tencent_secret_id,
            cfg.tencent_secret_key,
            payload_json,
        )
        resp = requests.post(
            "https://asr.tencentcloudapi.com",
            headers=headers,
            data=payload_json.encode(),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        resp_data = data.get("Response", {})
        if "Error" in resp_data:
            raise RuntimeError(f"腾讯云 ASR 错误: {resp_data['Error']}")

        text = resp_data.get("Result", "").strip()
        word_list = resp_data.get("WordList", [])

        if word_list:
            char_ts = _word_list_to_char_ts(
                word_list,
                start_key="OffsetStartMs",
                end_key="OffsetEndMs",
                word_key="Word",
                time_unit="ms",
                time_offset=vad_start,
            )
        else:
            char_ts = _linear_char_timestamps(text, vad_start, vad_end)

        return text, char_ts


# ---------------------------------------------------------------------------
# Engine 2: 百度语音识别（Pro API，简单 OAuth）
# ---------------------------------------------------------------------------

class BaiduASREngine(_SegmentCloudEngine):
    """百度语音识别 Pro API（无词级时间戳，自动线性插值）"""

    PROVIDER_NAME = "百度 ASR"
    _token: str | None = None
    _token_expires: float = 0.0

    def is_configured(self) -> bool:
        cfg = get_settings().cloud_asr
        return bool(cfg.baidu_api_key and cfg.baidu_secret_key)

    def _get_token(self) -> str:
        if self._token and time.time() < self._token_expires:
            return self._token

        cfg = get_settings().cloud_asr
        resp = requests.post(
            "https://aip.baidubce.com/oauth/2.0/token",
            params={
                "grant_type": "client_credentials",
                "client_id": cfg.baidu_api_key,
                "client_secret": cfg.baidu_secret_key,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = data["access_token"]
        self._token_expires = time.time() + data.get("expires_in", 2592000) - 300
        return self._token

    def _transcribe_chunk(
        self,
        pcm_b64: str,
        wav_bytes: bytes,
        duration_ms: int,
        vad_start: float,
        vad_end: float,
        language: str | None,
    ) -> tuple[str, list[CharTimestamp]]:
        token = self._get_token()
        payload = {
            "format": "wav",
            "rate": 16000,
            "channel": 1,
            "cuid": "meeting-ai",
            "token": token,
            "speech": pcm_b64,
            "len": len(wav_bytes),
            "dev_pid": 80001,  # 普通话 + 远场
        }
        if language == "en":
            payload["dev_pid"] = 1737  # 英语

        resp = requests.post(
            "https://vop.baidu.com/pro_api",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("err_no", 0) != 0:
            raise RuntimeError(f"百度 ASR 错误 {data.get('err_no')}: {data.get('err_msg')}")

        results = data.get("result", [])
        text = "".join(results).strip()
        char_ts = _linear_char_timestamps(text, vad_start, vad_end)
        return text, char_ts


# ---------------------------------------------------------------------------
# Engine 3: 讯飞语音识别（录音文件转写 lfasr v2，整文件异步模式）
# ---------------------------------------------------------------------------

def _iflytek_signa(app_id: str, ts: str, api_key: str) -> str:
    """讯飞 lfasr v2 签名: Base64(HMAC-SHA1(MD5(appId+ts), apiKey))"""
    md5_str = hashlib.md5((app_id + ts).encode("utf-8")).hexdigest()
    sig = hmac.new(
        api_key.encode("utf-8"),
        md5_str.encode("utf-8"),
        hashlib.sha1,
    ).digest()
    return base64.b64encode(sig).decode("utf-8")


class IflyTekASREngine(CloudASREngine):
    """讯飞录音文件转写 lfasr v2（整文件提交 → 轮询 → 词级时间戳）"""

    PROVIDER_NAME = "讯飞 ASR"
    _BASE = "https://raasr.xfyun.cn/v2/api"

    def is_configured(self) -> bool:
        cfg = get_settings().cloud_asr
        return bool(cfg.iflytek_app_id and cfg.iflytek_api_key and cfg.iflytek_api_secret)

    def _headers(self) -> dict[str, str]:
        cfg = get_settings().cloud_asr
        ts = str(int(time.time()))
        return {
            "appId": cfg.iflytek_app_id,
            "signa": _iflytek_signa(cfg.iflytek_app_id, ts, cfg.iflytek_api_secret),
            "ts": ts,
        }

    def _do_transcribe(
        self, audio_path: Path, duration: float, language: str | None,
    ) -> TranscriptResult:
        """整文件上传 → 轮询 → 解析词级时间戳"""
        cfg = get_settings().cloud_asr

        # 1. 上传文件
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        ts = str(int(time.time()))
        headers = {
            "appId": cfg.iflytek_app_id,
            "signa": _iflytek_signa(cfg.iflytek_app_id, ts, cfg.iflytek_api_secret),
            "ts": ts,
            "fileSize": str(len(audio_bytes)),
            "fileName": audio_path.name,
            "duration": str(int(duration * 1000)),
        }
        resp = requests.post(
            f"{self._BASE}/upload",
            headers=headers,
            data=audio_bytes,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "000000":
            raise RuntimeError(f"讯飞上传失败: {data}")
        task_id = data["data"]["taskId"]
        logger.info(f"[讯飞 ASR] 上传成功, taskId={task_id}")

        # 2. 轮询结果（最多 10 分钟）
        result_data: dict | None = None
        for _ in range(120):
            time.sleep(5)
            ts2 = str(int(time.time()))
            qh = {
                "appId": cfg.iflytek_app_id,
                "signa": _iflytek_signa(cfg.iflytek_app_id, ts2, cfg.iflytek_api_secret),
                "ts": ts2,
            }
            qr = requests.post(
                f"{self._BASE}/getResult",
                headers=qh,
                json={"taskId": task_id},
                timeout=30,
            )
            qr.raise_for_status()
            qd = qr.json()
            if qd.get("code") != "000000":
                raise RuntimeError(f"讯飞查询失败: {qd}")
            status = qd["data"].get("taskStatus", "0")
            if status == "4":  # 完成
                result_data = qd["data"]
                break
            elif status in ("3", "9"):  # 失败
                raise RuntimeError(f"讯飞转写失败: {qd}")
            logger.debug(f"[讯飞 ASR] 等待转写, status={status}")
        else:
            raise RuntimeError("讯飞 ASR 转写超时（10分钟）")

        # 3. 解析结果
        return self._parse_result(result_data, duration)

    def _parse_result(self, data: dict, duration: float) -> TranscriptResult:
        """解析讯飞 orderResult JSON"""
        order_result_str = data.get("orderResult", "")
        if not order_result_str:
            return TranscriptResult(segments=[], char_timestamps=[], duration=duration)

        try:
            order = json.loads(order_result_str)
        except json.JSONDecodeError:
            logger.warning("[讯飞 ASR] orderResult 解析失败")
            return TranscriptResult(segments=[], char_timestamps=[], duration=duration)

        # orderResult 格式: {"lattice": [{"json_1best": "{...}"}]}
        lattice = order.get("lattice", [])
        segments: list[Segment] = []
        all_char_ts: list[list[CharTimestamp]] = []
        seg_id = 0

        for item in lattice:
            try:
                best = json.loads(item.get("json_1best", "{}"))
                words_info = best.get("st", {}).get("rt", [])
                if not words_info:
                    continue

                text_parts: list[str] = []
                char_ts: list[CharTimestamp] = []
                seg_start: float | None = None
                seg_end: float = 0.0

                for rt in words_info:
                    for ws in rt.get("ws", []):
                        for cw in ws.get("cw", []):
                            w = cw.get("w", "")
                            if not w:
                                continue
                            t_start = float(ws.get("bg", 0)) / 1000.0
                            t_end = float(ws.get("ed", 0)) / 1000.0
                            if seg_start is None:
                                seg_start = t_start
                            seg_end = max(seg_end, t_end)
                            text_parts.append(w)
                            for ch in list(w):
                                char_ts.append(CharTimestamp(
                                    char=ch,
                                    start=round(t_start, 3),
                                    end=round(t_end, 3),
                                ))

                text = "".join(text_parts).strip()
                if not text or seg_start is None:
                    continue

                segments.append(Segment(
                    id=seg_id,
                    start=round(seg_start, 3),
                    end=round(seg_end, 3),
                    text=text,
                    speaker=None,
                ))
                all_char_ts.append(char_ts)
                seg_id += 1

            except Exception as e:
                logger.debug(f"[讯飞 ASR] 解析 lattice 项失败: {e}")

        logger.info(f"[讯飞 ASR] 解析完成: {len(segments)} 段")
        return TranscriptResult(segments=segments, char_timestamps=all_char_ts, duration=duration)


# ---------------------------------------------------------------------------
# Engine 4: 字节跳动/火山引擎语音识别（同步 REST，utterance 级时间戳）
# ---------------------------------------------------------------------------

class BytedanceASREngine(_SegmentCloudEngine):
    """字节跳动/火山引擎语音识别（同步 REST，支持词级时间戳）"""

    PROVIDER_NAME = "字节跳动 ASR"

    def is_configured(self) -> bool:
        cfg = get_settings().cloud_asr
        return bool(cfg.bytedance_app_id and cfg.bytedance_access_token)

    def _transcribe_chunk(
        self,
        pcm_b64: str,
        wav_bytes: bytes,
        duration_ms: int,
        vad_start: float,
        vad_end: float,
        language: str | None,
    ) -> tuple[str, list[CharTimestamp]]:
        cfg = get_settings().cloud_asr
        lang_code = "zh-CN"
        if language == "en":
            lang_code = "en-US"
        elif language == "yue":
            lang_code = "zh-HK"

        payload = {
            "app": {
                "appid": cfg.bytedance_app_id,
                "token": cfg.bytedance_access_token,
                "cluster": "volcano_algo",
            },
            "user": {"uid": "meeting-ai"},
            "request": {
                "reqid": uuid.uuid4().hex,
                "workflow": "audio_in,resample,partition,vad,fe,decode,itn,nlu_punctuation",
                "sequence": -1,
            },
            "audio": {
                "format": "wav",
                "rate": 16000,
                "encoding": "raw",
                "bits": 16,
                "channel": 1,
                "language": lang_code,
                "data": pcm_b64,
            },
        }
        headers = {
            "Authorization": f"Bearer; {cfg.bytedance_access_token}",
            "X-App-ID": cfg.bytedance_app_id,
            "Content-Type": "application/json",
        }
        resp = requests.post(
            "https://openspeech.bytedance.com/api/v1/asr",
            headers=headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") != 1000:
            raise RuntimeError(f"字节跳动 ASR 错误 {data.get('code')}: {data.get('message')}")

        utterances = data.get("data", {}).get("result", {}).get("utterances", [])
        if not utterances:
            return "", []

        text = " ".join(u.get("text", "") for u in utterances).strip()
        char_ts: list[CharTimestamp] = []

        for utt in utterances:
            words = utt.get("words", [])
            if words:
                char_ts.extend(_word_list_to_char_ts(
                    words,
                    start_key="start_time",
                    end_key="end_time",
                    word_key="text",
                    time_unit="ms",
                    time_offset=vad_start,
                ))
            else:
                # utterance 级回退
                t0 = utt.get("start_time", 0) / 1000.0 + vad_start
                t1 = utt.get("end_time", duration_ms) / 1000.0 + vad_start
                utt_text = utt.get("text", "")
                char_ts.extend(_linear_char_timestamps(utt_text, t0, t1))

        if not char_ts:
            char_ts = _linear_char_timestamps(text, vad_start, vad_end)

        return text, char_ts


# ---------------------------------------------------------------------------
# Engine 5: 阿里云 DashScope 语音识别（paraformer-v2，异步任务）
# ---------------------------------------------------------------------------

class AliASREngine(CloudASREngine):
    """阿里云 DashScope 语音识别（paraformer-v2，文件上传 → 异步任务 → 句级时间戳）"""

    PROVIDER_NAME = "阿里云 ASR"
    _BASE = "https://dashscope.aliyuncs.com/api/v1"

    def is_configured(self) -> bool:
        return bool(get_settings().cloud_asr.ali_api_key)

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {get_settings().cloud_asr.ali_api_key}",
            "Content-Type": "application/json",
        }

    def _do_transcribe(
        self, audio_path: Path, duration: float, language: str | None,
    ) -> TranscriptResult:
        # 1. 上传文件，获取 file_id
        file_id = self._upload_file(audio_path)
        logger.info(f"[阿里云 ASR] 文件上传完成: file_id={file_id}")

        # 2. 提交转写任务
        lang_hints = ["zh", "en"]
        if language == "en":
            lang_hints = ["en"]
        elif language == "yue":
            lang_hints = ["yue", "zh"]

        task_payload = {
            "model": "paraformer-v2",
            "input": {"file_urls": [f"fileid://{file_id}"]},
            "parameters": {
                "language_hints": lang_hints,
                "channel_id": [0],
            },
        }
        headers = self._auth_headers()
        headers["X-DashScope-Async"] = "enable"
        resp = requests.post(
            f"{self._BASE}/services/audio/asr/transcription",
            headers=headers,
            json=task_payload,
            timeout=30,
        )
        resp.raise_for_status()
        task_data = resp.json()
        task_id = task_data.get("output", {}).get("task_id")
        if not task_id:
            raise RuntimeError(f"阿里云 ASR 提交失败: {task_data}")
        logger.info(f"[阿里云 ASR] 任务提交成功: task_id={task_id}")

        # 3. 轮询结果
        result_url = self._poll_task(task_id)

        # 4. 下载并解析结果
        return self._fetch_result(result_url, duration)

    def _upload_file(self, audio_path: Path) -> str:
        """上传音频文件到 DashScope, 返回 file_id"""
        headers = {
            "Authorization": f"Bearer {get_settings().cloud_asr.ali_api_key}",
        }
        with open(audio_path, "rb") as f:
            resp = requests.post(
                f"{self._BASE}/files",
                headers=headers,
                files={"file": (audio_path.name, f, "audio/wav")},
                data={"purpose": "asr"},
                timeout=120,
            )
        resp.raise_for_status()
        data = resp.json()
        file_id = data.get("id") or data.get("file_id")
        if not file_id:
            raise RuntimeError(f"阿里云 ASR 文件上传失败: {data}")
        return file_id

    def _poll_task(self, task_id: str, max_wait: int = 600) -> str:
        """轮询任务完成，返回结果 URL"""
        headers = self._auth_headers()
        for _ in range(max_wait // 5):
            time.sleep(5)
            resp = requests.get(
                f"{self._BASE}/tasks/{task_id}",
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            status = data.get("output", {}).get("task_status", "")
            if status == "SUCCEEDED":
                results = data["output"].get("results", [])
                if results:
                    return results[0].get("transcription_url", "")
                raise RuntimeError("阿里云 ASR: 无结果 URL")
            elif status == "FAILED":
                raise RuntimeError(f"阿里云 ASR 任务失败: {data}")
            logger.debug(f"[阿里云 ASR] 等待任务, status={status}")
        raise RuntimeError("阿里云 ASR 转写超时")

    def _fetch_result(self, result_url: str, duration: float) -> TranscriptResult:
        """下载并解析转写结果 JSON"""
        resp = requests.get(result_url, timeout=30)
        resp.raise_for_status()
        result = resp.json()

        transcripts = result.get("transcripts", [])
        segments: list[Segment] = []
        all_char_ts: list[list[CharTimestamp]] = []
        seg_id = 0

        for transcript in transcripts:
            for sentence in transcript.get("sentences", []):
                text = sentence.get("text", "").strip()
                if not text:
                    continue
                t_start = sentence.get("begin_time", 0) / 1000.0
                t_end = sentence.get("end_time", 0) / 1000.0

                # 词级时间戳（如有）
                words = sentence.get("words", [])
                if words:
                    char_ts = _word_list_to_char_ts(
                        words,
                        start_key="begin_time",
                        end_key="end_time",
                        word_key="text",
                        time_unit="ms",
                    )
                else:
                    char_ts = _linear_char_timestamps(text, t_start, t_end)

                segments.append(Segment(
                    id=seg_id,
                    start=round(t_start, 3),
                    end=round(t_end, 3),
                    text=text,
                    speaker=None,
                ))
                all_char_ts.append(char_ts)
                seg_id += 1

        logger.info(f"[阿里云 ASR] 解析完成: {len(segments)} 段")
        return TranscriptResult(segments=segments, char_timestamps=all_char_ts, duration=duration)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_CLOUD_ENGINES: dict[str, type[CloudASREngine]] = {
    "cloud-tencent": TencentASREngine,
    "cloud-baidu": BaiduASREngine,
    "cloud-iflytek": IflyTekASREngine,
    "cloud-bytedance": BytedanceASREngine,
    "cloud-ali": AliASREngine,
}

# 云端引擎元数据（用于前端展示）
CLOUD_ENGINE_META: dict[str, dict[str, str]] = {
    "cloud-tencent": {
        "label": "腾讯云 ASR",
        "desc": "微信同款技术，词级时间戳",
        "cer": "商业级",
        "vram": "☁ 云端",
    },
    "cloud-baidu": {
        "label": "百度 ASR",
        "desc": "Pro API，支持远场/噪音场景",
        "cer": "商业级",
        "vram": "☁ 云端",
    },
    "cloud-iflytek": {
        "label": "讯飞 ASR",
        "desc": "科大讯飞，录音文件转写，词级时间戳",
        "cer": "商业级",
        "vram": "☁ 云端",
    },
    "cloud-bytedance": {
        "label": "字节跳动 ASR",
        "desc": "火山引擎，词级时间戳，同步 REST",
        "cer": "商业级",
        "vram": "☁ 云端",
    },
    "cloud-ali": {
        "label": "阿里云 ASR",
        "desc": "DashScope paraformer-v2，开源同款",
        "cer": "商业级",
        "vram": "☁ 云端",
    },
}

# 每个引擎需要填写的字段定义（供前端渲染表单）
CLOUD_ENGINE_FIELDS: dict[str, list[dict[str, str]]] = {
    "cloud-tencent": [
        {"key": "tencent_secret_id",  "label": "SecretId",  "placeholder": "AKIDxxxxxxxx"},
        {"key": "tencent_secret_key", "label": "SecretKey", "placeholder": "密钥值"},
    ],
    "cloud-baidu": [
        {"key": "baidu_api_key",    "label": "API Key",    "placeholder": "从百度智能云控制台获取"},
        {"key": "baidu_secret_key", "label": "Secret Key", "placeholder": "密钥值"},
    ],
    "cloud-iflytek": [
        {"key": "iflytek_app_id",     "label": "App ID",     "placeholder": "从讯飞开放平台获取"},
        {"key": "iflytek_api_key",    "label": "API Key",    "placeholder": "密钥值"},
        {"key": "iflytek_api_secret", "label": "API Secret", "placeholder": "密钥值"},
    ],
    "cloud-bytedance": [
        {"key": "bytedance_app_id",        "label": "App ID",       "placeholder": "从火山引擎控制台获取"},
        {"key": "bytedance_access_token",  "label": "Access Token", "placeholder": "密钥值"},
    ],
    "cloud-ali": [
        {"key": "ali_api_key", "label": "API Key", "placeholder": "sk-xxxxxxxx"},
    ],
}

# 单例缓存
_cloud_engine_instances: dict[str, CloudASREngine] = {}


def get_cloud_engine(engine_id: str) -> CloudASREngine:
    """获取云端 ASR 引擎单例"""
    if engine_id not in _CLOUD_ENGINES:
        raise ValueError(f"未知云端 ASR 引擎: {engine_id}")

    if engine_id not in _cloud_engine_instances:
        _cloud_engine_instances[engine_id] = _CLOUD_ENGINES[engine_id]()

    return _cloud_engine_instances[engine_id]


def is_cloud_engine_id(engine_id: str) -> bool:
    return engine_id in _CLOUD_ENGINES
