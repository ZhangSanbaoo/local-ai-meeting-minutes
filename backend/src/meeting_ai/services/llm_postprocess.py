"""
LLM 后处理服务

使用 LLM 进行：
1. 性别推断（基于对话内容和社会常识）
2. 错别字修复（基于上下文）
"""

import json
import re
from pathlib import Path

from llama_cpp import Llama

from ..config import get_settings
from ..logger import get_logger
from ..models import Gender, Segment, SpeakerInfo

logger = get_logger("services.llm_postprocess")


class LLMPostprocessor:
    """
    LLM 后处理器
    
    使用方式：
        processor = LLMPostprocessor(llm)
        
        # 推断性别
        genders = processor.infer_genders(segments, speakers)
        
        # 修复错别字
        fixed_segments = processor.fix_typos(segments)
    """

    def __init__(self, llm: Llama) -> None:
        """
        Args:
            llm: 已加载的 Llama 模型实例
        """
        self._llm = llm
        self._settings = get_settings().llm

    def infer_genders(
        self,
        segments: list[Segment],
        speakers: dict[str, SpeakerInfo],
    ) -> dict[str, Gender]:
        """
        使用 LLM 推断说话人性别
        
        基于：
        1. 对话内容中的称呼和代词
        2. 名字/职位的社会常识（如"张教授"在中国学术界男性比例更高）
        3. 说话风格
        
        Args:
            segments: 对话片段列表
            speakers: 说话人信息字典
            
        Returns:
            {speaker_id: Gender, ...}
        """
        if not speakers:
            return {}

        # 构建对话摘要
        dialog_lines = []
        for seg in segments[:30]:  # 限制长度
            if seg.text and seg.speaker:
                name = speakers.get(seg.speaker, SpeakerInfo(
                    id=seg.speaker, display_name=seg.speaker,
                    total_duration=0, segment_count=0
                )).display_name
                dialog_lines.append(f"{name}: {seg.text[:80]}")

        dialog_text = "\n".join(dialog_lines)

        # 构建说话人信息
        speaker_info = []
        for spk_id, info in speakers.items():
            speaker_info.append(f"- {spk_id}: 显示名称=\"{info.display_name}\"")

        prompt = f"""你是一个语音分析助手。请根据对话内容和中国社会常识，推断每个说话人的性别。

对话内容：
{dialog_text}

说话人信息：
{chr(10).join(speaker_info)}

请分析：
1. 对话中是否有"他/她"、"先生/女士"等性别线索
2. 根据名字或职位的社会常识推断（如"教授"在中国学术界男性比例约70%）
3. 说话风格和用词习惯

请用 JSON 格式回复，每个说话人给出性别和理由：
{{
  "SPEAKER_00": {{"gender": "female", "reason": "被称为主持人，说话风格偏女性"}},
  "SPEAKER_01": {{"gender": "male", "reason": "被称为张教授，教授中男性比例较高"}}
}}

gender 只能是 "male"、"female" 或 "unknown"。
只输出 JSON，不要其他内容。"""

        logger.debug(f"Gender inference prompt:\n{prompt}")

        try:
            response = self._llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=512,
            )

            content = response["choices"][0]["message"]["content"]
            logger.debug(f"Gender inference response:\n{content}")

            # 解析 JSON
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                result = json.loads(json_match.group())
                
                genders = {}
                for spk_id, data in result.items():
                    gender_str = data.get("gender", "unknown").lower()
                    if gender_str == "male":
                        genders[spk_id] = Gender.MALE
                    elif gender_str == "female":
                        genders[spk_id] = Gender.FEMALE
                    else:
                        genders[spk_id] = Gender.UNKNOWN
                    
                    reason = data.get("reason", "")
                    logger.info(f"LLM 性别推断 {spk_id}: {gender_str} ({reason})")
                
                return genders

        except Exception as e:
            logger.warning(f"LLM 性别推断失败: {e}")

        return {}

    def fix_typos(
        self,
        segments: list[Segment],
        batch_size: int = 5,
    ) -> list[Segment]:
        """
        使用 LLM 修复错别字
        
        Args:
            segments: 对话片段列表
            batch_size: 每批处理的片段数
            
        Returns:
            修复后的片段列表
        """
        if not segments:
            return []

        fixed_segments = []
        
        # 分批处理
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            fixed_batch = self._fix_typos_batch(batch)
            fixed_segments.extend(fixed_batch)

        return fixed_segments

    def _fix_typos_batch(self, segments: list[Segment]) -> list[Segment]:
        """修复一批片段的错别字"""
        
        # 构建文本列表
        texts = []
        for j, seg in enumerate(segments):
            texts.append(f"{j+1}. {seg.text}")

        prompt = f"""你是一个中文校对助手。以下是语音转文字的结果，可能包含一些错别字或同音字错误。
请修复明显的错误，保持原意不变。

原文：
{chr(10).join(texts)}

要求：
1. 只修复明显的错别字和同音字错误
2. 不要改变原文的意思和风格
3. 如果不确定，保持原文不变
4. 常见错误如：外相→外向，和群→合群，生学→升学

请用 JSON 格式回复，只包含需要修改的行：
{{
  "1": "修复后的文本",
  "3": "修复后的文本"
}}

如果某行不需要修改，就不要包含在 JSON 中。
只输出 JSON，不要其他内容。如果全部不需要修改，输出 {{}}"""

        logger.debug(f"Typo fix prompt:\n{prompt}")

        try:
            response = self._llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
            )

            content = response["choices"][0]["message"]["content"]
            logger.debug(f"Typo fix response:\n{content}")

            # 解析 JSON
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                fixes = json.loads(json_match.group())
                
                # 应用修复
                fixed = []
                for j, seg in enumerate(segments):
                    key = str(j + 1)
                    if key in fixes and fixes[key]:
                        new_text = fixes[key]
                        if new_text != seg.text:
                            logger.info(f"错别字修复: \"{seg.text[:30]}...\" -> \"{new_text[:30]}...\"")
                            seg = seg.model_copy()
                            seg.text = new_text
                    fixed.append(seg)
                
                return fixed

        except Exception as e:
            logger.warning(f"LLM 错别字修复失败: {e}")

        # 失败时返回原文
        return segments


def create_postprocessor(llm: Llama) -> LLMPostprocessor:
    """创建后处理器实例"""
    return LLMPostprocessor(llm)
