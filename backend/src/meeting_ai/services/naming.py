"""
智能命名服务

为说话人分配有意义的名字，优先级：
1. 从对话中提取的真实姓名（如"张教授"、"小柔"）
2. LLM 推断的角色（如"主持人"）
3. 基于性别的兜底命名（如"男性01"）
"""

import json
import re
from collections import Counter
from pathlib import Path

from ..config import get_settings
from ..logger import get_logger
from ..models import Gender, NameKind, Segment, SpeakerInfo

logger = get_logger("services.naming")


# 常见的称呼后缀
HONORIFICS = [
    "教授", "老师", "主任", "经理", "总", "博士", "医生", "律师",
    "院长", "书记", "同学", "工", "工程师", "秘书", "科长", "处长",
    "局长", "部长", "董事", "总裁", "CEO", "CTO", "先生", "女士",
]

# 自我介绍模式
SELF_INTRO_PATTERNS = [
    r"我是(?P<n>[\u4e00-\u9fa5]{1,4})(?P<title>教授|老师|博士|主任|经理|总|医生)?",
    r"我叫(?P<n>[\u4e00-\u9fa5]{1,4})(?P<title>教授|老师|博士|主任|经理|总)?",
    r"我姓(?P<n>[\u4e00-\u9fa5])(?P<title>教授|老师|博士|主任|经理|总)?",
]

# 称呼他人模式（带后缀）
# 添加边界检查，避免匹配到"跟张教授"、"将跟张教授"这样的片段
MENTION_PATTERNS = [
    # 前面是句首、标点、空白、或特定动词（请、问），避免误匹配"跟XX教授"中的"跟XX"
    r"(?:^|[，,。！？：:\s]|请问?)(?P<n>[\u4e00-\u9fa5]{1,3})(?P<title>教授|老师|博士|主任|经理|总|医生|律师)",
]

# 直接称呼模式（名字后跟标点或特定词）
# 如 "小柔，我想问你" 或 "小明你好"
DIRECT_ADDRESS_PATTERNS = [
    r"(?P<n>小[\u4e00-\u9fa5]{1,2})[，,。！？：:\s]",   # 小X + 标点/空格
    r"(?P<n>小[\u4e00-\u9fa5]{1,2})(?:你|您|啊|呀|吧|我|他|她)",  # 小X + 代词/语气词
    r"(?P<n>阿[\u4e00-\u9fa5]{1,2})[，,。！？：:\s]",   # 阿X + 标点
    r"(?P<n>阿[\u4e00-\u9fa5]{1,2})(?:你|您|啊|呀|吧)",  # 阿X + 语气词
    r"(?P<n>老[\u4e00-\u9fa5])[，,。！？：:\s]",        # 老X + 标点（如"老王，你来"）
]


# 常见的"小X"但不是人名的词
NOT_NAMES = {
    # 事物
    "小程序", "小项目", "小组", "小队", "小团", "小班", "小区", "小店", "小号",
    "小说", "小品", "小曲", "小调", "小节", "小段", "小章", "小册", "小报",
    "小时", "小会", "小票", "小费", "小计", "小结", "小题", "小考", "小测",
    "小车", "小船", "小路", "小道", "小巷", "小街", "小院", "小楼", "小屋",
    "小吃", "小菜", "小碗", "小碟", "小杯", "小瓶", "小包", "小盒", "小袋",
    "小数", "小型", "小微", "小众", "小康", "小资", "小农", "小贩", "小商",
    # 形容词用法
    "小心", "小气", "小看", "小觑",
    # 阿X
    "阿姨", "阿伯", "阿婆", "阿公",
    # 老X 非人名
    "老师", "老板", "老总", "老大", "老二", "老三", "老婆", "老公", 
    "老人", "老年", "老化", "老旧", "老家", "老乡",
}


def extract_names_from_text(text: str) -> list[str]:
    """
    从文本中提取可能的人名/称呼

    Args:
        text: 对话文本

    Returns:
        提取到的名字列表，如 ["张教授", "李老师", "小柔"]
    """
    names = []

    # 自我介绍（"我是XX"、"我叫XX"）
    for pattern in SELF_INTRO_PATTERNS:
        for match in re.finditer(pattern, text):
            name = match.group("n")
            title = match.group("title") or ""
            if name:
                full_name = f"{name}{title}".strip()
                if full_name not in NOT_NAMES:
                    names.append(full_name)
                    logger.debug(f"自我介绍提取: {full_name}")

    # 称呼他人（带后缀，如"张教授"）
    for pattern in MENTION_PATTERNS:
        for match in re.finditer(pattern, text):
            name = match.group("n")
            title = match.group("title") or ""
            if name and title:
                full_name = f"{name}{title}"
                if full_name not in NOT_NAMES:
                    names.append(full_name)
                    logger.debug(f"称呼提取: {full_name}")

    # 直接称呼（小X、阿X、老X 等）
    for pattern in DIRECT_ADDRESS_PATTERNS:
        for match in re.finditer(pattern, text):
            name = match.group("n")
            if name and len(name) >= 2:
                # 过滤非人名
                if name not in NOT_NAMES:
                    names.append(name)
                    logger.debug(f"直接称呼提取: {name}")
                else:
                    logger.debug(f"过滤非人名: {name}")

    return names


def extract_self_intro_name(text: str) -> str | None:
    """
    从文本中提取自我介绍的名字

    "我叫小王" -> "小王"
    "我是张教授" -> "张教授"
    """
    for pattern in SELF_INTRO_PATTERNS:
        match = re.search(pattern, text)
        if match:
            name = match.group("n")
            title = match.group("title") or ""
            if name:
                full_name = f"{name}{title}".strip()
                if full_name not in NOT_NAMES:
                    return full_name
    return None

def guess_host_speaker(segments: list[Segment]) -> str | None:
    """
    猜测谁是主持人/提问者
    
    特征：
    - 问句比例高
    - 经常使用"请问"、"您如何看待"等
    
    Args:
        segments: 片段列表
        
    Returns:
        可能的主持人 speaker_id，找不到返回 None
    """
    question_words = ["请问", "您如何", "怎么看", "能否", "是不是", "对吗", "您觉得", "您认为"]
    
    speaker_stats = {}
    
    for seg in segments:
        speaker = seg.speaker
        if not speaker or speaker == "UNKNOWN":
            continue
            
        if speaker not in speaker_stats:
            speaker_stats[speaker] = {"questions": 0, "total": 0}
        
        speaker_stats[speaker]["total"] += 1
        
        text = seg.text or ""
        # 检查是否是问句
        is_question = (
            text.strip().endswith(("?", "？")) or
            any(w in text for w in question_words)
        )
        if is_question:
            speaker_stats[speaker]["questions"] += 1
    
    # 找问句比例最高的（至少 2 句话）
    best_speaker = None
    best_ratio = 0.0
    
    for speaker, stats in speaker_stats.items():
        if stats["total"] < 2:
            continue
        ratio = stats["questions"] / stats["total"]
        if ratio > best_ratio:
            best_ratio = ratio
            best_speaker = speaker
    
    # 只有问句比例超过 30% 才认为是主持人
    if best_ratio >= 0.3:
        return best_speaker
    
    return None


class NamingService:
    """
    智能命名服务
    
    使用方式：
        service = NamingService()
        speakers = service.name_speakers(segments, gender_map, wav_path)
    """
    
    def __init__(self) -> None:
        self._settings = get_settings().llm

    @property
    def llm(self):
        """获取共享的 LLM 实例"""
        from .llm import get_llm
        return get_llm()
    
    def _validate_names_with_llm(self, candidate_names: list[str], context: str) -> list[str]:
        """
        使用 LLM 验证候选名字是否真的是人名
        
        Args:
            candidate_names: 候选名字列表
            context: 上下文（对话片段）
            
        Returns:
            验证通过的人名列表
        """
        if not candidate_names or not self._settings.enabled or self.llm is None:
            return candidate_names
        
        # 构建 prompt
        names_str = "、".join(candidate_names)
        prompt = f"""请判断以下词语是否包含中文人名（昵称、小名、姓名、职称都算）。

候选词语：{names_str}

上下文：
{context[:500]}

对于包含人名的词语，请提取出纯粹的人名部分（可以包含职称）。
对于不包含人名的词语，忽略它。

示例：
- "小柔" → 小柔
- "请问张教授" → 张教授（提取出人名部分）
- "将跟张教授" → 张教授（提取出人名部分）
- "小程序" → （忽略，不是人名）
- "小项目" → （忽略，不是人名）
- "老王" → 老王
- "老师" → （忽略，仅职业）

只输出提取的人名，用逗号分隔。如果都没有人名，输出"无"："""

        try:
            response = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100,
            )
            
            content = response["choices"][0]["message"]["content"].strip()
            logger.debug(f"LLM 名字验证结果: {content}")
            
            if content == "无" or not content:
                return []

            # 解析返回的名字
            # LLM 可能返回提取后的人名（如"张教授"），也可能返回原候选词
            validated = []
            for name in content.replace("，", ",").split(","):
                name = name.strip()
                if not name:
                    continue

                # 1. 直接匹配原候选词
                if name in candidate_names:
                    validated.append(name)
                # 2. LLM 提取出的名字（可能是从"请问张教授"中提取的"张教授"）
                # 检查这个名字是否是某个候选词的子串
                elif any(name in cand or cand in name for cand in candidate_names):
                    validated.append(name)

            # 去重
            validated = list(dict.fromkeys(validated))
            logger.info(f"名字验证: {candidate_names} -> {validated}")
            return validated
            
        except Exception as e:
            logger.warning(f"LLM 名字验证失败: {e}")
            return candidate_names  # 失败时返回原列表
    
    def name_speakers(
        self,
        segments: list[Segment],
        gender_map: dict[str, tuple[Gender, float]],
    ) -> dict[str, SpeakerInfo]:
        """
        为所有说话人命名
        
        Args:
            segments: 对话片段列表
            gender_map: 性别检测结果 {speaker_id: (gender, f0_median)}
            
        Returns:
            {speaker_id: SpeakerInfo, ...}
        """
        # 收集所有说话人
        speakers = set(
            seg.speaker for seg in segments
            if seg.speaker and seg.speaker != "UNKNOWN"
        )
        
        if not speakers:
            return {}
        
        # 1. 从对话中提取可能的名字
        all_text = " ".join(seg.text for seg in segments if seg.text)
        mentioned_names = extract_names_from_text(all_text)
        
        # 1.5 用 LLM 验证候选名字是否真的是人名
        if mentioned_names and self._settings.enabled:
            unique_names = list(set(mentioned_names))
            context = " ".join(seg.text for seg in segments[:10] if seg.text)
            validated_names = self._validate_names_with_llm(unique_names, context)
            # 只保留验证通过的名字
            mentioned_names = [n for n in mentioned_names if n in validated_names]
        
        name_counts = Counter(mentioned_names)
        
        logger.info(f"从对话中提取到的名字: {dict(name_counts)}")
        
        # 2. 猜测主持人
        host_speaker = guess_host_speaker(segments)
        if host_speaker:
            logger.info(f"推测主持人: {host_speaker}")
        
        # 3. 尝试用 LLM 命名（如果启用）
        llm_names = {}
        if self._settings.enabled and self.llm is not None:
            try:
                llm_names = self._llm_name_speakers(segments, list(speakers))
            except Exception as e:
                logger.warning(f"LLM 命名失败: {e}")
        
        # 4. 构建最终命名
        results = {}
        gender_counters = {"male": 0, "female": 0, "unknown": 0}
        role_counters = {}  # 用于角色编号，如 "嘉宾01", "嘉宾02"
        
        # 收集对话中所有文本，用于验证名字是否真实存在
        all_text = " ".join(seg.text for seg in segments if seg.text)
        
        def is_name_in_text(name: str, text: str) -> bool:
            """检查名字是否真的在对话中出现"""
            if not name:
                return False
            # 去掉常见后缀再检查
            base_name = name
            for suffix in ["教授", "老师", "博士", "主任", "经理", "总", "医生", "律师", "同学"]:
                if name.endswith(suffix):
                    base_name = name[:-len(suffix)]
                    break
            
            # 名字必须在对话中出现（完整名字或去掉后缀的名字）
            return name in text or (len(base_name) >= 2 and base_name in text)
        
        def get_role_with_number(role: str) -> str:
            """给角色名添加编号"""
            if role not in role_counters:
                role_counters[role] = 0
            role_counters[role] += 1
            return f"{role}{role_counters[role]:02d}"
        
        used_names = set()  # 防止两个说话人获得相同名字

        for speaker_id in sorted(speakers):
            gender, f0_median = gender_map.get(speaker_id, (Gender.UNKNOWN, 0.0))
            
            # 尝试各种命名方式
            display_name = None
            kind = NameKind.UNKNOWN
            confidence = 0.0
            evidence = []
            
            # ==== 优先级 1：自我介绍（最高优先级）====
            # 如果说话人说了"我叫小王"，直接使用该名字
            speaker_segments = [seg for seg in segments if seg.speaker == speaker_id]
            for seg in speaker_segments:
                if seg.text:
                    self_name = extract_self_intro_name(seg.text)
                    if self_name and is_name_in_text(self_name, all_text) and self_name not in used_names:
                        display_name = self_name
                        kind = NameKind.NAME
                        confidence = 0.95  # 自我介绍置信度最高
                        evidence = [f"自我介绍: '我叫{self_name}'"]
                        logger.info(f"自我介绍命名: {speaker_id} -> {self_name}")
                        break

            # ==== 优先级 2：从对话中提取的被称呼名字 ====
            # 如 "小柔，我想问你" -> 小柔是被称呼者
            if display_name is None and name_counts:
                first_speaker = segments[0].speaker if segments else None
                first_segments_text = " ".join(seg.text for seg in segments[:5] if seg.text)

                for name, count in name_counts.most_common():
                    if is_name_in_text(name, all_text):
                        # 判断：如果第一个说话人说了"小柔，..."，那小柔是另一个人
                        if first_speaker and speaker_id != first_speaker:
                            if name in first_segments_text and name not in used_names:
                                display_name = name
                                kind = NameKind.NAME
                                confidence = 0.85
                                evidence = [f"对话中被称呼为'{name}'"]
                                logger.info(f"正则提取名字: {speaker_id} -> {name}")
                                break
            
            # ==== 优先级 2：LLM 命名的真实名字 ====
            if display_name is None and speaker_id in llm_names:
                llm_result = llm_names[speaker_id]
                llm_name = llm_result.get("name")
                llm_confidence = llm_result.get("confidence", 0)
                llm_kind = llm_result.get("kind", "")
                
                # 只接受 LLM 的"名字"类型（不是角色）
                if llm_kind == "name" and llm_name:
                    if (is_name_in_text(llm_name, all_text)
                        and llm_confidence >= self._settings.confidence_threshold
                        and llm_name not in used_names):
                        display_name = llm_name
                        kind = NameKind.NAME
                        confidence = llm_confidence
                        evidence = llm_result.get("evidence", [])
                        logger.info(f"接受 LLM 名字 '{llm_name}'")
            
            # ==== 优先级 3：主持人（通过问句比例判断）====
            if display_name is None and speaker_id == host_speaker:
                display_name = "主持人"
                kind = NameKind.ROLE
                confidence = 0.6
                evidence = ["问句比例最高"]
            
            # ==== 优先级 4：LLM 推断的角色 ====
            if display_name is None and speaker_id in llm_names:
                llm_result = llm_names[speaker_id]
                llm_role = llm_result.get("role")
                llm_confidence = llm_result.get("confidence", 0)
                llm_kind = llm_result.get("kind", "")

                if llm_kind == "role" and llm_role:
                    if llm_role in ["主持人", "主持"]:
                        display_name = "主持人"
                    else:
                        display_name = get_role_with_number(llm_role)
                    kind = NameKind.ROLE
                    confidence = llm_confidence
                    evidence = llm_result.get("evidence", [])
                    logger.info(f"LLM 角色: {speaker_id} -> {display_name}")
            
            # 优先级 4：性别兜底（带编号）
            if display_name is None:
                gender_counters[gender.value] += 1
                count = gender_counters[gender.value]
                
                if gender == Gender.MALE:
                    display_name = f"男性{count:02d}"
                elif gender == Gender.FEMALE:
                    display_name = f"女性{count:02d}"
                else:
                    display_name = f"说话人{count:02d}"
                
                kind = NameKind.GENDER
                confidence = 0.3
                evidence = [f"基频 {f0_median:.1f}Hz"] if f0_median > 0 else []
            
            # 记录已使用的名字，防止重复分配
            if kind == NameKind.NAME:
                used_names.add(display_name)

            # 构建 SpeakerInfo
            total_duration = sum(seg.duration for seg in segments if seg.speaker == speaker_id)
            segment_count = sum(1 for seg in segments if seg.speaker == speaker_id)
            
            results[speaker_id] = SpeakerInfo(
                id=speaker_id,
                display_name=display_name,
                gender=gender,
                kind=kind,
                confidence=confidence,
                evidence=evidence,
                f0_median=f0_median if f0_median > 0 else None,
                total_duration=total_duration,
                segment_count=segment_count,
            )
            
            logger.info(
                f"命名 {speaker_id} -> {display_name} "
                f"(kind={kind.value}, confidence={confidence:.2f})"
            )
        
        return results
    
    def _llm_name_speakers(
        self,
        segments: list[Segment],
        speakers: list[str],
    ) -> dict[str, dict]:
        """
        使用 LLM 为说话人命名
        
        Args:
            segments: 对话片段
            speakers: 说话人 ID 列表
            
        Returns:
            {speaker_id: {"name": "...", "kind": "name/role", "confidence": 0.8, "evidence": [...]}}
        """
        # 构建对话摘要（限制长度）
        dialog_lines = []
        for seg in segments[:50]:  # 最多 50 条
            if seg.text and seg.speaker:
                text = seg.text[:100]  # 每条最多 100 字
                dialog_lines.append(f"{seg.speaker}: {text}")
        
        dialog_text = "\n".join(dialog_lines)
        
        # 构建 prompt - 分两步：先提取名字，再推断角色
        prompt = f"""你是一个会议纪要助手。请分析以下对话，为每个说话人确定身份。

对话内容：
{dialog_text}

说话人列表：{', '.join(speakers)}

请按以下步骤分析：

**第一步：检查对话中是否【明确提到】了人名**
- 只有对话中【字面出现】的名字才能使用，如"张教授"、"李老师"、"我叫王明"
- 如果对话中没有提到任何人名，所有 name 都必须是 null

**第二步：如果无法确定名字，根据对话中的【实际行为和言语】推断角色**
- 角色必须有对话内容直接支撑，不能凭空猜测
- evidence 必须引用对话中的原文片段作为依据
- 如果对话内容不足以判断具体角色，role 设为 null，不要强行推断

**输出格式（JSON）：**
{{
  "SPEAKER_00": {{"name": "张老师", "kind": "name", "confidence": 0.9, "evidence": ["SPEAKER_01说了'张老师您好'"], "role": null}},
  "SPEAKER_01": {{"name": null, "kind": "role", "confidence": 0.6, "evidence": ["说了'我来汇报一下进度'"], "role": "汇报人"}}
}}

**重要规则：**
1. 【严禁编造名字】除非对话中真的出现该名字
2. 【严禁凭空推断角色】role 必须有对话原文支撑，evidence 必须引用实际对话内容
3. 如果无法从对话中判断身份，name 和 role 都设为 null
4. kind 只能是 "name"（有真实名字）或 "role"（有对话支撑的角色）
5. confidence 只有在有明确证据时才能大于 0.7

只输出 JSON："""

        logger.debug(f"LLM Prompt:\n{prompt}")
        
        # 调用 LLM
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=self._settings.temperature,
            max_tokens=self._settings.max_tokens,
        )
        
        content = response["choices"][0]["message"]["content"]
        logger.debug(f"LLM Response:\n{content}")
        
        # 解析 JSON
        try:
            # 尝试提取 JSON 部分
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                result = json.loads(json_match.group())
                return result
        except json.JSONDecodeError as e:
            logger.warning(f"LLM 返回的 JSON 解析失败: {e}")
        
        return {}


# 模块级别的单例
_service: NamingService | None = None


def get_naming_service() -> NamingService:
    """获取命名服务的单例"""
    global _service
    if _service is None:
        _service = NamingService()
    return _service
