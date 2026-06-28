"""
依存树提取器 —— 从自然语言提取句法结构

白盒原则：
    仅使用句法依存解析（结构分析），严禁情感分析（语义黑盒）。
    spaCy 依存解析器基于转移式/图式解析算法，输出语法角色，不输出情感分数。
    jieba 词性标注基于 HMM + 词典，输出语法类别，不输出情感分数。

两种后端：
    1. spaCy（首选）：完整依存树，精确的主-谓-宾拓扑
    2. jieba（备选）：词性标注 + 规则匹配，近似的主-谓-宾结构

提取的结构信息（全部为语法性质，无情感内容）：
    - 动词原形 + 配价 + 及物性 + 体态
    - 主语人称
    - 宾语语义类别（NER/POS 推导）
    - 依存树深度 + 修饰语数量 + 否定检测
    - 时间频率 + 时间周期
"""

from __future__ import annotations

import re
from typing import Any

from .force_ontology import StructuralFeatures


class DependencyExtractor:
    """
    依存树提取器：自然语言 → StructuralFeatures。

    支持双后端：
        - spaCy（首选，完整依存树）
        - jieba（备选，词性标注 + 规则匹配）
    """

    def __init__(self, backend: str = "auto"):
        """
        参数：
            backend: "spacy" / "jieba" / "auto"（自动选择可用后端）
        """
        self._spacy_nlp = None
        self._jieba = None
        self._backend = backend

        if backend in ("auto", "spacy"):
            self._spacy_nlp = self._try_load_spacy()

        if self._spacy_nlp is None and backend in ("auto", "jieba"):
            self._jieba = self._try_load_jieba()

        if self._spacy_nlp is not None:
            self._active_backend = "spacy"
        elif self._jieba is not None:
            self._active_backend = "jieba"
        else:
            raise RuntimeError("无可用 NLP 后端：spaCy 和 jieba 均未安装")

    def _try_load_spacy(self):
        """尝试加载 spaCy 中文模型。"""
        try:
            import spacy
            nlp = spacy.load("zh_core_web_sm")
            return nlp
        except Exception:
            return None

    def _try_load_jieba(self):
        """尝试加载 jieba 词性标注。"""
        try:
            import jieba.posseg as pseg
            return pseg
        except Exception:
            return None

    @property
    def backend(self) -> str:
        return self._active_backend

    def extract(self, text: str) -> StructuralFeatures:
        """
        从自然语言文本提取结构特征。

        参数：
            text: 中文叙事文本（如"父亲每周严厉批评我三次"）

        返回：
            StructuralFeatures: 纯语法性质的结构特征
        """
        if self._active_backend == "spacy":
            return self._extract_spacy(text)
        else:
            return self._extract_jieba(text)

    def _extract_spacy(self, text: str) -> StructuralFeatures:
        """
        spaCy 后端：完整依存树分析。

        提取：
            - ROOT 动词 + 其依存关系（nsubj/obj/advmod/nummod/neg）
            - 依存树深度（DFS）
            - 修饰语计数
            - 否定检测
            - 时间频率（nummod + 时间实体）
        """
        doc = self._spacy_nlp(text)

        # 找到 ROOT 动词
        root = None
        for token in doc:
            if token.dep_ == "ROOT":
                root = token
                break

        if root is None:
            # 回退：取第一个动词
            for token in doc:
                if token.pos_ in ("VERB", "AUX"):
                    root = token
                    break

        if root is None:
            return StructuralFeatures(verb_lemma=text[:10])

        features = StructuralFeatures(verb_lemma=root.lemma_ or root.text)

        # 配价与及物性
        children = list(root.children)
        features.valency = len(children)
        features.is_transitive = any(c.dep_ in ("obj", "dobj", "obj:animal") for c in children)

        # 情态/助动词
        features.is_modal = root.pos_ == "AUX" or root.lemma_ in ("要", "能", "会", "可以", "应该", "必须", "想", "敢")

        # 状态动词检测（进行体/持续体标记）
        aspect_markers = {"在", "正在", "着", "一直", "始终"}
        features.is_stative = any(c.text in aspect_markers for c in children) or root.lemma_ in ("是", "有", "存在", "属于", "等于")

        # 依存树深度（DFS）
        features.tree_depth = self._tree_depth(root)

        # 修饰语计数
        mod_deps = {"advmod", "amod", "nummod", "compound", "nmod"}
        features.n_modifiers = sum(1 for c in children if c.dep_ in mod_deps)
        # 递归统计所有后代中的修饰语
        features.n_modifiers += self._count_modifiers_recursive(root, mod_deps)

        # 否定检测
        features.has_negation = any(c.dep_ == "neg" or c.text in ("不", "没", "无", "未", "别", "勿") for c in children)

        # 主语人称
        for c in children:
            if c.dep_ in ("nsubj", "nsubjpass"):
                features.subject_person = self._detect_person(c)
                break

        # 宾语语义类别
        for c in children:
            if c.dep_ in ("obj", "dobj", "obj:animal"):
                features.object_category = self._detect_object_category(c)
                break

        # 时间频率与周期
        count, period = self._extract_temporal_spacy(doc)
        features.temporal_count = count
        features.temporal_period = period

        return features

    def _extract_jieba(self, text: str) -> StructuralFeatures:
        """
        jieba 后端：词性标注 + 规则匹配。

        规则：
            - 找第一个动词（POS 以 'v' 开头）
            - 动词前的代词/名词 = 主语
            - 动词后的代词/名词 = 宾语
            - 数词 + 时间词 = 时间频率
            - 副词 = 修饰语
            - 被动语态"被" → 状态约束
            - 情态动词"必须/应该/可以" → 势能场
        """
        words = list(self._jieba.cut(text))

        features = StructuralFeatures()
        verb_idx = -1

        # 情态动词检测（检查所有词，jieba可能将情态动词标为副词d）
        modal_words = {"要", "能", "会", "可以", "应该", "必须", "想", "敢", "需要", "可能", "得", "该"}
        for i, (word, flag) in enumerate(words):
            if word in modal_words:
                features.is_modal = True
                # 情态动词作为主动词
                if verb_idx == -1:
                    features.verb_lemma = word
                    verb_idx = i
                break

        # 被动语态检测："被"字结构 → 状态约束
        has_passive = any(w == "被" for w, f in words)
        # "被关在...里"表示约束状态
        constraint_patterns = {"被关", "被困", "被锁", "被绑", "被限制", "被束缚"}
        full_text = "".join(w for w, f in words)
        has_constraint = any(p in full_text for p in constraint_patterns)

        # 找动词（若情态动词已作为主动词，则找后续实义动词）
        if verb_idx == -1 or features.is_modal:
            for i, (word, flag) in enumerate(words):
                if verb_idx != -1 and i <= verb_idx:
                    continue  # 跳过情态动词之前
                if flag.startswith("v") or flag == "aux":
                    if verb_idx == -1:
                        features.verb_lemma = word
                        verb_idx = i
                        if not features.is_modal:
                            features.is_modal = flag == "aux" or word in modal_words
                    else:
                        # 情态动词后的实义动词，更新verb_lemma但保持is_modal
                        features.verb_lemma = word
                    break

        if verb_idx == -1:
            return features

        # 主语（动词前）
        for i in range(verb_idx - 1, -1, -1):
            word, flag = words[i]
            if flag.startswith("r") or flag.startswith("n") or flag.startswith("p"):
                features.subject_person = self._detect_person_jieba(word, flag)
                break

        # 宾语（动词后）
        for i in range(verb_idx + 1, len(words)):
            word, flag = words[i]
            if flag.startswith("n") or flag.startswith("r"):
                features.object_category = self._detect_object_category_jieba(word, flag)
                features.is_transitive = True
                break

        # 配价
        features.valency = 1 + (1 if features.is_transitive else 0) + (1 if features.subject_person > 0 else 0)

        # 状态动词
        stative_verbs = {"是", "有", "存在", "属于", "等于"}
        # "在"单独使用时是方位词，仅"正在"和"着"是进行体标记
        aspect_words = {"正在", "着", "一直", "始终"}
        features.is_stative = (
            features.verb_lemma in stative_verbs
            or any(w in aspect_words for w, f in words)
            or has_constraint  # 被动约束结构 → 状态
        )
        # 被动语态且含约束语义 → 强制状态动词
        if has_passive and has_constraint:
            features.is_stative = True
            features.is_transitive = False  # 被动约束非及物冲击
            features.valency = max(1, features.valency - 1)  # 降低配价

        # 修饰语计数（副词 + 形容词 + 数词）
        features.n_modifiers = sum(
            1 for w, f in words
            if f.startswith("a") or f.startswith("d") or f.startswith("m") or f.startswith("ad")
        )

        # 否定
        neg_words = {"不", "没", "无", "未", "别", "勿", "没有"}
        features.has_negation = any(w in neg_words for w, f in words)

        # 依存树深度（近似：句子结构复杂度）
        features.tree_depth = max(1, len([w for w, f in words if f.startswith("v") or f.startswith("n") or f.startswith("a")]))

        # 时间频率
        count, period = self._extract_temporal_jieba(words)
        features.temporal_count = count
        features.temporal_period = period

        return features

    def _tree_depth(self, token) -> int:
        """递归计算依存树深度（DFS）。"""
        children = list(token.children)
        if not children:
            return 1
        return 1 + max(self._tree_depth(c) for c in children)

    def _count_modifiers_recursive(self, token, mod_deps: set) -> int:
        """递归统计修饰语数量。"""
        count = 0
        for c in token.children:
            if c.dep_ in mod_deps:
                count += 1
            count += self._count_modifiers_recursive(c, mod_deps)
        return count

    def _detect_person(self, token) -> int:
        """检测主语人称（spaCy）。"""
        if token.text in ("我", "我们", "咱们"):
            return 1
        if token.text in ("你", "你们", "您"):
            return 2
        if token.pos_ == "PRON":
            return 3
        # 名词主语默认第三人称
        return 3

    def _detect_person_jieba(self, word: str, flag: str) -> int:
        """检测主语人称（jieba）。"""
        if word in ("我", "我们", "咱们"):
            return 1
        if word in ("你", "你们", "您"):
            return 2
        return 3

    def _detect_object_category(self, token) -> str:
        """
        检测宾语语义类别（spaCy NER/POS，非情感字典）。

        类别：
            first_person: 第一人称代词 → 自我指涉
            person: 人名/人称 → 依恋
            time: 时间实体 → 时间贴现
            abstract: 抽象名词 → 意义
            physical: 具体名词 → 威胁感知
            none: 无宾语
        """
        if token.text in ("我", "我们", "咱们"):
            return "first_person"
        if token.ent_type_ in ("PERSON", "NORP", "ORG"):
            return "person"
        if token.ent_type_ in ("DATE", "TIME"):
            return "time"
        if token.pos_ == "PRON":
            return "person"
        if token.pos_ == "NOUN":
            # 抽象名词 vs 具体名词：通过依存深度区分
            # 抽象概念通常有更多修饰语或更深嵌套
            n_children = len(list(token.children))
            if n_children >= 2:
                return "abstract"
            return "physical"
        return "none"

    def _detect_object_category_jieba(self, word: str, flag: str) -> str:
        """检测宾语语义类别（jieba POS，非情感字典）。"""
        if word in ("我", "我们", "咱们"):
            return "first_person"
        if word in ("你", "你们", "他", "她", "它", "他们", "她们", "它们"):
            return "person"
        if flag.startswith("t") or flag.startswith("nr"):
            return "time"
        if flag in ("n", "nr", "ns", "nt", "nz"):
            # 通过词长区分抽象/具体（启发式）
            if len(word) >= 3:
                return "abstract"
            return "physical"
        if flag.startswith("r"):
            return "person"
        return "none"

    def _extract_temporal_spacy(self, doc) -> tuple[float, float]:
        """
        提取时间频率与周期（spaCy）。

        "每周三次" → count=3, period=7
        "每天" → count=1, period=1
        "每月两次" → count=2, period=30

        周期单位：天
        """
        count = 1.0
        period = 1.0

        # 时间实体检测
        for ent in doc.ents:
            if ent.label_ in ("DATE", "TIME"):
                period = self._parse_period(ent.text)

        # 数词检测（频率）
        for token in doc:
            if token.dep_ == "nummod" and token.like_num:
                try:
                    count = float(token._.number) if hasattr(token._, "number") else float(token.text)
                except (ValueError, AttributeError):
                    # 尝试中文数字解析
                    count = self._parse_chinese_number(token.text)

        return count, period

    def _extract_temporal_jieba(self, words) -> tuple[float, float]:
        """提取时间频率与周期（jieba）。"""
        count = 1.0
        period = 1.0

        # 时间周期：检查所有词中是否含时间关键词
        # jieba 可能将"每周"标为 m/n 而非 t，因此用关键词匹配
        for word, flag in words:
            parsed_period = self._parse_period(word)
            if parsed_period != 1.0:
                period = parsed_period
            # 频率数词
            if flag.startswith("m"):
                num = self._parse_chinese_number(word)
                if num > 0:
                    count = num

        # 额外检查：合并的词如"每周"可能被分词为"每"+"周"
        # 或"三次"可能被分词为"三"+"次"
        full_text = "".join(w for w, f in words)
        # 检查完整文本中的时间周期
        if period == 1.0:
            period = self._parse_period(full_text)
        # 检查完整文本中的频率数词
        if count == 1.0:
            # 用正则提取数字
            import re
            nums = re.findall(r'\d+', full_text)
            if nums:
                count = float(nums[0])
            else:
                # 中文数字
                cn_nums = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
                           "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
                for char in full_text:
                    if char in cn_nums:
                        count = float(cn_nums[char])
                        break

        return count, period

    def _parse_period(self, text: str) -> float:
        """
        解析时间周期为天数。

        "每周" → 7, "每天" → 1, "每月" → 30, "每年" → 365
        "小时" → 1/24, "分钟" → 1/1440
        """
        period_map = {
            "天": 1.0, "日": 1.0,
            "周": 7.0, "星期": 7.0, "礼拜": 7.0,
            "月": 30.0,
            "年": 365.0,
            "小时": 1.0 / 24,
            "分钟": 1.0 / 1440,
            "秒": 1.0 / 86400,
        }
        for key, val in period_map.items():
            if key in text:
                return val
        return 1.0

    def _parse_chinese_number(self, text: str) -> float:
        """
        解析中文/阿拉伯数字。

        "三" → 3, "3" → 3, "十" → 10, "百" → 100
        "两" → 2, "万" → 10000
        """
        # 阿拉伯数字
        try:
            return float(text)
        except ValueError:
            pass

        # 中文数字
        digit_map = {
            "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
            "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
        }
        unit_map = {"十": 10, "百": 100, "千": 1000, "万": 10000}

        result = 0.0
        current = 0.0
        for char in text:
            if char in digit_map:
                current = digit_map[char]
            elif char in unit_map:
                if current == 0:
                    current = 1
                result += current * unit_map[char]
                current = 0
            else:
                result += current
                current = 0
        result += current

        return result if result > 0 else 1.0
