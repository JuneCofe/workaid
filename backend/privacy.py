"""Privacy filter applied before anything is stored or sent to an external service.

Two classes of personal information are handled differently:

* NEVER_STORE (ID card, phone, bank card, e-mail, passport, plate, social accounts, IP): always replaced
  by a placeholder, even if the user asks to remember it.
* OPT_IN (name, employer name, detailed address): replaced by a placeholder in chat text, and kept only
  as a profile item when the user explicitly asks to remember it ("请记住…").

Province/city level locations are kept: jurisdiction and the aid route depend on them. Detection is
rule-based (checksums for ID and bank card numbers), so it works offline and never depends on a model.
Names, employers and addresses are best-effort heuristics; the pattern-based kinds are reliable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

NEVER_STORE = frozenset({"身份证号", "手机号", "座机号", "银行卡号", "邮箱", "护照号", "车牌号", "社交账号", "IP地址"})
OPT_IN = frozenset({"姓名", "单位", "地址"})
_PLACEHOLDERS = {"地址": "[详细地址]"}

SURNAMES = (
    "王李张刘陈杨黄赵吴周徐孙马朱胡郭何高林罗郑梁谢宋唐许韩冯邓曹彭曾萧田董袁潘于蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金"
    "石廖贾夏韦付方白邹孟熊秦邱江尹薛闫段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴莫孔汤"
    # 向 is a surname but far more often the preposition ("申请人向仲裁委"), so it is not listed.
    # Rarer surnames. Characters that are also everyday words right after a role ("申请", "尚未", "连续",
    # "厉害", "岳父", "晋升") are left out on purpose: a missed rare name costs less than mangling a sentence.
    "邢滕蔺柴瞿霍窦尉仇戎祖詹屠卓乔胥翟姬雍桑桂濮燕冀晏阎茹艾庾匡寇禄隆聂晁敖沙鞠巢蒯荆竺桓楚鄢钦佟柯"
    "伍葛樊凌虞裴翁荀羊甄靳储芮邴糜娄骆栾"
)
COMPOUND_SURNAMES = (
    "欧阳 司马 诸葛 慕容 尉迟 上官 夏侯 皇甫 东方 令狐 公孙 独孤 南宫 轩辕 长孙 宇文 端木 澹台 百里 呼延 西门 东郭 南门 闻人 赫连 "
    "公羊 太史 申屠 公冶 宗政 濮阳 淳于 仲孙 钟离 谷梁 拓跋 夹谷 巫马 梁丘 左丘 微生 乐正 万俟 司寇 亓官"
).split()
_S = f"(?:{'|'.join(COMPOUND_SURNAMES)}|[{SURNAMES}])"
_HANZI = r"[一-鿿]"
# Words that start with a surname character but are not names ("马上", "高兴", "方便" ...).
_NOT_NAMES = frozenset(
    "马上 高兴 高薪 高温 白天 方便 方面 方式 方法 于是 可能 可以 可是 向来 何必 何况 何时 长期 安排 安全 常常 常见 范围 关于 关系 关键 "
    "管理 金额 金钱 毛病 田地 文件 文化 程序 程度 汤圆 林业 石头 万一 万分 严重 严格 武力 孔雀 白色 金融 钟点 夏天 秦朝 史上 龙头 "
    "高度 高级 余下 任何 任意 任务 周围 周期 周末 周一 周二 周三 周四 周五 周六 周日 许可 许多 黄色 蒋 郭 "
    "曾经 莫名 莫非 艾滋 沙发 柴火 乔装 燕子 楚楚 耿直 牟利 符合 凌晨".split()
)
_GENERIC_ORG_PREFIX = frozenset(
    "老板 领导 同事 我们 你们 他们 她们 这家 那家 一家 别家 原 前 现 新 旧 该 本 某 什么 哪家 别的 其他 另一 当地 大 小 贵 敝 "
    "本公司 该公司 这个 那个 一个 每个 整个 全部 所有".split()
)
# Words that end the text *before* a name. Single characters that occur inside real names ("华为" 为,
# "和平" 和, "成都" 都, "于", "与", "到", "从", "对", "去", "来", "过") are deliberately absent: cutting
# there would leave half of a company name unredacted. Missing a cut only over-redacts a few words.
_CUT_WORDS = (
    "入职", "离职", "任职", "就职", "辞职", "上班", "打工", "应聘", "加入", "进入", "进了", "来到", "调到", "转到", "跳槽", "起诉", "投诉",
    "举报", "告了", "在", "是", "的", "给", "向", "跟", "把", "让", "叫", "就", "还", "又", "也", "要", "想", "说", "被", "我", "你", "他",
    "她", "们", "了", "着", "，", "。", "、", "；", "：", " ",
)
_REGION_SUFFIX = "省市区县州旗盟"
# Function words that cannot be part of a proper company name and so mark where a name starts.
_STOP_CHARS = "的了吗么呢吧能会要想说什怎这那哪谁我你他她它们被把让给对向跟或是不没别就才也还又很太更最在"
REGIONS = (
    "北京 天津 上海 重庆 河北 山西 辽宁 吉林 黑龙江 江苏 浙江 安徽 福建 江西 山东 河南 湖北 湖南 广东 海南 四川 贵州 云南 陕西 甘肃 青海 "
    "台湾 内蒙古 广西 西藏 宁夏 新疆 香港 澳门 深圳 广州 杭州 南京 苏州 成都 武汉 西安 郑州 长沙 青岛 宁波 厦门 福州 济南 合肥 东莞 佛山 "
    "无锡 徐州 南通 常州 昆明 沈阳 大连 哈尔滨 长春 石家庄 太原 南昌 贵阳 南宁 兰州 乌鲁木齐 呼和浩特 银川 西宁 拉萨 海口 珠海 温州 嘉兴 "
    "绍兴 泉州 烟台 唐山 洛阳 惠州 中山 江门 扬州 盐城 台州 金华 潍坊 临沂 保定 邯郸 襄阳 宜昌 株洲 柳州 桂林 三亚 遵义 绵阳 乌鲁木齐"
).split()
_REGIONS_LONGEST_FIRST = sorted(set(REGIONS), key=len, reverse=True)

_ID_CARD = re.compile(r"(?<![0-9A-Za-z])[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[0-9Xx](?![0-9A-Za-z])")
_ID_KEYWORD = re.compile(r"(?:身份证(?:号码?)?|证件号码?)\s*[:：是为]?\s*([0-9Xx][0-9Xx \-]{13,21}[0-9Xx])")
_PHONE = re.compile(r"(?<!\d)(?:\+?86[ -]?)?1[3-9]\d[ -]?\d{4}[ -]?\d{4}(?!\d)")
_LANDLINE = re.compile(r"(?<!\d)(?:0\d{2,3}[ -]\d{7,8}|0\d{10,11}|[48]00[ -]?\d{3}[ -]?\d{4})(?!\d)")
_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PLATE = re.compile(r"[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领][A-HJ-NP-Z][A-HJ-NP-Z0-9]{4,5}[A-HJ-NP-Z0-9挂学警港澳]")
_SOCIAL = re.compile(r"(?:微信号?|微信|vx|VX|wx|WX|QQ号?|qq号?|支付宝(?:账号)?)\s*[:：是为]?\s*([A-Za-z0-9_\-]{5,30})")
_PASSPORT = re.compile(r"护照(?:号码?)?\s*[:：是为]?\s*([A-Za-z]?\d{7,9})")
_IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
_CARD_CANDIDATE = re.compile(r"(?<!\d)(?:\d[ -]?){15,18}\d(?!\d)")
_GROUPED_CARD = re.compile(r"\d{4}(?:[ -]\d{4}){3}(?:[ -]\d{1,3})?")
_CARD_CONTEXT = re.compile(r"(?:卡号|银行卡|账号|账户|开户|信用卡|储蓄卡|转账|汇到|汇款|汇至|打款|收款|付款|转到|转入|转给)\D{0,6}$")

_NAME_DECLARED = re.compile(
    rf"(?:我叫|我的名字是|我的名字叫|名字叫|名叫|[他她]叫|姓名[:：]\s*|本人姓名[:：]\s*)({_S}{_HANZI}{{1,2}})"
    # "我是薛佳琪，" — only when a name could end right there, so "我是高中生" is left alone.
    rf"|我是({_S}{_HANZI}{{1,2}})(?=[，。、；,\s]|\Z)"
)
_SURNAME_ONLY = re.compile(rf"我姓({_S})(?!{_HANZI}?[先女])")
_ROLES = (
    "老板|领导|经理|主管|总监|同事|房东|HR|人事|律师|法官|仲裁员|对方|前夫|前妻|丈夫|妻子|男朋友|女朋友|客户|工友|师傅|负责人|法人|联系人|当事人"
    "|申请人|被申请人|代理人|股东|法定代表人|项目组长|组长|班长|工头|包工头|队长|厂长|店长|主任|科长|部长|处长|院长|校长|会计|出纳|前台|秘书"
    "|助理|邻居|朋友|同学|老公|老婆|亲戚|女儿|儿子|父亲|母亲|爸爸|妈妈|护士|保安|承包人|中介|业主"
)
_AFTER_NAME = r"[，。、；：！？,.;:!?\s说称让把给被是的在又还也就却不没要到一和与跟等]"
# A three-character name must be followed by punctuation or a particle; otherwise the two-character
# reading wins ("张三一直" is 张三 + 一直, "王小明一直" is 王小明 + 一直).
_ROLE_NAME = re.compile(
    rf"(?:{_ROLES})(?:(?P<n3>{_S}{_HANZI}{{2}})(?={_AFTER_NAME}|\Z)|(?P<n2>{_S}{_HANZI}))"
)
_TITLE_NAME = re.compile(rf"({_S}(?:{_HANZI}{{1,2}})?)(?=先生|女士|经理|老师|主管|师傅|律师|法官|老板|医生)")
# "方" is a surname but, right after these characters, a word of its own ("对方律师", "双方老板").
_FANG_WORDS = frozenset("对双地单一我甲乙各多他你其被原")

_FORMAL_SUFFIX = frozenset({"股份有限公司", "有限责任公司", "有限公司", "分公司", "集团", "事务所"})
# A bare "公司/饭店/医院" only names an employer when something like "入职/在/应聘" leads into it.
# Single characters that occur inside real names ("华为" 为, "和平" 和) are not cues.
_EMPLOYMENT_CUES = (
    "入职", "就职", "任职", "加入", "进入", "进了", "应聘", "跳槽到", "调到", "转到", "来到", "去了", "被", "在", "去", "到",
)
_ORG = re.compile(
    r"(?P<pre>[一-鿿A-Za-z0-9（）()·]{2,24}?)"
    r"(?P<suf>股份有限公司|有限责任公司|有限公司|分公司|集团|事务所|公司|工厂|医院|学校|酒店|饭店|餐厅|超市|商场|门店|工作室|协会|银行|网吧|农场"
    r"|制衣厂|服装厂|电子厂|机械厂|化工厂|食品厂|纺织厂|印刷厂|塑料厂|加工厂|制造厂|玩具厂|家具厂|五金厂|钢厂|砖厂)"
)
# Descriptions of a kind of employer ("劳务派遣公司", "大型连锁超市"): not a name, nothing to redact.
_GENERIC_ORG_WORDS = (
    "劳务派遣", "大型", "小型", "中型", "连锁", "国有", "私营", "民营", "外资", "合资", "当地", "本地", "正规", "非法", "黑",
    "电子", "机械", "化工", "食品", "纺织", "印刷", "塑料", "服装", "玩具", "家具", "五金", "人力资源", "物业", "餐饮", "建筑", "运输", "物流",
)
_ADDRESS = re.compile(
    r"(?P<pre>[一-鿿]{1,14})"
    r"(?P<road>路|街|大道|巷|弄|村|镇|乡|小区|花园|苑|大厦|广场|园区|工业园|产业园|开发区|社区)"
    r"(?P<rest>[一-鿿A-Za-z0-9\-—（）()]{0,16}?"
    r"(?:(?:\d{1,5}[-—]?\d{0,5}\s*(?:号|室|栋|幢|座|单元|楼|层|户))+|\d{1,5}[-—]\d{1,5}(?:[-—]\d{1,5})?))"
)
_REMEMBER = re.compile(
    r"(请|麻烦|帮我|能不能|可以)?(记住|记下|记一下|长期记住|永久记住)|存为(我的)?(资料|信息|档案|画像)"
    r"|保存(到|为|成|进)(我的)?(资料|信息|档案|画像|长期记忆)|长期(记住|保存)"
)


def placeholder(kind: str) -> str:
    return _PLACEHOLDERS.get(kind, f"[{kind}]")


@dataclass
class Analysis:
    text: str
    counts: dict[str, int] = field(default_factory=dict)
    entities: list[tuple[str, str]] = field(default_factory=list)
    remember: bool = False

    @property
    def changed(self) -> bool:
        return bool(self.counts)


def _luhn(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        number = int(char)
        if index % 2:
            number *= 2
            number -= 9 if number > 9 else 0
        total += number
    return total % 10 == 0


def _cut_after_tokens(text: str) -> int:
    """Index just past the last verb/particle in text, so "我在深圳华为公司" starts at the company."""
    cut = 0
    for token in _CUT_WORDS:
        position = text.rfind(token)
        if position >= 0:
            cut = max(cut, position + len(token))
    return cut


def _strip_generic(prefix: str) -> str:
    """What is left of an employer's prefix after removing descriptive words; empty means no name."""
    rest = prefix
    for word in sorted(_GENERIC_ORG_WORDS, key=len, reverse=True):
        rest = rest.replace(word, "")
    return rest.strip()


def wants_to_remember(text: str) -> bool:
    return bool(_REMEMBER.search(text))


def _collect(text: str) -> list[tuple[int, int, str, str]]:
    """(start, end, kind, value) for every sensitive span, in priority order."""
    spans: list[tuple[int, int, str, str]] = []

    def add(match_start: int, match_end: int, kind: str, value: str | None = None) -> None:
        spans.append((match_start, match_end, kind, text[match_start:match_end] if value is None else value))

    for match in _ID_CARD.finditer(text):
        add(match.start(), match.end(), "身份证号")
    for match in _ID_KEYWORD.finditer(text):
        add(match.start(1), match.end(1), "身份证号")
    for match in _CARD_CANDIDATE.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        if 16 <= len(digits) <= 19 and (
            _luhn(digits)
            or digits.startswith("62")  # UnionPay, the common domestic card; no legal-case number looks like this
            or _GROUPED_CARD.fullmatch(match.group(0))  # 4-4-4-4 is how cards are written
            or _CARD_CONTEXT.search(text[max(0, match.start() - 10) : match.start()])
        ):
            add(match.start(), match.end(), "银行卡号")
    for match in _PHONE.finditer(text):
        add(match.start(), match.end(), "手机号")
    for match in _LANDLINE.finditer(text):
        add(match.start(), match.end(), "座机号")
    for match in _EMAIL.finditer(text):
        add(match.start(), match.end(), "邮箱")
    for match in _PASSPORT.finditer(text):
        add(match.start(1), match.end(1), "护照号")
    for match in _SOCIAL.finditer(text):
        add(match.start(1), match.end(1), "社交账号")
    for match in _PLATE.finditer(text):
        add(match.start(), match.end(), "车牌号")
    for match in _IPV4.finditer(text):
        if all(int(part) <= 255 for part in match.group(0).split(".")):
            add(match.start(), match.end(), "IP地址")

    for match in _ADDRESS.finditer(text):
        head = match.group("pre")
        token_cut = _cut_after_tokens(head)
        region_cut = max([token_cut] + [i + 1 for i, ch in enumerate(head) if ch in _REGION_SUFFIX])
        if len(head) - region_cut + len(match.group("road")) < 2:
            continue
        start = match.start() + region_cut
        add(start, match.end(), "地址", text[match.start() + token_cut : match.end()])

    for match in _ORG.finditer(text):
        prefix = match.group("pre")
        informal = match.group("suf") not in _FORMAL_SUFFIX
        cued = False
        if informal:
            cue_end = max((prefix.rfind(cue) + len(cue) for cue in _EMPLOYMENT_CUES if cue in prefix), default=0)
            if cue_end:
                prefix, cued = prefix[cue_end:], True
        prefix = prefix[_cut_after_tokens(prefix) :]
        for region in _REGIONS_LONGEST_FIRST:
            if prefix.startswith(region) and len(prefix) - len(region) >= 2:
                keep = len(region) + (1 if prefix[len(region)] in "市省区县" else 0)
                if len(prefix) - keep >= 2:
                    prefix = prefix[keep:]
                break
        if informal:
            # "制裁公司" must not read as an employer: take only what follows the last function word,
            # and without a cue like "在/入职/被" require a name of at least three characters.
            prefix = prefix[max((prefix.rfind(ch) + 1 for ch in _STOP_CHARS), default=0) :]
            if len(prefix) < (2 if cued else 3):
                continue
        if (
            len(prefix) < 2
            or "某" in prefix
            or prefix in _GENERIC_ORG_PREFIX
            or any(prefix.startswith(word) for word in _GENERIC_ORG_PREFIX if len(word) >= 2)
            or re.fullmatch(r"[Xx×*＊○O0]+", prefix)
            or not _strip_generic(prefix)
        ):
            continue
        start = match.end() - len(prefix) - len(match.group("suf"))
        add(start, match.end(), "单位")

    for match in _NAME_DECLARED.finditer(text):
        group = 1 if match.group(1) else 2
        add(match.start(group), match.end(group), "姓名")
    for match in _SURNAME_ONLY.finditer(text):
        add(match.start(1), match.end(1), "姓名")
    for match in _ROLE_NAME.finditer(text):
        group = "n3" if match.group("n3") else "n2"
        name = match.group(group)
        if name not in _NOT_NAMES and name[:2] not in _NOT_NAMES:
            add(match.start(group), match.end(group), "姓名")
    for match in _TITLE_NAME.finditer(text):
        name = match.group(1)
        before = text[match.start(1) - 1] if match.start(1) > 0 else ""
        if name in _NOT_NAMES or (name == "方" and before in _FANG_WORDS):
            continue
        add(match.start(1), match.end(1), "姓名")
    return spans


def analyze(text: str, allow: frozenset[str] = frozenset()) -> Analysis:
    """Redact `text`; kinds listed in `allow` (only OPT_IN kinds) are left as written.

    Overlapping spans keep the earliest-priority one, so a phone number inside an address stays a
    phone number. The original text of OPT_IN spans is returned in `entities` for the caller to
    store *only* if the user asked to be remembered.
    """
    allow = frozenset(kind for kind in allow if kind in OPT_IN)
    chosen: list[tuple[int, int, str, str]] = []
    for span in _collect(text):
        if span[2] in allow:
            continue
        if any(span[0] < other[1] and other[0] < span[1] for other in chosen):
            continue
        chosen.append(span)
    counts: dict[str, int] = {}
    entities: list[tuple[str, str]] = []
    out = text
    for start, end, kind, value in sorted(chosen, key=lambda s: s[0], reverse=True):
        out = out[:start] + placeholder(kind) + out[end:]
        counts[kind] = counts.get(kind, 0) + 1
        if kind in OPT_IN:
            entities.append((kind, value))
    entities.reverse()
    return Analysis(out, counts, entities, wants_to_remember(text))


def redact_text(text: str, allow: frozenset[str] = frozenset()) -> str:
    return analyze(text, allow).text


def redact_stored(text: str) -> str:
    """For text a person typed into their own profile: opt-in kinds stay, never-store kinds go."""
    return analyze(text, allow=OPT_IN).text
