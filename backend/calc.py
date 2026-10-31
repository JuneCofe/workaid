"""Deterministic legal arithmetic: limitation periods, double wages, severance, injury deadlines.

Every figure is computed here, never by a language model. Each result names the statute it rests on
(so the answer verifier can check the citation) and lists the assumptions a person must confirm.
Dates may be "YYYY-MM" or "YYYY-MM-DD"; when the day is unknown the earliest possible day is used so
a deadline is never overstated.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

ONGOING = "未离职"
MAX_DOUBLE_WAGE_MONTHS = 11


@dataclass
class Result:
    key: str
    title: str
    summary: str
    details: list[str] = field(default_factory=list)
    basis: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.key, "title": self.title, "summary": self.summary, "details": self.details,
                "basis": self.basis, "caveats": self.caveats}


def parse_date(value: str | None) -> tuple[date, bool] | None:
    """(date, day_is_exact) from "2024-03" or "2024-03-15"; None if it is not a usable date."""
    match = re.fullmatch(r"\s*(\d{4})-(\d{1,2})(?:-(\d{1,2}))?\s*", value or "")
    if not match:
        return None
    year, month = int(match.group(1)), int(match.group(2))
    if not (1990 <= year <= 2100 and 1 <= month <= 12):
        return None
    day = match.group(3)
    try:
        return date(year, month, int(day) if day else 1), bool(day)
    except ValueError:
        return None


def add_months(start: date, months: int) -> date:
    index = start.year * 12 + start.month - 1 + months
    year, month = divmod(index, 12)
    month += 1
    return date(year, month, min(start.day, calendar.monthrange(year, month)[1]))


def full_months(start: date, end: date) -> int:
    """Whole months from start to end (negative if end is earlier)."""
    months = (end.year - start.year) * 12 + end.month - start.month
    if end.day < start.day:
        months -= 1
    return months


def money(value: float) -> str:
    return f"{value:,.0f}元" if value == int(value) else f"{value:,.2f}元"


def _fmt(day: date) -> str:
    return f"{day.year}年{day.month}月{day.day}日"


def arbitration_limit(event: str | None, today: date, *, arrears_ongoing: bool = False) -> Result | None:
    """One-year limitation for labour arbitration (劳动争议调解仲裁法第二十七条)."""
    basis = ["《劳动争议调解仲裁法》第二十七条"]
    caveats = ["时效可因向对方主张权利、向有关部门请求救济等事由中断，也可能因不可抗力中止，请结合证据核实。"]
    if arrears_ongoing:
        return Result(
            "arbitration_limit", "仲裁时效",
            "劳动关系还在存续期间因拖欠劳动报酬发生争议的，不受一年仲裁时效限制；关系终止后，应当自终止之日起一年内提出。",
            ["建议不要等到离职后才处理，保留好每月未发工资的证据。"], basis, caveats,
        )
    parsed = parse_date(event)
    if parsed is None:
        return None
    start, exact = parsed
    deadline = add_months(start, 12)
    remaining = (deadline - today).days
    if remaining >= 0:
        summary = f"仲裁时效一年，最迟应在 {_fmt(deadline)} 前申请仲裁，距今还剩约 {remaining} 天。"
    else:
        summary = f"按 {_fmt(start)} 起算，一年时效在 {_fmt(deadline)} 届满，现已超过约 {-remaining} 天；是否仍可主张取决于是否有时效中断、中止等情形，请尽快咨询律师或仲裁委。"
    details = [] if exact else ["只知道年月，按当月 1 日起算，这是最保守的估计，实际期限可能更晚。"]
    return Result("arbitration_limit", "仲裁时效", summary, details, basis, caveats)


def double_wage(entry: str | None, exit_value: str | None, pay: int | None, today: date) -> Result | None:
    """Double wages for not signing a written contract (劳动合同法第八十二条, 实施条例第六、七条)."""
    start = parse_date(entry)
    if start is None:
        return None
    first, _ = start
    ended = parse_date(exit_value) if exit_value and exit_value != ONGOING else None
    end = ended[0] if ended else today
    liable_from = add_months(first, 1)
    liable_to = min(end, add_months(first, 12))
    months = max(0, min(MAX_DOUBLE_WAGE_MONTHS, full_months(liable_from, liable_to)))
    basis = ["《劳动合同法》第八十二条", "《劳动合同法实施条例》第六条", "《劳动合同法实施条例》第七条"]
    caveats = [
        "用工第一个月内未签合同不算违法；满一年仍未签，视为已订立无固定期限劳动合同，二倍工资最多计 11 个月。",
        "已经发放的正常工资应当扣除，实务中通常主张的是“额外一倍”的差额，具体以当地仲裁和法院口径为准。",
        "二倍工资的请求同样受一年仲裁时效限制，要分月看是否已过。",
    ]
    if months == 0:
        return Result("double_wage", "未签书面劳动合同的二倍工资", "按目前信息，还不到可以主张二倍工资的期间（用工满一个月后才开始计算）。", [], basis, caveats)
    details = [f"计算区间：{_fmt(liable_from)} 至 {_fmt(liable_to)}，共 {months} 个整月。"]
    if pay:
        details.append(f"按月薪 {money(pay)} 估算，额外一倍的差额约 {money(months * pay)}。")
    else:
        details.append("提供月薪后可以算出大致金额。")
    return Result("double_wage", "未签书面劳动合同的二倍工资", f"可主张二倍工资的期间约 {months} 个月。", details, basis, caveats)


def severance(entry: str | None, exit_value: str | None, pay: int | None, *, unlawful: bool) -> Result | None:
    """Economic compensation N and, for unlawful termination, the 2N damages (劳动合同法第四十七、八十七条)."""
    start, end = parse_date(entry), parse_date(exit_value) if exit_value and exit_value != ONGOING else None
    if start is None or end is None:
        return None
    months = full_months(start[0], end[0])
    if months < 0:
        return None
    years, rest = divmod(months, 12)
    n = years + (1 if rest >= 6 else 0.5 if rest > 0 else 0)
    if months == 0:
        n = 0.5
    basis = ["《劳动合同法》第四十七条"] + (["《劳动合同法》第八十七条", "《劳动合同法》第四十八条"] if unlawful else [])
    caveats = [
        "月工资应按离职前十二个月的平均工资计算，包含奖金、津贴等；这里用您提供的月薪估算。",
        "月工资高于当地上年度职工月平均工资三倍的，按三倍计算且年限最高十二年；当地平均工资需另行查询。",
        "2008 年 1 月 1 日以前的工作年限，补偿规则不同，请向律师确认。",
    ]
    details = [f"工作年限约 {years} 年 {rest} 个月，折算补偿基数 N = {n:g} 个月工资。"]
    if pay:
        details.append(f"按月薪 {money(pay)} 估算，经济补偿约 {money(n * pay)}。")
        if unlawful:
            details.append(f"若公司属于违法解除，赔偿金为经济补偿的二倍，约 {money(2 * n * pay)}（二者不能同时主张）。")
    summary = f"经济补偿的计算基数是 {n:g} 个月工资" + ("；违法解除可主张二倍赔偿金。" if unlawful else "。")
    return Result("severance", "经济补偿（赔偿金）", summary, details, basis, caveats)


def injury_deadlines(injury: str | None, today: date) -> Result | None:
    """Work-injury recognition deadlines (工伤保险条例第十七条)."""
    parsed = parse_date(injury)
    if parsed is None:
        return None
    start, exact = parsed
    employer_by = start.fromordinal(start.toordinal() + 30)
    person_by = add_months(start, 12)
    remaining = (person_by - today).days
    summary = (
        f"用人单位应在事故发生后 30 日内（{_fmt(employer_by)} 前）申请工伤认定；单位没有申请的，本人或亲属可以在一年内（{_fmt(person_by)} 前）自行申请"
        + (f"，距今还剩约 {remaining} 天。" if remaining >= 0 else f"，现已超过约 {-remaining} 天，请尽快咨询。")
    )
    details = [] if exact else ["只知道年月，按当月 1 日起算，是最保守的估计。"]
    caveats = ["因单位拖延或不可抗力等耽误的时间，通常不计入申请期限，请保留相关证明。"]
    return Result("injury_deadline", "工伤认定申请期限", summary, details, ["《工伤保险条例》第十七条"], caveats)


def wage_arrears_remedy(months: int | None, pay: int | None) -> Result:
    """Remedies for unpaid wages (劳动合同法第三十条、第三十八条、第八十五条)."""
    details = []
    if months and pay:
        details.append(f"按月薪 {money(pay)}、拖欠 {months} 个月估算，欠付工资约 {money(months * pay)}。")
    details += [
        "可以向劳动保障监察部门（12333 或当地监察大队）投诉，也可以申请劳动仲裁或向法院申请支付令。",
        "用人单位逾期不支付的，劳动保障部门可责令按应付金额 50% 以上 100% 以下加付赔偿金。",
        "因拖欠工资解除劳动合同的，劳动者可以主张经济补偿。",
    ]
    return Result("wage_arrears", "拖欠工资的救济", "拖欠劳动报酬可以通过投诉、仲裁或支付令主张，并可能加付赔偿金。", details,
                  ["《劳动合同法》第三十条", "《劳动合同法》第三十八条", "《劳动合同法》第八十五条"],
                  ["欠薪金额以工资条、银行流水、考勤等证据能证明的部分为准。"])
