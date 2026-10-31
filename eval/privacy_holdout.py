"""Held-out privacy check: sentences written after the rules were tuned, never edited to fit them.

    python -I eval/privacy_holdout.py

Add new sentences here when a real miss is found, then fix the rules; the first run on each new
batch is the honest number. Positives: every listed secret must disappear. Negatives: unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend import privacy  # noqa: E402

POSITIVE = [
    ("项目组长陆志远每次都说月底发，到现在也没发", ["陆志远"]), ("我是薛佳琪，在无锡新区的锦绣家纺有限公司做挡车工", ["薛佳琪", "锦绣家纺有限公司"]),
    ("法定代表人伍德海拒绝出庭", ["伍德海"]), ("申请人曹雨桐向仲裁委提交了材料", ["曹雨桐"]), ("人事部的钱晓燕通知我明天不用来了", ["钱晓燕"]),
    ("我的收款账号是6217 8561 0000 1234 567，麻烦转这里", ["6217 8561 0000 1234 567"]), ("请汇到 6013820000123456789", ["6013820000123456789"]),
    ("我住在天津市和平区南京路189号津汇广场写字楼2108室", ["南京路189号"]), ("单位宿舍在盐城市亭湖区开放大道66号3栋", ["开放大道66号"]),
    ("我在青岛海信电器股份有限公司干了五年", ["海信电器股份有限公司"]), ("入职宁波慈溪市永昌塑料厂后一直没签合同", ["永昌塑料厂"]),
    ("身份证是320311198807153456，手机18852001234", ["320311198807153456", "18852001234"]),
    ("邮箱 liu.yang_88@outlook.com 也可以联系", ["liu.yang_88@outlook.com"]), ("我的车牌浙A8K123被扣在公司", ["浙A8K123"]),
    ("她叫上官婉儿，是我们车间主任", ["上官婉儿"]), ("同事宇文泰一直替他说话", ["宇文泰"]), ("我去了郑州富士康科技集团做普工", ["富士康科技集团"]),
    ("房东蒋海波说押金不退了", ["蒋海波"]), ("电话 0371-65432100 没人接", ["0371-65432100"]), ("微信 zhaoliu_2025 加我", ["zhaoliu_2025"]),
    ("包工头孟德胜带着钱跑了，我们十几个人的工资没发", ["孟德胜"]),
]
NEGATIVE = [
    "项目组长说月底发工资", "法定代表人没有出庭", "申请人向仲裁委提交了材料", "人事部通知我明天不用来了", "我的账号被公司冻结了，没法提现",
    "和平区的仲裁委受理了我的案子", "公司在南京路上，离我家很远", "宿舍条件很差，一间住八个人", "在电子厂干了五年，没签合同", "入职一年多，一直没有劳动合同",
    "我的手机号码被公司登记过，但我没提供", "合同上写的试用期是六个月", "车牌限行导致我迟到被扣钱", "总经理说要扣我三个月工资", "对方当事人没有到庭",
    "同事们都被拖欠了工资", "老板娘说下个月一起发", "领导让我写检讨", "我在医院住了十天，花了两万多", "仲裁委员会通知我下周开庭",
    "单位以经营困难为由不发工资", "我们车间一共三十多人", "厂里的食堂也不给吃了", "项目经理让我们先垫钱", "劳动合同第三条写的是月薪八千",
    "我是高中毕业的，没学过法律", "我是在2023年入职的", "组长高兴地答应了", "班长曾经帮我说过话", "邻居凌晨三点还在装修",
]


def run() -> tuple[int, int, list, list]:
    leaks, total = [], 0
    for text, secrets in POSITIVE:
        out = privacy.analyze(text).text
        for secret in secrets:
            total += 1
            if secret.replace(" ", "") in out.replace(" ", ""):
                leaks.append((text, secret, out))
    changed = [(t, privacy.analyze(t).text) for t in NEGATIVE if privacy.analyze(t).text != t]
    return total, len(NEGATIVE), leaks, changed


if __name__ == "__main__":
    total, negatives, leaks, changed = run()
    print(f"敏感值 {total} 个，漏检 {len(leaks)}（召回 {1 - len(leaks) / total:.0%}）；正常句 {negatives} 条，误伤 {len(changed)}")
    for text, secret, out in leaks:
        print(f"  漏: {secret} | {text} -> {out}")
    for text, out in changed:
        print(f"  误伤: {text} -> {out}")
