"""Measure how well backend/privacy.py redacts, using synthetic sentences with known answers.

    python -I eval/privacy_stress.py [-v]

Every positive sentence is built from a template plus an injected value, so whether the value
survives redaction is known. Negatives are ordinary legal sentences that must come out unchanged.
The values are invented; this measures the rules, not real-world distribution. -v lists the misses.
"""
from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend import privacy  # noqa: E402

random.seed(20261030)
NAMES_COMMON = ["王小明", "张伟", "李娜", "刘芳", "陈思远", "赵四", "郝建国", "周杰", "吴敏", "孙丽", "马国强", "朱红"]
NAMES_RARE = ["欧阳俊杰", "司马青", "诸葛明", "滕云", "蔺晓", "柴进", "瞿晨", "邢磊", "霍然", "窦娥", "慕容复", "尉迟恭"]
COMPANIES_FORMAL = ["深圳华为技术有限公司", "北京字节跳动科技有限公司", "苏州海天机械有限公司", "杭州万象餐饮管理有限公司", "上海申通物流集团",
                    "成都蜀香火锅饭店", "南京鼓楼区永安物业公司", "广州白云区恒丰制衣厂", "(北京)京东世纪贸易有限公司"]
COMPANIES_BARE = ["海底捞", "美团", "华为", "顺丰", "京东", "拼多多"]
ADDRESSES = ["北京市朝阳区建国路88号", "海淀区中关村大街1号院3号楼502室", "深圳市南山区科技园路12号", "幸福小区3号楼2单元501", "杨柳村12号",
             "上海市浦东新区张江路100弄5号", "东城区景山东街15号", "广州市天河区体育西路103号维多利广场B座2305", "成都市武侯区人民南路四段27号",
             "朝阳北路109号金色家园7栋1203", "西安市雁塔区小寨东路45号"]
PHONES = ["13812345678", "138-1234-5678", "138 1234 5678", "+86 13912345678", "15987654321", "18600001111"]
LANDLINES = ["010-12345678", "0755-88889999", "021-6543210", "02566778899"]
IDS = ["110101199003071234", "32010219851212123X", "440305200001018765", "510104199512301236"]
CARDS = ["6222020200000000003", "6217 0000 1001 2345 678", "4003 6100 0000 0001", "6228480402564890018"]
EMAILS = ["zhang.wei@qq.com", "lina_2024@163.com", "hr-office@company.cn"]
PLATES = ["苏A12345", "京B8Z999", "粤B12345", "沪C7K321"]
SOCIAL = ["微信号 wx_zhang2024", "QQ 123456789", "vx：abc_12345"]

NAME_FRAMES = ["我叫{}，", "老板{}一直不给我发工资", "经理{}说要把我开除", "同事{}也被拖欠了", "{}经理让我签空白合同", "我的名字是{}", "人事{}通知我明天不用来",
               "房东{}不退押金", "对方{}拒绝协商", "{}师傅带我上岗", "联系人{}电话没人接", "主管{}威胁我"]
COMPANY_FRAMES = ["我在{}上班", "{}拖欠了我两个月工资", "我被{}辞退了", "我入职{}一年了", "{}不给我签劳动合同", "我应聘到{}做销售", "在{}做服务员被开除", "我去了{}之后一直没交社保"]
ADDRESS_FRAMES = ["我住在{}", "公司地址是{}", "我家在{}，快递寄这里", "工作地点：{}", "我现在住{}，离公司很远"]
PLAIN_FRAMES = ["我的{}是{}", "{}，请记一下", "对方给我发了{}", "我留的联系方式是{}"]
NEGATIVES = [
    "公司没跟我签劳动合同，已经半年了", "依据劳动合同法第八十二条可以主张二倍工资", "老板马上就走了，工资还没发", "我们公司拖欠工资三个月",
    "他们公司不给交社保，这家公司太过分", "我被公司辞退了，公司说我违纪", "我月薪8000，2024年3月入职，2025年1月被辞退", "案号（2023）苏03民终1234号可以参考",
    "我想咨询12333怎么投诉", "高温补贴一直没发，方便面厂也没有", "金额大概有两万多块", "关于工伤认定需要什么材料", "我在厂里干活摔伤了，工厂不管",
    "安全生产许可证过期了，管理很混乱", "起诉有限公司需要准备什么", "工资条上显示有13个月，扣了社保", "银行转账记录显示每月5日发放", "合同编号20240305001，签了两份",
    "我想知道经济补偿怎么算，工作了五年三个月", "仲裁时效一年，我离职半年了", "领导让我周末加班，也没有加班费", "劳务派遣公司把我派到别的单位",
"高兴的是仲裁委受理了", "于是我就去投诉了", "万一公司不认账怎么办", "马上要过年了，工资还没结", "白天上班，晚上还要加班到十点",
    "程序上先仲裁还是先起诉", "关系到我的社保缴纳年限", "金融行业的试用期可以约定多久", "石头厂的工人没有劳动合同", "周末加班按三倍工资计算对吗",
    "对方律师说可以调解", "双方老板都同意了", "老板曾经答应过补发", "同事关系一直不错", "领导冷静处理了此事", "老板简单说了几句",
    "经理满意地走了", "对方经理拒绝出面", "我们工厂老板不管", "有什么法律能制裁公司吗", "我想问问这家公司有没有资质", "在公司上班三年了",
    "被公司开除后怎么办", "大型连锁超市也要签合同", "领导莫名其妙地扣了我的奖金", "老板符合条件才能辞退吗",
    "我在北京工作，老板在上海", "在深圳打工，月薪一万二", "我们是苏州的一家工厂", "公司在杭州，我在成都远程办公",
]


def run() -> dict:
    cases: list[tuple[str, str, str]] = []  # (category, text, secret)
    for name in NAMES_COMMON:
        for frame in random.sample(NAME_FRAMES, 4):
            cases.append(("姓名(常见姓)", frame.format(name), name))
    for name in NAMES_RARE:
        for frame in random.sample(NAME_FRAMES, 4):
            cases.append(("姓名(复姓/少见姓)", frame.format(name), name))
    for company in COMPANIES_FORMAL:
        for frame in random.sample(COMPANY_FRAMES, 4):
            cases.append(("单位(全称)", frame.format(company), company.replace("(北京)", "")))
    for company in COMPANIES_BARE:
        for frame in random.sample(COMPANY_FRAMES, 2):
            cases.append(("单位(只有简称)", frame.format(company), company))
    for address in ADDRESSES:
        for frame in random.sample(ADDRESS_FRAMES, 3):
            cases.append(("地址", frame.format(address), address))
    groups = [("手机号", PHONES, "手机"), ("座机号", LANDLINES, "座机"), ("身份证号", IDS, "身份证"), ("银行卡号", CARDS, "银行卡号"),
              ("邮箱", EMAILS, "邮箱"), ("车牌号", PLATES, "车牌"), ("社交账号", SOCIAL, "联系方式")]
    for label, values, word in groups:
        for value in values:
            for frame in random.sample(PLAIN_FRAMES, 2):
                text = frame.format(word, value) if frame.count("{}") == 2 else frame.format(value)
                secret = value.split()[-1] if label == "社交账号" else value
                cases.append((label, text, secret))

    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    misses: list[tuple[str, str, str]] = []
    for category, text, secret in cases:
        out = privacy.analyze(text).text
        compact = secret.replace(" ", "").replace("-", "")
        leaked = secret in out or compact in out.replace(" ", "").replace("-", "")
        stats[category][1] += 1
        if leaked:
            misses.append((category, text, out))
        else:
            stats[category][0] += 1

    false_positive = [(t, privacy.analyze(t).text) for t in NEGATIVES if privacy.analyze(t).text != t]
    return {"stats": stats, "misses": misses, "false_positive": false_positive, "negatives": len(NEGATIVES)}


def main() -> None:
    result = run()
    print(f"{'类别':<16}{'已脱敏':>8}{'总数':>6}{'召回率':>8}")
    total_ok = total = 0
    for category, (ok, n) in result["stats"].items():
        print(f"{category:<16}{ok:>8}{n:>6}{ok / n:>8.0%}")
        total_ok += ok
        total += n
    print(f"{'合计':<16}{total_ok:>8}{total:>6}{total_ok / total:>8.0%}")
    fp = result["false_positive"]
    print(f"\n正常句子 {result['negatives']} 条，被误改 {len(fp)} 条（误伤率 {len(fp) / result['negatives']:.0%}）")
    for text, out in fp:
        print(f"  误伤: {text}\n   ->  {out}")
    if "-v" in sys.argv:
        print("\n漏检明细：")
        for category, text, out in result["misses"]:
            print(f"  [{category}] {text}\n      {out}")


if __name__ == "__main__":
    main()
