"""Stage 5 synthesis of frozen development results; no model or metric fitting."""
from pathlib import Path
import hashlib
import json

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
EXPECTED = Path(r"C:\Users\28908\Codex\fung-research\v2-diagnosis").resolve()
assert ROOT == EXPECTED and "onedrive" not in str(ROOT).lower()
for ancestor in (ROOT, *ROOT.parents):
    assert not ancestor.is_symlink() and not ancestor.is_junction(), ancestor

SOURCES = [ROOT / name / "metrics.json" for name in ("E0", "E1", "E2", "E3", "E4", "P1")]
before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES}
for source in SOURCES:
    json.loads(source.read_text(encoding="utf-8"))

OUTDIR = ROOT / "outputs"
assert not OUTDIR.is_symlink() and not OUTDIR.is_junction()
OUTDIR.mkdir(exist_ok=True)
assert OUTDIR.resolve().is_relative_to(ROOT)
OUT = OUTDIR / "stage5_group_meeting_summary.pdf"
assert not OUT.is_symlink()

pdfmetrics.registerFont(TTFont("YaHei", r"C:\Windows\Fonts\msyh.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("YaHeiBold", r"C:\Windows\Fonts\msyhbd.ttc", subfontIndex=0))
pdfmetrics.registerFontFamily("YaHei", normal="YaHei", bold="YaHeiBold")

body = ParagraphStyle("Body", fontName="YaHei", fontSize=11, leading=15.5,
                      textColor=colors.black, wordWrap="CJK", spaceAfter=5)
table_body = ParagraphStyle("Cell", parent=body, fontSize=10.7, leading=15, spaceAfter=0)
center = ParagraphStyle("Center", parent=table_body, alignment=TA_CENTER)
header = ParagraphStyle("Header", parent=table_body, fontName="YaHeiBold", alignment=TA_CENTER)
small = ParagraphStyle("Small", parent=body, fontSize=9.7, leading=13.2, spaceAfter=4)
title = ParagraphStyle("Title", parent=body, fontName="YaHeiBold", fontSize=17.5, leading=23,
                       spaceAfter=5)
heading = ParagraphStyle("Heading", parent=body, fontName="YaHeiBold", fontSize=11.2,
                         leading=15, spaceBefore=2, spaceAfter=3)


def p(text, style=body):
    return Paragraph(text, style)


rows = [
    ["诊断", "n / 组", "核心开发阶段证据  点估计 [95% CI]"],
    ["E0<br/>原规则分支", "100 / 40",
     "BA 61.8% [39.5, 77.6]，HGB 88.4% [70.4, 98.1]；"
     "配对差 -26.5 pp [-48.5, -5.2]。规则分数仅 0.488-0.510。"],
    ["E1<br/>规则信息", "3764 /<br/>1117",
     "R-info BA 71.9% [66.5, 77.2]；较 R-cov +19.7 pp [12.8, 26.4]。"
     "R-info-HGB 为 -9.1 pp [-14.3, -3.8]；"
     "R-all-HGB 为 -1.1 pp [-4.2, 2.3]。"],
    ["E2<br/>LLM 对照", "300 / 81",
     "A1-R-info 为 +1.6 pp [0.3, 3.3]。<b>A2 BA 98.4% [95.0, 100]</b>；"
     "较 HGB +12.8 pp [1.7, 30.1]，较 5NN +30.6 pp [12.7, 50.0]。"
     "无规则和邻居的 A3 BA 93.6% [84.5, 99.8]。"],
    ["E3 E4<br/>误差与解释", "300 / 81",
     "A2 原始错误 35 例 / 3 组，其中 33 例属同一组。平均引用 "
     "0.98 [0.72, 1.42] 条规则，84 例无引用；前 30 条规则中 29 条被暂标为代理。"
     "E3 选取 30 例 / 13 组，15 个 A1 样例中 13 个已弃权。"],
    ["P1<br/>近似验证器", "200 / 193",
     "CHGNet 对 DFT 的 MAE 0.058 [0.047, 0.072] eV/atom；"
     "均耗时 0.217 [0.123, 0.326] 秒/结构。TRAIN 1024 次尝试中 1020 个有效；"
     "完成 3 版 × 5 条规则，总用时约 26 分钟。"],
    ["P1<br/>规则精度", "2000 /<br/>1821",
     "留出集 MAE：v1 0.484 [0.465, 0.504] → v3 0.457 [0.439, 0.475]；"
     "RF 0.215 [0.202, 0.229] eV/atom。"
     "v3-RF 为 +0.241 [0.224, 0.259] eV/atom。"],
]

content = [
    p("超导分类 Agent 诊断与形成能可行性", title),
    p("阶段5 组会总结　　2026年10月17日　　全部为开发阶段结果", small),
    p("<b>主要发现</b>　A2 在固定开发集上优于 HGB，但规则的独立贡献尚未确定。"
      "结果提示应优先排查检索、聚合及强制赋分；P1 证明形成能近似验证循环能运行，当前规则仍落后于 RF。"),
]
cells = [[p(v, header if i == 0 else center if j < 2 else table_body)
          for j, v in enumerate(row)] for i, row in enumerate(rows)]
table = Table(cells, colWidths=[72, 65, 391], repeatRows=1, hAlign="LEFT")
table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EDF2")),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D9D9D9")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("LEFTPADDING", (0, 0), (-1, -1), 7),
    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
]))
content.extend([
    table, Spacer(1, 6),
    p("统计口径：BA 为平衡准确率；P/R 为精确率/召回率；pp 为百分点。性能均按原始组大小倒数加权。"
      "引用条数和耗时为材料均值，CI 均按组重采样。"
      "E0-E4 用 5000 次关联组 bootstrap，P1 用 2000 次组成组 bootstrap，差值均配对。"
      "A2 的 P 89.7% [77.6, 100]，R 97.5% [91.2, 100]（300 例 / 81 组）。", small),
    p("E2 均为 forced 结果：A1 选规则并取均值；A2 用规则和 5 个 TRAIN 邻居直接报概率；A3 仅材料信息。", small),
    p("路线建议", heading),
    p("<b>若继续 Tc 分类</b>，保留 A2 作为最佳 LLM 臂，优先辨清化学知识、邻居和规则各自的贡献。"
      "A1 增益小、A3 远高于随机，尚不支持“选规则 Agent 已奏效”或“LLM 推理或知识失败”。"
      "R-all 的结果说明规则库有可利用信息。"),
    p("<b>若转向形成能规则学习</b>，R-info 明显落后 HGB，已触发原方案第三条的转向建议；"
      "结合 R-all，仍需区分规则库与检索聚合的瓶颈。P1 支持开展“规则生成＋近似验证器”研究，预测优势尚未建立。"
      "缓存验证值后 100 轮约 2.0 小时；每轮换新结构约 8.4 小时，均为线性外推。路线选择由研究者确认。"),
    p("边界：以上为开发阶段证据，尚无独立测试结论；A2/A3 未保存配对差 CI。R-all 区间跨零不证明等效。"
      "E3 分层样本和 E4 物理性标签需人工复核，自报引用不证明忠实性。P1 使用 DFT 弛豫结构且存在预训练重叠，负形成能不等于凸包稳定；"
      "审计曾提前查看留出集验证器质量，未用于迭代。", small),
    p("依据：本研究 E0、E1、E2、E3、E4、P1 的冻结 metrics.json；本阶段仅汇总，未启动新实验。", small),
])

doc = SimpleDocTemplate(str(OUT), pagesize=letter, rightMargin=42, leftMargin=42,
                        topMargin=32, bottomMargin=30,
                        title="超导分类 Agent 诊断与形成能可行性", author="研究汇总")
doc.build(content)
reader = PdfReader(OUT)
assert len(reader.pages) == 1, f"Expected one page, got {len(reader.pages)}"
text = reader.pages[0].extract_text()
for expected in ("98.4", "0.241", "路线建议", "未用于迭代"):
    assert expected in text, expected
assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES}
print(json.dumps({"output": str(OUT), "pages": len(reader.pages), "bytes": OUT.stat().st_size,
                  "scientific_source_hashes_unchanged": True}, ensure_ascii=False))
