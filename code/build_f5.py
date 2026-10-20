"""Publish the finite F0--F3 one-page report and frozen F4 review exports."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle
from pypdf import PdfReader

from f5_evidence import extract_evidence
from f5_f4_tables import extract_f4_review_tables

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'F5_summary'
WEIGHTS = ('group_weighted', 'unweighted')
METRICS = ('balanced_accuracy', 'precision', 'recall', 'accuracy')
INK = colors.HexColor('#182D40')
BLUE = colors.HexColor('#245B7A')
PALE = colors.HexColor('#EAF1F5')
GRAY = colors.HexColor('#526373')


def safe(path):
    result = Path(path).resolve()
    if ROOT not in result.parents or any('onedrive' in s.lower() for s in result.parts):
        raise ValueError(f'Output outside approved local root: {result}')
    return result


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    safe(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def stat(value, *, ci=True, signed=False, digits=1):
    if value['point'] is None:
        return 'NA'
    fmt = ('+' if signed else '') + f'.{digits}f'
    p = format(100 * value['point'], fmt)
    lo, hi = value['ci95']
    if not ci:
        return p
    if lo is None or hi is None:
        return p + ' [NA]'
    return p + f' [{100 * lo:.{digits}f}, {100 * hi:.{digits}f}]'


def para(text, size=9.4, *, bold=False, color=INK, align=TA_LEFT, leading=None):
    return Paragraph(text, ParagraphStyle('body', fontName='CN-Bold' if bold else 'CN',
        fontSize=size, leading=leading or size * 1.34, textColor=color, alignment=align,
        wordWrap='CJK', spaceBefore=0, spaceAfter=0))


def text_at(c, text, x, top, width, size=9.4, **kwargs):
    p = para(text, size, **kwargs)
    _, h = p.wrap(width, 1000)
    if top - h < 19:
        raise ValueError(f'Text below page boundary: {text[:35]}')
    p.drawOn(c, x, top - h)
    return h


def table_at(c, rows, x, top, widths, *, size=8.5, row_height=21, header_rows=1, spans=()):
    data = [[para(escape(str(v)), size, bold=i < header_rows,
                  align=TA_LEFT if j < 2 else TA_CENTER)
             for j, v in enumerate(row)] for i, row in enumerate(rows)]
    heights = [23 if i < header_rows else row_height for i in range(len(rows))]
    t = Table(data, colWidths=widths, rowHeights=heights, repeatRows=header_rows)
    rules = [('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
             ('BACKGROUND', (0, 0), (-1, header_rows - 1), PALE),
             ('LINEBELOW', (0, header_rows - 1), (-1, header_rows - 1), .7, BLUE),
             ('LEFTPADDING', (0, 0), (-1, -1), 5),
             ('RIGHTPADDING', (0, 0), (-1, -1), 5),
             ('TOPPADDING', (0, 0), (-1, -1), 2),
             ('BOTTOMPADDING', (0, 0), (-1, -1), 2)]
    for i in range(header_rows, len(rows)):
        rules.append(('LINEBELOW', (0, i), (-1, i), .25, colors.HexColor('#D9E2E8')))
        if i % 2 == 0:
            rules.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor('#F7F9FA')))
    rules += [('SPAN', a, b) for a, b in spans]
    t.setStyle(TableStyle(rules))
    _, h = t.wrap(sum(widths), 1000)
    if top - h < 19:
        raise ValueError('Table below page boundary')
    t.drawOn(c, x, top - h)
    return h


def pdf_report(e, f4, path):
    pdfmetrics.registerFont(TTFont('CN', 'C:/Windows/Fonts/msyh.ttc', subfontIndex=0))
    pdfmetrics.registerFont(TTFont('CN-Bold', 'C:/Windows/Fonts/msyhbd.ttc', subfontIndex=0))
    w, h = landscape(A3)
    c = canvas.Canvas(str(safe(path)), pagesize=(w, h), pageCompression=1)
    c.setTitle('超导分类 F5 一页总结：F0-F3 开发证据')
    c.setAuthor('Codex')
    x, usable = 30, w - 60
    text_at(c, '超导 Tc &gt; 10 K：F0-F3 证据摘要', x, h-24, usable, 21, bold=True)
    text_at(c, '开发结果。F0/F1/F2：固定 300 例 / 81 组；F3：3764 例 / 1117 组。F3 每臂复用 E2 的 300 例。', x, h-56, usable, 10)
    text_at(c, '1　全量比较与去掉 component 93 的结果', x, h-78, usable, 12, bold=True)
    text_at(c, '单位 %；每格为点估计 [95% CI]。全部 n=3764 / 1117 组；去93 n=3650 / 1116 组。P=精确率，R=召回率，Acc=准确率。', x, h-98, usable, 9)
    rows = [['子集','方法','组加权','','','','不加权','','',''],
            ['','','BA','P','R','Acc','BA','P','R','Acc']]
    aliases = {'A2_forced':'A2','A3_forced':'A3','HGB127':'HGB127','StrongML':'E0 强ML','5NN':'5NN'}
    for subset, label in [('all','全部'), ('exclude_93','去93')]:
        d = e['F3']['subsets'][subset]
        for arm in aliases:
            rows.append([label, aliases[arm]] + [stat(d[wt]['metrics'][arm][m]) for wt in WEIGHTS for m in METRICS])
    table_at(c, rows, x, h-115, [43,73]+[(usable-116)/8]*8, size=8.35,
             row_height=20, header_rows=2, spans=(((2,0),(5,0)),((6,0),(9,0))))
    text_at(c, '全部与去93：A2/A3 对 HGB127、强ML、5NN 的配对 ΔBA（两种权重）95% CI 均高于0；完整配对数值见 metrics.json。',
            x,h-115-46-200-4,usable,8.7)
    y = h - 115 - 46 - 210 - 12
    text_at(c, '2　精确率的优势（配对 ΔP，百分点；组加权 / 不加权）', x, y, usable, 11.2, bold=True)
    for offset, (subset, label) in enumerate([('all','全部'),('exclude_93','去93')]):
        d=e['F3']['precision_focus'][subset]
        pairs=[]
        for a in ('A2_forced','A3_forced'):
            vals=[stat(d[wt][a+' - HGB127'],signed=True) for wt in WEIGHTS]
            pairs.append(aliases[a]+'−HGB：'+' / '.join(vals))
        text_at(c, label+'；'+'　　'.join(pairs), x, y-20-16*offset, usable, 9.35)

    top = y - 61
    left, right, gap = 640, usable-660, 20
    xr=x+left+gap
    text_at(c, '3　家族与原始组大小：对 HGB127 的配对 ΔBA', x, top, left, 11.5, bold=True)
    text_at(c, '百分点 [95% CI]；分层为探索性分析，未作多重比较校正。', x, top-20, left, 8.6)
    strata=[('families','copper','铜基'),('families','iron','铁基'),('families','A15','严格 A15'),
            ('families','hydride','氢化物'),('families','other','其他'),
            ('group_size_bins','1','组大小 1'),('group_size_bins','2-5','组大小 2-5'),
            ('group_size_bins','6-20','组大小 6-20'),('group_size_bins','>20','组大小 >20')]
    sr=[['分层 n/组','A2 组加权','A3 组加权','A2 不加权','A3 不加权']]
    for section,key,label in strata:
        d=e['F3'][section][key]['all']
        sr.append([f"{label} {d['n']}/{d['n_groups']}"]+[stat(d[wt]['paired_differences'][a+' - HGB127']['balanced_accuracy'],signed=True)
                  for wt in WEIGHTS for a in ('A2_forced','A3_forced')])
    ht=table_at(c,sr,x,top-37,[126]+[(left-126)/4]*4,size=8.25,row_height=18)
    after=top-37-ht-9
    text_at(c, '<b>组大小 1 和 2-5：</b>两臂对 HGB 的 ΔBA CI 均包含 0；该结论不能套用到强ML。2-5 档对强ML的两种权重 CI 均高于 0。', x, after, left, 9.2)
    text_at(c, '严格 A15=声明空间群223且二元A3B：33例/30组、4正类，与93仅重叠2例。氢化物仅4例。仅93有114例/1组，CI退化。', x, after-31, left, 8.6)

    text_at(c, '4　F2 隐去身份后下降（同一固定300例）', xr, top, right, 11.5, bold=True)
    text_at(c, '表为 BA 点估计%；去93 n=186 / 80组。M1：匿名粗分档性质+结构；M2：匿名标准化 f1-f127。', xr, top-20, right, 8.6)
    fr=[['表示/方法','全部组','全部材料','去93组','去93材料']]
    for arm,label in [('A3_E2','A3 原输入'),('A3-M1','A3-M1'),('A3-M2','A3-M2'),
                      ('A2-M2','A2-M2'),('HGB127','HGB127'),('5NN','5NN')]:
        fr.append([label]+[stat(e['F2']['subsets'][s][wt]['metrics'][arm]['balanced_accuracy'],ci=False)
                          for s in ('all','exclude_93') for wt in WEIGHTS])
    fh=table_at(c,fr,xr,top-48,[94]+[(right-94)/4]*4,size=9,row_height=18)
    fy=top-48-fh-9
    drops=e['F2']['identity_BA_drops']['all']
    for offset,(pair,label) in enumerate([('A3_E2 - A3-M1','A3→M1'),('A3-M1 - A3-M2','M1→M2')]):
        text_at(c, label+' 的 BA 下降（组 / 材料）：'+ ' / '.join(stat(drops[wt][pair],signed=True) for wt in WEIGHTS)+' pp',xr,fy-16*offset,right,8.9)
    text_at(c, 'A3-M2：300/300弃权且p=0.5。A2-M2未检出超过HGB/5NN。遮蔽同时改变表示/量化，下降不是记忆比例；HGB与5邻居提示的监督量不同。',xr,fy-35,right,8.7)
    five_y=fy-73
    text_at(c,'5　A2≈A3：未检出规则和近邻的额外总体收益',xr,five_y,right,11,bold=True)
    same=e['F3']['subsets']['all']
    vals=[stat(same[wt]['paired_differences']['A2_forced - A3_forced']['balanced_accuracy'],signed=True) for wt in WEIGHTS]
    text_at(c,'全量 ΔBA（组 / 材料）：'+' / '.join(vals)+' pp。CI均含0；这不证明贡献严格为0，也不是等效性检验。',xr,five_y-20,right,9)

    # Context and scope sit below both columns and remain part of the one-page handout.
    context_y=88
    text_at(c,'F0复算确认固定300例原结果；F1自报认识/部分认识297/300（明确认识35、部分262），对照2/20部分认识。自报认识不等于核实训练曝光，无法据此分解记忆与推理。',x,context_y,usable,9.3)
    text_at(c,'F4：新材料不足，未评测。核查45篇重点文献、55个候选配比；仅5例/3家族合格，全部Tc≤10 K；19待定、31排除。低于40例门槛且缺少正类，研究现按用户要求暂停。',x,context_y-22,usable,9.3,bold=True)
    text_at(c,'统计：两种权重均按component组重采样5000次；组权重=1/原始组大小；强制判断p≥0.5（含模型标记弃权）。完整三子集、双权重、四指标CI及配对值见同目录metrics.json。',x,context_y-44,usable,8.3,color=GRAY)
    text_at(c,'来源：F0_reanalysis、F1_recall、F2_mask、F3_full3764 的冻结 metrics.json；F4仅引用冻结收集状态。不新增调用、拟合或bootstrap。',x,context_y-58,usable,8.1,color=GRAY)
    c.showPage();c.save()
    pdf=PdfReader(path)
    if len(pdf.pages)!=1:
        raise ValueError('F5 must be exactly one page')
    content=pdf.pages[0].extract_text()
    if '\x00' in content or '\ufffd' in content:
        raise ValueError('PDF contains an unsupported or replacement glyph')
    for required in ('精确率','组大小','F2','A2','A3','未评测','未检出'):
        if required not in content:
            raise ValueError(f'PDF missing required text: {required}')
    return {'pages':1,'page_size_points':[w,h],'format':'A3 landscape','text_characters':len(content)}


def main():
    safe(OUT).mkdir(exist_ok=True)
    e=extract_evidence(ROOT)
    f4=extract_f4_review_tables(ROOT)
    assert len(f4['pending'])==19 and len(f4['excluded'])==31
    for key,name in [('pending','F4_pending.csv'),('excluded','F4_excluded.csv')]:
        with safe(OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f)
            writer.writerow([f4['column_titles'][col] for col in f4['columns']])
            for row in f4[key]:
                writer.writerow([row[col] for col in f4['columns']])
    checks=pdf_report(e,f4,OUT/'F5_summary.pdf')
    stamp=datetime.now(timezone.utc).isoformat()
    e['F4_status_only']={'status':'paused_by_user','statement':'新材料不足，未评测',
        'candidate_papers':45,'candidate_materials':55,'accepted_n':5,'accepted_groups':3,
        'accepted_positive_n':0,'accepted_negative_n':5,'pending_n':19,'excluded_n':31,
        'hard_minimum_n':40,'new_LLM_calls':0,'new_fits':0,'performance':None}
    e['F4_review_tables']=f4
    e['publication_checks']=checks
    e['status']='complete_summary_draft_pending_independent_review'
    write_json(OUT/'metrics.json',e)
    config={'stage':'F5','created_utc':stamp,'scope':'One-page F0-F3 development evidence only; F4 collection status and frozen backup tables',
        'F4_research_status':'paused_by_user','new_LLM_calls':0,'new_fits':0,'new_bootstrap_resamples':0,
        'source_metrics':e['sources'],'F4_sources':f4['source_sha256'],
        'output_format':'one-page A3 landscape PDF; one XLSX with two sheets; two UTF-8 BOM CSV tables',
        'statistical_source':'All estimates and group-bootstrap intervals copied from completed source metrics; reversed paired differences negate endpoints only',
        'interpretation':'No detected overall A2-A3 benefit; not proof of zero causal contribution or statistical equivalence',
        'output_directory_role':'Summary and requested backup deliverables; not a new evaluated experiment',
        'maximum_directory_files':50}
    write_json(OUT/'config.json',config)
    readme=[
        '# F5 一页总结与 F4 备用原因表',
        '仅汇总 F0-F3 开发结果；F4研究按用户要求暂停。',
        'F5_summary.pdf：一页横版，包含全量/去93双权重四指标及95%CI。',
        '同页列出精确率配对差、家族/组大小BA差、F2遮蔽下降与A2≈A3边界。',
        '完整三子集、双权重指标与配对CI见 metrics.json。',
        'F4_material_review.xlsx：待定19条、排除31条，两张可筛选表。',
        'F4_pending.csv 与 F4_excluded.csv：同源完整记录，UTF-8 BOM。',
        '原因表保留中文说明、原文原因、来源、日期歧义和原始JSON定位。',
        '未补填缺失Tc、未重判资格、未新增文献或评测。',
        'F4合格仅5例/3家族、全部负类，低于40门槛；新材料不足，未评测。',
        'F0-F4源目录保持只读；不引用作废的stage5摘要。',
        '无新LLM调用、模型拟合、特征调参或bootstrap。',
        '组1/2-5不显著特指对HGB；不笼统套用到强ML。',
        'A2≈A3表示未检出额外总体收益，不证明贡献为零或等效。',
        '本目录为报告交付而非新实验；完成后停止，不自动选择研究路线。']
    safe(OUT/'README.md').write_text('\n'.join(readme)+'\n',encoding='utf-8')
    with safe(OUT/'run.log').open('a',encoding='utf-8') as f:
        f.write(stamp+' F5 published PDF pages=1;F4 pending=19 excluded=31;no new calls/fits/bootstrap;await final visual/independent audit\n')
    print(json.dumps({'pdf':str(OUT/'F5_summary.pdf'),'checks':checks,'CSV_rows':[19,31],'files':len(list(OUT.iterdir()))},ensure_ascii=False))


if __name__=='__main__':
    main()
