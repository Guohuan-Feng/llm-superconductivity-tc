"""In-memory complete G5 classification, regression, agreement and paired tables."""
from f_report import _classification_section, _table, _cell

SUBSETS=(('all','全部'),('exclude_93','去掉component93'),('only_93','仅component93'))

def numeric(stat,signed=False):
    point=stat.get('point'); ci=stat.get('ci95',[None,None])
    if point is None:return 'NA'
    fmt='{:+.3f}' if signed else '{:.3f}'
    center=fmt.format(point)
    return center+' ['+', '.join('NA' if v is None else fmt.format(v) for v in ci)+']'

def build_report(m):
    lines=['# G5 Tc估计与多阈值开发结果','',
        '全部为原3764条材料的开发分析；A3-reg使用与原A3相同的化学式、成分和结构对象，输出Tc点估计及5/10/15K概率。',
        '分类真值严格Tc>阈值，概率p≥0.5；点估计严格Tc_est>阈值。两种LLM判定分别报告，不按实测标签修改输出。',
        'HGB分类器各阈值重训原三折；HGB回归目标log1p(Tc)，回归训练仅用TRAIN组权重，反变换max(0,expm1)。均固定原结构参数，无调参。',
        '组加权=1/完整原组大小；不加权每材料=1，两种CI都按组bootstrap5000次。所有配对使用相同组draw和共同有效分母；单组区间退化。',
        'Spearman：不加权为重复组展开后平均秩Pearson；组加权为weighted mid-CDF秩的加权Pearson。每个bootstrap内重新计算秩，未定义次数保留。',
        'G3十二类来自先前公式标注并冻结，包括“不是超导体”这一先验判断；仅用于分层，不能视作真标签或纯化学家族。','']
    for threshold,stats in m['classification'].items():
        lines+=['## 分类 '+threshold,'']
        partitions=[('主比较',stats['subsets'])]
        for key,title in [('families','F3家族'),('group_size_bins','原组大小'),('G3_families','G3十二类')]:
            partitions += [(title+'·'+label,subsets) for label,subsets in stats[key].items()]
        for title,subsets in partitions:
            for key,subtitle in SUBSETS:
                data=subsets[key]
                lines += [f"正类组数={data['n_positive_groups']}，负类组数={data['n_negative_groups']}（组可包含两类）。",'']
                _classification_section(lines,threshold+' '+title+' '+subtitle,subsets[key],stats['method_order'],[tuple(v) for v in stats['report_pairs']])
    lines+=['## Tc回归','', 'MAE单位K、差值单位K；Spearman为相关系数、差值为相关系数差。配对顺序为前者减后者，MAE越低越好。','']
    for kind,partitions in m['regression'].items():
        for label,subsets in partitions.items():
            for key,subtitle in SUBSETS:
                for population,stats in subsets[key].items():
                    lines += ['### '+kind+'·'+label+' '+subtitle+' '+population,'',
                              f"有效n={stats['n']}，组数={stats['n_groups']}；Tc>10正类组数={stats['n_positive_groups']}，Tc≤10负类组数={stats['n_negative_groups']}；请求n={stats['n_requested']}，缺失n={stats['n_missing']}。",'']
                    rows=[]
                    for method in stats['group_weighted']['metrics']:
                        row=[method]
                        for scheme in ('group_weighted','unweighted'):
                            row += [numeric(stats[scheme]['metrics'][method][metric]) for metric in ('MAE_K','Spearman')]
                        rows.append(row)
                    lines += [_table(['方法','组加权 MAE K','组加权 Spearman','不加权 MAE K','不加权 Spearman'],rows),'']
                    rows=[]
                    for pair in stats['group_weighted']['paired_differences']:
                        row=[pair]
                        for scheme in ('group_weighted','unweighted'):
                            row += [numeric(stats[scheme]['paired_differences'][pair][metric],True) for metric in ('MAE_K','Spearman')]
                        rows.append(row)
                    lines += [_table(['配对差','组加权 ΔMAE K','组加权 ΔSpearman','不加权 ΔMAE K','不加权 ΔSpearman'],rows),'']
                    undefined=[]
                    for scheme in ('group_weighted','unweighted'):
                        for method,vals in stats[scheme]['metrics'].items():
                            for metric,item in vals.items():
                                if item['n_undefined_boot']:undefined.append(f"{scheme}/{method}/{metric}:{item['n_undefined_boot']}/5000")
                    if undefined:lines += ['未定义bootstrap次数：'+'；'.join(undefined)+'。','']
    lines+=['## 与原A3在10K的一致性','']
    rows=[]
    for subset,title in SUBSETS:
        for method,stats in m['agreement_original_A3_10K'][subset].items():
            rows.append([title,method,stats['n'],stats['n_groups'],_cell(stats['group_weighted']),_cell(stats['unweighted'])])
    lines += [_table(['子集','A3-reg判定方式','n','组数','组加权一致率','不加权一致率'],rows),'']
    for family,subsets in m['agreement_original_A3_10K_G3_families'].items():
        rows=[]
        for subset,title in SUBSETS:
            for method,stats in subsets[subset].items():
                rows.append([title,method,stats['n'],stats['n_groups'],_cell(stats['group_weighted']),_cell(stats['unweighted'])])
        lines += ['### 一致性·G3十二类·'+family,'',_table(['子集','A3-reg判定方式','n','组数','组加权一致率','不加权一致率'],rows),'']
    return '\n'.join(lines)
