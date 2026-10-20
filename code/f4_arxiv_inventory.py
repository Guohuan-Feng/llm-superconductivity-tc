"""Read-only, finite primary-source bibliography inventory for F4 collection."""
from __future__ import annotations

from collections import Counter
import json
import re
import urllib.request
import xml.etree.ElementTree as ET

URL = ('https://export.arxiv.org/api/query?search_query=cat%3Acond-mat.supr-con%20AND%20'
       'submittedDate%3A%5B202605300000%20TO%202610172359%5D&start=0&max_results=2000'
       '&sortBy=submittedDate&sortOrder=ascending')


def retrieve_inventory():
    """Return metadata only, never candidate acceptance or scientific labels."""
    request = urllib.request.Request(URL, headers={'User-Agent': 'F4 finite academic literature audit'})
    raw = urllib.request.urlopen(request, timeout=40).read()
    root = ET.fromstring(raw)
    ns = {'a': 'http://www.w3.org/2005/Atom', 'o': 'http://a9.com/-/spec/opensearch/1.1/',
          'x': 'http://arxiv.org/schemas/atom'}
    entries = []
    for record in root.findall('a:entry', ns):
        get = lambda tag: ' '.join(record.findtext(tag, '', ns).split())
        entries.append({'url': get('a:id'), 'title': get('a:title'), 'first_date': get('a:published'),
                        'doi': get('x:doi'), 'abstract': get('a:summary')})
    return {'source_url': URL, 'total_results': int(root.findtext('o:totalResults', '0', ns)),
            'returned_entries': len(entries), 'month_counts': dict(Counter(e['first_date'][:7] for e in entries)),
            'entries': entries, 'boundary': 'Category/date inventory is not proof of material novelty or ambient experimental Tc.'}


def discovery_candidates(inventory):
    """Transparent discovery heuristic only; no row is accepted by this filter."""
    selected = []
    for record in inventory['entries']:
        title, abstract = record['title'].lower(), record['abstract'].lower()
        material_experiment = bool(re.search(r'synthesi[sz]|synthesiz|new superconductor|new compound|'
                                            r'discover.{0,50}superconduct|superconduct.{0,40}discov', title + ' ' + abstract))
        experiment = bool(re.search(r'experiment|measurements|polycrystall|single.crystal|synthesi[sz]|synthesiz', abstract))
        theory_only = bool(re.search(r'we predict|first.principles|ab.initio|density.functional', abstract)) and not experiment
        if material_experiment and experiment and not theory_only:
            selected.append(record)
    return selected


if __name__ == '__main__':
    inventory = retrieve_inventory()
    selected = discovery_candidates(inventory)
    print(json.dumps({k: v for k, v in inventory.items() if k != 'entries'}, ensure_ascii=False))
    print(json.dumps({'discovery_candidate_titles': len(selected), 'candidates': [
        {k: value for k, value in row.items() if k != 'abstract'} for row in selected]}, ensure_ascii=False))
