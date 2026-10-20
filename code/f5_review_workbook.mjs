/** Publish the frozen F4 material-review rows without changing any decision. */
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { createRequire } from "node:module";
import { execFileSync } from "node:child_process";
import assert from "node:assert/strict";
import crypto from "node:crypto";

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const WORK = path.join(ROOT, "work", "f5_publish");
const OUTPUT = path.join(ROOT, "F5_summary", "F4_material_review.xlsx");
const PYTHON = "C:/Users/28908/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe";
const require = createRequire(path.join(WORK, "package.json"));
const { Workbook, SpreadsheetFile } = await import(pathToFileURL(require.resolve("@oai/artifact-tool")).href);

async function assertDestination(target) {
  const parent = await fs.realpath(path.dirname(target));
  assert(!parent.toLowerCase().includes("onedrive"), "OneDrive destination rejected");
  const relative = path.relative(await fs.realpath(ROOT), parent);
  assert(!relative.startsWith("..") && !path.isAbsolute(relative), "Destination outside project");
}

for (const filename of [OUTPUT, path.join(WORK, "f4_review_input.json")]) await assertDestination(filename);
const inputText = execFileSync(PYTHON, ["-X", "utf8", "-B", "-c",
  "import json; from f5_f4_tables import extract_f4_review_tables; print(json.dumps(extract_f4_review_tables(), ensure_ascii=False))"],
  { cwd: ROOT, encoding: "utf8", maxBuffer: 10 * 1024 * 1024 });
const input = JSON.parse(inputText);
assert.equal(input.pending.length, 19);
assert.equal(input.excluded.length, 31);
assert.equal(new Set([...input.pending, ...input.excluded].map(row => row.material_id)).size, 50);
await fs.writeFile(path.join(WORK, "f4_review_input.json"), JSON.stringify(input, null, 2), "utf8");

const first = ["material_id", "formula_original", "reason_group_zh", "reason_zh", "Tc_K_recorded", "items_to_verify_zh"];
const columns = [...first, ...input.columns.filter(key => !first.includes(key))];
assert.equal(columns.length, 36);
assert.equal(new Set(columns).size, 36);
const widths = {
  material_id: 32, formula_original: 29, reason_group_zh: 32, reason_zh: 78,
  Tc_K_recorded: 14, items_to_verify_zh: 78, classification_zh: 24,
  eligibility_original: 42, reason_original_en: 125,
  Tc_note_zh: 60, Tc_details_original: 115,
  measurement_pressure_GPa_recorded: 20, ambient_measurement_verified: 20,
  pressure_details_original: 105, parent_family_original: 55, space_group_original: 25,
  paper_id: 32, paper_title_original: 100, paper_status_snapshot_original: 55,
  DOI_recorded: 65, arxiv_recorded: 40, primary_url: 90, evidence_urls_recorded: 115,
  material_report_date_recorded: 24, paper_earliest_date_recorded: 26,
  date_details_original: 115, first_report_verified: 24,
  exact_matching_accepted_ids: 65, parent_matching_id_formula_pairs: 110,
  dedup_status_original: 70, dedup_evidence_original: 145,
  source_json_locations: 145, material_original_json: 160,
  label_original_json: 130, paper_original_json: 160, parent_check_original_json: 155,
};
const dateKeys = new Set(["material_report_date_recorded", "paper_earliest_date_recorded"]);
const numberKeys = new Set(["Tc_K_recorded", "measurement_pressure_GPa_recorded"]);
const textUnits = text => [...String(text)].reduce((n, char) => n + (char.codePointAt(0) > 255 ? 1.9 : 1), 0);
const colName = index => {
  let out = "", number = index + 1;
  while (number) { number--; out = String.fromCharCode(65 + number % 26) + out; number = Math.floor(number / 26); }
  return out;
};
const asValue = (key, value) => {
  if (value == null || value === "") return null;
  if (dateKeys.has(key)) {
    assert.match(value, /^\d{4}-\d{2}-\d{2}$/);
    // Write Excel's sortable date serial explicitly; source ISO text also remains in raw JSON.
    return new Date(`${value}T00:00:00.000Z`).getTime() / 86400000 + 25569;
  }
  if (numberKeys.has(key)) assert.equal(typeof value, "number", `Non-numeric ${key}`);
  if (typeof value === "string") assert(!value.startsWith("="), "Unexpected leading formula marker in source text");
  return value;
};
const comparable = value => value instanceof Date ? value.getTime() / 86400000 + 25569 : value;
const wb = Workbook.create();
const checks = [];
const previews = [];
const specifications = [
  { key: "pending", name: "待定材料19", table: "F4PendingMaterials", colour: "#4F5967" },
  { key: "excluded", name: "排除材料31", table: "F4ExcludedMaterials", colour: "#253E50" },
];

for (const specification of specifications) {
  const rows = input[specification.key];
  const sheet = wb.worksheets.add(specification.name);
  sheet.showGridLines = false;
  sheet.tabColor = specification.colour;
  const headers = columns.map(key => input.column_titles[key]);
  const values = rows.map(row => columns.map(key => asValue(key, row[key])));
  const lastColumn = colName(columns.length - 1);
  const endRow = rows.length + 1;
  const fullRange = sheet.getRange(`A1:${lastColumn}${endRow}`);
  fullRange.values = [headers, ...values];
  const table = sheet.tables.add(`A1:${lastColumn}${endRow}`, true, specification.table);
  table.showFilterButton = true;
  table.showTotals = false;
  table.style = "TableStyleLight1";
  fullRange.format = {
    font: { name: "Microsoft YaHei", size: 10, color: "#17212B" },
    verticalAlignment: "top", horizontalAlignment: "left", wrapText: true,
  };
  const header = sheet.getRange(`A1:${lastColumn}1`);
  header.format = {
    fill: "#203C50", font: { name: "Microsoft YaHei", size: 10, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center", verticalAlignment: "center", wrapText: true, rowHeight: 42,
    borders: { insideVertical: { style: "thin", color: "#FFFFFF" } },
  };
  columns.forEach((key, index) => {
    const letter = colName(index);
    sheet.getRange(`${letter}1:${letter}${endRow}`).format.columnWidth = widths[key];
    if (numberKeys.has(key)) {
      sheet.getRange(`${letter}2:${letter}${endRow}`).setNumberFormat("General");
      sheet.getRange(`${letter}2:${letter}${endRow}`).format.horizontalAlignment = "right";
    } else if (dateKeys.has(key)) {
      sheet.getRange(`${letter}2:${letter}${endRow}`).setNumberFormat("yyyy-mm-dd");
      sheet.getRange(`${letter}2:${letter}${endRow}`).format.horizontalAlignment = "right";
    }
  });
  rows.forEach((row, rowIndex) => {
    const lines = Math.max(...columns.map(key => {
      const value = row[key];
      if (value == null) return 1;
      return String(value).split("\n").reduce((n, line) => n + Math.max(1, Math.ceil(textUnits(line) / (widths[key] * 0.89))), 0);
    }));
    const rowHeight = Math.max(54, Math.min(390, lines * 13.5 + 12));
    sheet.getRange(`A${rowIndex + 2}:${lastColumn}${rowIndex + 2}`).format.rowHeight = rowHeight;
    if (rowIndex % 2 === 1) sheet.getRange(`A${rowIndex + 2}:${lastColumn}${rowIndex + 2}`).format.fill = "#F1F5F7";
  });
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(1);
  const observed = sheet.getRange(`A2:${lastColumn}${endRow}`).values;
  values.forEach((row, r) => row.forEach((expected, c) => {
    const actual = observed[r][c];
    if (expected == null) assert(actual == null || actual === "", `${specification.name} ${r},${c} expected blank`);
    else assert.deepEqual(comparable(actual), comparable(expected), `${specification.name} ${r},${c}`);
  }));
  checks.push({ sheet: specification.name, records_verified: rows.length, columns_verified: columns.length, cells_verified: rows.length * columns.length });
}

wb.recalculate();
for (const specification of specifications) {
  const check = await wb.inspect({ kind: "region", sheetId: specification.name, range: "A1:F3",
    maxChars: 2000, tableMaxRows: 3, tableMaxCols: 6, tableMaxCellChars: 90 });
  console.log(check.ndjson);
  const preview = await wb.render({ sheetName: specification.name, range: "A1:F5", scale: 1.2, format: "png" });
  const filename = path.join(WORK, `F4_${specification.key}_preview.png`);
  await fs.writeFile(filename, new Uint8Array(await preview.arrayBuffer()));
  previews.push(filename);
  const originalReasonColumn = colName(columns.indexOf("reason_original_en"));
  const evidencePreview = await wb.render({ sheetName: specification.name,
    range: `${originalReasonColumn}1:${originalReasonColumn}3`, scale: 1.2, format: "png" });
  const evidenceFilename = path.join(WORK, `F4_${specification.key}_original_reason_preview.png`);
  await fs.writeFile(evidenceFilename, new Uint8Array(await evidencePreview.arrayBuffer()));
  previews.push(evidenceFilename);
}
const errors = await wb.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 20 }, maxChars: 2000, summary: "final error scan" });
console.log(errors.ndjson);
const file = await SpreadsheetFile.exportXlsx(wb);
await file.save(OUTPUT);
// The runtime emits this diagnostic beside exports; keep support files in WORK.
try { await fs.rename(`${OUTPUT}.inspect.ndjson`, path.join(WORK, "F4_material_review.inspect.ndjson")); }
catch (error) { if (error.code !== "ENOENT") throw error; }
const sha256 = crypto.createHash("sha256").update(await fs.readFile(OUTPUT)).digest("hex");
const verification = {
  api: "@oai/artifact-tool", output: OUTPUT, sha256, sheets: checks,
  source_sha256: input.source_sha256, columns, date_keys: [...dateKeys],
  numeric_keys: [...numberKeys], previews,
  data_decisions_unchanged: true, new_literature_searches: 0, model_evaluations: 0,
};
await fs.writeFile(path.join(WORK, "f4_workbook_verification.json"), JSON.stringify(verification, null, 2), "utf8");
const xmlVerificationScript = String.raw`
import datetime, hashlib, json, pathlib, posixpath, sys, zipfile
import xml.etree.ElementTree as ET
root = pathlib.Path(sys.argv[1])
work = root / 'work' / 'f5_publish'
source = json.loads((work / 'f4_review_input.json').read_text(encoding='utf8'))
details = json.loads((work / 'f4_workbook_verification.json').read_text(encoding='utf8'))
ns = {'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main', 'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
rn = {'p':'http://schemas.openxmlformats.org/package/2006/relationships'}
columns = details['columns']
def letter(number):
    result = ''
    number += 1
    while number:
        number, rem = divmod(number-1, 26)
        result = chr(65+rem)+result
    return result
def check_cell(cell, expected, key, shared, date_formats, xfs):
    if cell is None:
        assert expected is None or expected == '', (key, 'missing nonempty cell')
        return
    kind = cell.get('t')
    text = cell.findtext('m:v', default='', namespaces=ns)
    if kind == 's': value = shared[int(text)]
    elif kind == 'inlineStr': value = ''.join(cell.find('m:is', ns).itertext())
    elif kind == 'b': value = text == '1'
    elif kind in ('str','e'): value = text
    elif text: value = float(text)
    else: value = None
    assert cell.find('m:f', ns) is None, 'Unexpected formula'
    if expected is None or expected == '':
        assert value is None or value == '', (key, value, expected)
    elif key in details['date_keys']:
        serial = (datetime.date.fromisoformat(expected)-datetime.date(1899,12,30)).days
        assert kind not in ('s','inlineStr','str') and value == serial, (key, value, serial)
        fmt_id = int(xfs[int(cell.get('s','0'))].get('numFmtId', '0'))
        assert date_formats.get(fmt_id) == 'yyyy-mm-dd', (key, fmt_id)
    elif isinstance(expected, bool):
        assert kind == 'b' and value is expected, (key, kind, value, expected)
    elif isinstance(expected, (int,float)):
        assert kind not in ('s','inlineStr','str','b') and value == expected, (key, kind, value, expected)
    else:
        assert value == expected, (key, value, expected)
with zipfile.ZipFile(root/'F5_summary'/'F4_material_review.xlsx') as z:
    book = ET.fromstring(z.read('xl/workbook.xml'))
    relations = ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
    targets = {r.get('Id'):r.get('Target') for r in relations}
    shared = []
    if 'xl/sharedStrings.xml' in z.namelist():
        shared = [''.join(node.itertext()) for node in ET.fromstring(z.read('xl/sharedStrings.xml'))]
    styles = ET.fromstring(z.read('xl/styles.xml'))
    date_formats = {int(v.get('numFmtId')):v.get('formatCode') for v in styles.findall('m:numFmts/m:numFmt',ns)}
    xfs = styles.findall('m:cellXfs/m:xf',ns)
    names = [s.get('name') for s in book.findall('m:sheets/m:sheet', ns)]
    assert names == ['待定材料19','排除材料31'], names
    output = []
    for sheet, category in zip(book.findall('m:sheets/m:sheet', ns), ('pending','excluded')):
        target = targets[sheet.get('{'+ns['r']+'}id')]
        name = target.lstrip('/') if target.startswith('/') else posixpath.normpath(posixpath.join('xl',target))
        xml = ET.fromstring(z.read(name))
        cells = {cell.get('r'):cell for cell in xml.findall('m:sheetData/m:row/m:c',ns)}
        for c,key in enumerate(columns): check_cell(cells.get(letter(c)+'1'),source['column_titles'][key],'header_'+key,shared,date_formats,xfs)
        for r,row in enumerate(source[category],2):
            for c,key in enumerate(columns): check_cell(cells.get(letter(c)+str(r)),row[key],key,shared,date_formats,xfs)
        pane = xml.find('m:sheetViews/m:sheetView/m:pane',ns)
        assert pane is not None and pane.get('state') == 'frozen'
        assert pane.get('xSplit') == '1' and pane.get('ySplit') == '1', dict(pane.attrib)
        assert pane.get('topLeftCell') == 'B2', dict(pane.attrib)
        table_parts = xml.findall('m:tableParts/m:tablePart',ns)
        assert len(table_parts) == 1
        relname = posixpath.join(posixpath.dirname(name),'_rels',posixpath.basename(name)+'.rels')
        sheetrels = ET.fromstring(z.read(relname))
        table_target = next(v.get('Target') for v in sheetrels if v.get('Id') == table_parts[0].get('{'+ns['r']+'}id'))
        table_path = table_target.lstrip('/') if table_target.startswith('/') else posixpath.normpath(posixpath.join(posixpath.dirname(name),table_target))
        table_xml = ET.fromstring(z.read(table_path))
        expected_ref = 'A1:AJ'+str(len(source[category])+1)
        assert table_xml.get('ref') == expected_ref
        assert table_xml.find('m:autoFilter',ns).get('ref') == expected_ref
        assert len(table_xml.findall('m:tableColumns/m:tableColumn',ns)) == 36
        output.append({'sheet':sheet.get('name'),'rows_verified':len(source[category]),'cells_verified':len(source[category])*36,'filter_range':expected_ref,'frozen':'B2'})
for filename, sha in source['source_sha256'].items():
    assert hashlib.sha256(pathlib.Path(filename).read_bytes()).hexdigest() == sha, 'F4 source changed'
assert not any('comment' in name.lower() for name in z.namelist()), 'Unexpected comments'
result = {'exported_xml_verified':True,'rows':50,'cells':1800,'numeric_date_boolean_and_blank_types_verified':True,'source_hashes_unchanged':True,'sheets':output}
(work/'f4_workbook_xml_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(result,ensure_ascii=False))
`;
const xmlVerification = JSON.parse(execFileSync(PYTHON, ["-X", "utf8", "-B", "-c", xmlVerificationScript, ROOT],
  { cwd: ROOT, encoding: "utf8", maxBuffer: 1024 * 1024 }));
verification.exported_xml = xmlVerification;
await fs.writeFile(path.join(WORK, "f4_workbook_verification.json"), JSON.stringify(verification, null, 2), "utf8");
console.log(JSON.stringify(verification));
