from __future__ import annotations

import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports"
DOCS_DIR = REPO_ROOT / "docs"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)


providers = [
    {
        "name": "Sportradar",
        "coverage": "Broad international football coverage, strong enterprise distribution; historical odds archive availability needs explicit contract confirmation.",
        "historical_depth": "Likely multi-season; contract-specific",
        "corner_market_quality": "High potential, but unverified in this environment",
        "timestamp_quality": "Likely strong with feed-grade timestamps",
        "opening_closing": "Likely available via enterprise archive",
        "api_quality": "High",
        "documentation": "Good",
        "commercial_risk": "Medium",
        "integration": "Medium",
        "cost": "$75k-$200k/yr",
        "score": 84,
        "verdict": "Strong shortlist candidate",
    },
    {
        "name": "Stats Perform",
        "coverage": "Strong football data and betting-related products; historical odds archive needs direct confirmation.",
        "historical_depth": "Likely multi-season; contract-specific",
        "corner_market_quality": "Medium-high potential",
        "timestamp_quality": "Likely strong",
        "opening_closing": "Likely available subject to licence",
        "api_quality": "High",
        "documentation": "Good",
        "commercial_risk": "High",
        "integration": "Medium",
        "cost": "$60k-$180k/yr",
        "score": 80,
        "verdict": "Strong shortlist candidate",
    },
    {
        "name": "LSports",
        "coverage": "Strong sportsbook feed and odds ecosystem; historical archive access needs explicit commercial due diligence.",
        "historical_depth": "Likely multi-season; unclear procurement path",
        "corner_market_quality": "Medium-high potential",
        "timestamp_quality": "Likely strong",
        "opening_closing": "Not yet verified",
        "api_quality": "Medium-high",
        "documentation": "Medium",
        "commercial_risk": "High",
        "integration": "Medium",
        "cost": "$40k-$150k/yr",
        "score": 74,
        "verdict": "Needs sales validation",
    },
    {
        "name": "FeedConstruct",
        "coverage": "Enterprise feed player with odds products; historical corner archive access must be confirmed in the contract.",
        "historical_depth": "Likely multi-season; contract-specific",
        "corner_market_quality": "Medium potential",
        "timestamp_quality": "Likely strong",
        "opening_closing": "Not yet verified",
        "api_quality": "Medium-high",
        "documentation": "Medium",
        "commercial_risk": "High",
        "integration": "Medium",
        "cost": "$35k-$120k/yr",
        "score": 70,
        "verdict": "Needs sales validation",
    },
    {
        "name": "Betfair Historical Data",
        "coverage": "Excellent historical exchange-market depth; licensing and export path are the gating issues.",
        "historical_depth": "Very strong if licensed",
        "corner_market_quality": "High potential for exchange-based lines",
        "timestamp_quality": "Excellent",
        "opening_closing": "Likely available through historical market snapshots",
        "api_quality": "High",
        "documentation": "Good",
        "commercial_risk": "Medium",
        "integration": "High",
        "cost": "$25k-$100k/yr",
        "score": 68,
        "verdict": "Interesting but operationally heavy",
    },
    {
        "name": "API-Football",
        "coverage": "Live fixture and league coverage verified; historical odds archive not verified in this environment.",
        "historical_depth": "No verified historical odds archive",
        "corner_market_quality": "Not verified",
        "timestamp_quality": "Not verified",
        "opening_closing": "Not verified",
        "api_quality": "Medium",
        "documentation": "Good",
        "commercial_risk": "Medium",
        "integration": "Low",
        "cost": "$5k-$20k/yr",
        "score": 38,
        "verdict": "Useful for live fixtures only; not sufficient for scientific validation",
    },
    {
        "name": "The Odds API",
        "coverage": "Live odds coverage only; no verified historical corner-odds archive found in this environment.",
        "historical_depth": "Not verified",
        "corner_market_quality": "Not verified",
        "timestamp_quality": "Not verified",
        "opening_closing": "Not verified",
        "api_quality": "Medium",
        "documentation": "Good",
        "commercial_risk": "Medium",
        "integration": "Low",
        "cost": "$5k-$20k/yr",
        "score": 30,
        "verdict": "Not suitable for historical backtesting",
    },
    {
        "name": "Football-Data.co.uk",
        "coverage": "Useful for results and fixtures, not a true historical odds archive for Total Corners.",
        "historical_depth": "Very strong for results",
        "corner_market_quality": "Low for odds",
        "timestamp_quality": "Low for odds",
        "opening_closing": "Not available",
        "api_quality": "Medium",
        "documentation": "Good",
        "commercial_risk": "Low",
        "integration": "Low",
        "cost": "$0-$5k/yr",
        "score": 22,
        "verdict": "Complementary only; not sufficient for model validation",
    },
]


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_xlsx(path: Path, rows: list[dict[str, object]]) -> None:
    headers = [
        "provider",
        "coverage",
        "historical_depth",
        "corner_market_quality",
        "timestamp_quality",
        "opening_closing",
        "api_quality",
        "documentation",
        "commercial_risk",
        "integration",
        "cost",
        "score",
        "verdict",
    ]
    sheet_rows = [headers] + [[str(row.get(h, "")) for h in headers] for row in rows]
    shared_strings = []
    cells = []

    def add_cell(value: str) -> str:
        if value not in shared_strings:
            shared_strings.append(value)
        return f'<c t="s"><v>{shared_strings.index(value)}</v></c>'

    for row_index, row in enumerate(sheet_rows, start=1):
        cell_xml = []
        for value in row:
            cell_xml.append(add_cell(value))
        cells.append(f'<row r="{row_index}">{"".join(cell_xml)}</row>')

    sheet_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheetData>{"".join(cells)}</sheetData>
</worksheet>'''

    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
</Types>'''

    rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>'''

    workbook = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="Provider Scorecard" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>'''

    workbook_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>'''

    styles = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>
  <fills count="1"><fill><patternFill patternType="none"/></fill></fills>
  <borders count="1"><border/></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>'''

    app = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>CornerLab</Application>
</Properties>'''

    core = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Provider Scorecard</dc:title>
  <dc:creator>CornerLab</dc:creator>
</cp:coreProperties>'''

    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('[Content_Types].xml', content_types)
        zf.writestr('_rels/.rels', rels)
        zf.writestr('xl/workbook.xml', workbook)
        zf.writestr('xl/_rels/workbook.xml.rels', workbook_rels)
        zf.writestr('xl/worksheets/sheet1.xml', sheet_xml)
        zf.writestr('xl/styles.xml', styles)
        zf.writestr('docProps/app.xml', app)
        zf.writestr('docProps/core.xml', core)


lines = []
lines += ["# Provider RFQ", "", "## Objective", "", "Acquire a historical odds archive capable of supporting a statistically valid Total Corners betting model without violating licensing or research-use constraints.", "", "## Minimum commercial asks", "", "- Verified historical Serie A Total Corners odds for at least three completed seasons.", "- Opening and closing prices for target lines 8.5, 9.5, 10.5 and 11.5.", "- Bookmaker identity, fixture mapping keys, timestamps, and settlement information.", "- Internal research, caching, and backtesting rights in writing.", "- Sample export and pilot pricing with no hidden redistribution restrictions.", "", "## Current evidence", "", "- No provider in the current environment produced a verified historical corner-odds sample with opening and closing values.", "- API-Football verified fixture access but not a usable historical odds payload.", "- TheStatsAPI was blocked by a subscription error before historical seasons could be resolved.", ""]
write_text(REPORTS_DIR / "provider_rfq.md", "\n".join(lines) + "\n")

shortlist_lines = ["# Provider shortlist", "", "The current evidence supports a cautious shortlist of five vendors only if they can provide a signed sample export and a licence that allows internal research and backtesting.", "", "1. Sportradar", "2. Stats Perform", "3. LSports", "4. FeedConstruct", "5. Betfair Historical Data", "", "## Why these five", "", "- They are the most credible enterprise-grade options in the current review set.", "- They are the most plausible routes to multi-season historical odds with bookmaker and timestamp granularity.", "- All require formal sales validation before any procurement decision.", ""]
write_text(REPORTS_DIR / "provider_shortlist.md", "\n".join(shortlist_lines) + "\n")

cost_lines = ["# Provider cost ROI", "", "## Estimated annual cost ranges", "", "| Provider | Estimated annual cost | ROI view |", "| --- | ---: | --- |", "| Sportradar | $75k-$200k | Attractive only if historical corner samples are strong and licence terms are favourable |", "| Stats Perform | $60k-$180k | Attractive only if sample quality is proven |", "| LSports | $40k-$150k | Moderate; depends on archive depth and fee structure |", "| FeedConstruct | $35k-$120k | Moderate; depends on corner-specific archive availability |", "| Betfair Historical Data | $25k-$100k | Potentially attractive if exchange-based closing prices are acceptable |", "", "## Recommendation", "", "Do not spend budget until a signed sample export demonstrates opening and closing Total Corners prices, bookmaker identity, timestamps and research rights. Without that evidence, ROI cannot be credibly estimated.", ""]
write_text(REPORTS_DIR / "provider_cost_roi.md", "\n".join(cost_lines) + "\n")

license_lines = ["# Provider license matrix", "", "| Provider | Licence risk | Internal research | Commercial use | Notes |", "| --- | --- | --- | --- | --- |", "| Sportradar | Medium | Needs clause review | Needs clause review | Enterprise licence required; historical reuse rights must be explicit |", "| Stats Perform | High | Needs clause review | Needs clause review | Must confirm historical odds redistribution and backtesting rights |", "| LSports | High | Needs clause review | Needs clause review | Archive and derivative-data clauses need review |", "| FeedConstruct | High | Needs clause review | Needs clause review | Needs explicit commercial and research-use language |", "| Betfair Historical Data | Medium | Needs clause review | Needs clause review | Exchange-based historical data may have stricter usage terms |", "| API-Football | Medium | Limited | Limited | Not suitable for historical backtesting without new commercial terms |", ""]
write_text(REPORTS_DIR / "provider_license_matrix.md", "\n".join(license_lines) + "\n")

contacts_lines = ["# Provider contacts", "", "| Provider | Likely contact channel | Status |", "| --- | --- | --- |", "| Sportradar | Sales / enterprise contact form | Not verified in this environment |", "| Stats Perform | Sales / enterprise contact form | Not verified in this environment |", "| LSports | Sales / enterprise contact form | Not verified in this environment |", "| FeedConstruct | Sales / enterprise contact form | Not verified in this environment |", "| Betfair Historical Data | Developer / account team | Not verified in this environment |", "", "Recommended next step: request a vendor meeting and ask for a sample export before any procurement or implementation commitment.", ""]
write_text(REPORTS_DIR / "provider_contacts.md", "\n".join(contacts_lines) + "\n")

negotiation_lines = ["# Provider negotiation strategy", "", "1. Ask for a signed sample export covering at least three completed seasons of Serie A Total Corners odds.", "2. Require opening and closing prices, bookmaker identity, fixture mapping keys, timestamps and settlement fields.", "3. Make research rights explicit: internal research, model validation, backtesting, caching and derivative reporting.", "4. Ask for a pilot pricing structure and a short-term trial before full annual commitment.", "5. Keep a fallback plan: use public result data and manual odds archives only for initial validation, not production-grade model training.", ""]
write_text(REPORTS_DIR / "provider_negotiation_strategy.md", "\n".join(negotiation_lines) + "\n")

sample_lines = ["# Provider sample checklist", "", "- Provider name", "- League and season coverage", "- Market coverage for Total Corners Over/Under", "- Opening odds, closing odds, and last pre-kickoff odds", "- Bookmaker names and IDs", "- Fixture mapping to actual matches", "- Timestamps in UTC", "- Vendor licence terms for internal research", "- Files format: CSV, parquet, or API export", "- Sample size and data quality notes", ""]
write_text(REPORTS_DIR / "provider_sample_checklist.md", "\n".join(sample_lines) + "\n")

questions_lines = ["# Provider questions", "", "1. Can you provide three completed seasons of historical Serie A Total Corners odds?", "2. Are opening and closing odds available for 8.5, 9.5, 10.5 and 11.5 markets?", "3. Can you provide bookmaker identity, timestamps and fixture mapping keys?", "4. Are internal research, backtesting, caching and derivative reporting permitted?", "5. What is the pilot price, annual price and export format?", ""]
write_text(REPORTS_DIR / "provider_questions.md", "\n".join(questions_lines) + "\n")

manifesto = """# CornerLab Manifesto

## Mission
CornerLab exists to determine whether a statistically valid Total Corners betting model can be built from high-quality historical odds data without compromising scientific discipline.

## Scientific Principles
- Prefer externally verifiable evidence over optimism.
- Require a real historical odds sample before any model or strategy claim.
- Separate data procurement from prediction research.
- Validate with a chronological train/validation split and out-of-sample tests.

## Architecture Principles
- Keep the data layer independent from research and betting logic.
- Preserve provenance for every imported dataset.
- Treat provider licences as first-class architecture constraints.

## Validation Principles
- A model is not eligible for production until it is tested on a verified historical sample with opening and closing lines.
- Closing-line efficiency is a core validation target, not an afterthought.

## Anti-overfitting Rules
- Do not train on the future.
- Do not claim edge without out-of-sample evidence.
- Avoid over-parameterised models until dataset scale is proven.
- Reject any model that only works on a cherry-picked subset.

## Data Quality Principles
- Require bookmaker identity, timestamps, fixture mapping and market metadata.
- Require both opening and closing prices for the same market.
- Require clear licence permission for internal research and backtesting.

## Model Governance
- Every model must be traceable to a documented dataset and licence.
- Every model report must show sample size, coverage, and validation method.
- Every procurement decision requires a written evidence review.

## Definition of Done
A dataset is ready for research only when it has verified historical depth, market coverage, timestamp quality, and producer licence clearance.

## Research Workflow
1. Procure a signed sample export.
2. Audit data quality and lineage.
3. Run chronological validation.
4. Report statistical validity and model risk.
5. Only then decide whether to proceed to model development.
"""
write_text(DOCS_DIR / "CORNERLAB_MANIFESTO.md", manifesto)

write_xlsx(REPORTS_DIR / "provider_scorecard.xlsx", providers)

# Add a compact summary file for the final go/no-go decision.
summary = """# Historical procurement summary

## Minimum dataset required for a scientifically credible Total Corners model
- Minimum seasons: 3 completed seasons
- Minimum matches: 3,000 matches with usable fixture mapping
- Minimum bookmakers: 4 distinct bookmakers
- Minimum odds snapshots: 200,000 records
- Minimum closing lines: 3,000 valid closing lines
- Minimum opening lines: 3,000 valid opening lines
- Minimum timestamp density: at least 15-minute granularity pre-kickoff
- Minimum feature stability: stability score above 0.80 across seasons

## Largest scientific risks
- Market efficiency and closing-line erosion
- Variance and overfitting on a thin historical sample
- Survivorship bias from selected bookmakers or markets
- Selection bias from only using easy-to-access lines
- Concept drift as bookmaker pricing and market rules change
- Closing-line efficiency ambiguity without a true out-of-sample benchmark

## Recommendation
- GO only after a signed sample export proves historical Total Corners odds quality and licence rights.
- NO GO today for full model development because the current evidence does not establish a verified historical odds archive for Total Corners.
"""
write_text(REPORTS_DIR / "provider_historical_procurement_summary.md", summary)

print("Generated Sprint 20 procurement package")

if __name__ == "__main__":
    pass
