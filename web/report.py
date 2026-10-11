"""PDF reports: a portfolio report (what needs action, results) and a one-SKU plan.

Built with reportlab only. Text is limited to characters the built-in PDF fonts support (no currency symbols).
"""
from datetime import date
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.graphics.charts.barcharts import HorizontalBarChart
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import CondPageBreak, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

INK, MUTED, BRAND = colors.HexColor("#0F1B33"), colors.HexColor("#55637F"), colors.HexColor("#2563EB")
BAD, WARN, OK = colors.HexColor("#BE123C"), colors.HexColor("#B45309"), colors.HexColor("#047857")
LINE, HEAD = colors.HexColor("#D8E0EE"), colors.HexColor("#EEF2F8")
VIOLET = colors.HexColor("#7C5CFF")
WIDTH = A4[0] - 36 * mm

H1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=INK, spaceAfter=2)
H2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=INK, spaceBefore=14, spaceAfter=6)
BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=9.5, leading=13.5, textColor=INK)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8.5, leading=11.5, textColor=MUTED)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.5, leading=11)

STATUS_LABEL = {"Reorder now": "Urgent", "Low": "Warning", "Healthy": "Healthy", "On order": "On order"}
STATUS_COLOR = {"Reorder now": BAD, "Low": WARN, "Healthy": OK, "On order": BRAND}


def _p(text, style=BODY):
    return Paragraph(escape(str(text)), style)


def _fmt(n):
    return f"{round(n):,}"


def _table(rows, widths, header=True, align_right=(), status_col=None, status_values=None, font=8.5):
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    style = [("FONT", (0, 0), (-1, -1), "Helvetica", font), ("TEXTCOLOR", (0, 0), (-1, -1), INK),
             ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), HEAD), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", font), ("TEXTCOLOR", (0, 0), (-1, 0), MUTED)]
    for c in align_right:
        style.append(("ALIGN", (c, 0), (c, -1), "RIGHT"))
    if status_col is not None and status_values:
        for i, sv in enumerate(status_values, start=1 if header else 0):
            style += [("TEXTCOLOR", (status_col, i), (status_col, i), STATUS_COLOR.get(sv, INK)), ("FONT", (status_col, i), (status_col, i), "Helvetica-Bold", font)]
    t.setStyle(TableStyle(style))
    return t


def _legend(items, width):
    d = Drawing(width, 14)
    x = 0
    for label, color, dashed in items:
        if dashed:
            d.add(Line(x, 6, x + 16, 6, strokeColor=color, strokeWidth=1.5, strokeDashArray=[2, 2]))
        else:
            d.add(Line(x, 6, x + 16, 6, strokeColor=color, strokeWidth=2))
        d.add(String(x + 20, 3, label, fontName="Helvetica", fontSize=8, fillColor=MUTED))
        x += 26 + 5.2 * len(label)
    return d


def _line_chart(dates, series, height=170, y_title="Units per day"):
    """series: list of (label, values-with-None, color, dashed). Returns a Drawing."""
    w = WIDTH
    d = Drawing(w, height + 22)
    lp = LinePlot()
    lp.x, lp.y, lp.width, lp.height = 40, 22, w - 52, height - 22
    data = []
    for _, vals, _, _ in series:
        data.append([(i, v) for i, v in enumerate(vals) if v is not None] or [(0, 0)])
    lp.data = data
    for i, (_, _, color, dashed) in enumerate(series):
        lp.lines[i].strokeColor = color
        lp.lines[i].strokeWidth = 1.6 if not dashed else 1.2
        if dashed:
            lp.lines[i].strokeDashArray = [3, 2]
        lp.lines[i].symbol = None
    n = len(dates)
    lp.xValueAxis.valueMin, lp.xValueAxis.valueMax = 0, max(n - 1, 1)
    steps = sorted({int(round(k * (n - 1) / 5)) for k in range(6)}) if n > 1 else [0]
    lp.xValueAxis.valueSteps = steps
    lp.xValueAxis.labelTextFormat = lambda v: dates[min(max(int(round(v)), 0), n - 1)][5:]
    lp.xValueAxis.labels.fontName = lp.yValueAxis.labels.fontName = "Helvetica"
    lp.xValueAxis.labels.fontSize = lp.yValueAxis.labels.fontSize = 7.5
    lp.yValueAxis.valueMin = 0
    lp.yValueAxis.gridStrokeColor = LINE
    lp.yValueAxis.visibleGrid = True
    lp.yValueAxis.labelTextFormat = lambda v: f"{v:,.0f}"
    d.add(lp)
    d.add(String(2, height + 6, y_title, fontName="Helvetica", fontSize=7.5, fillColor=MUTED))
    return KeepTogether([_legend([(s[0], s[2], s[3]) for s in series], w), d])


def _bar_chart(rows, key, height=None):
    n = len(rows)
    h = height or max(60, 16 * n + 30)
    d = Drawing(WIDTH, h)
    bc = HorizontalBarChart()
    bc.x, bc.y, bc.width, bc.height = 70, 18, WIDTH - 90, h - 28
    bc.data = [[r[key] for r in reversed(rows)]]
    bc.categoryAxis.categoryNames = [r["sku"] for r in reversed(rows)]
    bc.categoryAxis.labels.fontName = bc.valueAxis.labels.fontName = "Helvetica"
    bc.categoryAxis.labels.fontSize = bc.valueAxis.labels.fontSize = 7.5
    bc.bars[0].fillColor = BRAND
    bc.bars[0].strokeColor = None
    bc.valueAxis.valueMin = 0
    bc.valueAxis.visibleGrid = True
    bc.valueAxis.gridStrokeColor = LINE
    bc.valueAxis.labelTextFormat = lambda v: f"{v:,.0f}"
    bc.barWidth = 9
    d.add(bc)
    return d


def _footer(title):
    def draw(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, 10 * mm, f"SupplyMind - AI for supply chain  |  {title}")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()
    return draw


def _header(title, subtitle, is_demo):
    out = [_p("SupplyMind", ParagraphStyle("brand", parent=BODY, fontName="Helvetica-Bold", fontSize=9, textColor=BRAND)),
           _p(title, H1), _p(subtitle, SMALL)]
    if is_demo:
        banner = Table([[_p("DEMO DATA: these numbers are generated examples, not real results.",
                            ParagraphStyle("demo", parent=BODY, fontName="Helvetica-Bold", textColor=BAD))]], colWidths=[WIDTH])
        banner.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, BAD), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
        out += [Spacer(1, 6), banner]
    return out + [Spacer(1, 8)]


def _doc(buf, title):
    return SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=18 * mm,
                             title=title, author="SupplyMind")


def _kv_table(pairs, cols=2):
    rows = []
    for i in range(0, len(pairs), cols):
        row = []
        for label, value in pairs[i:i + cols]:
            row += [_p(label, SMALL), _p(value, ParagraphStyle("kv", parent=BODY, fontName="Helvetica-Bold", fontSize=11))]
        while len(row) < cols * 2:
            row += ["", ""]
        rows.append(row)
    t = Table(rows, colWidths=[WIDTH / (cols * 2) * (1.15 if k % 2 == 0 else 0.85) for k in range(cols * 2)])
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return t


# ------------------------------------------------------------------ portfolio report
def portfolio_report(portfolio, experiments=None, is_demo=False, today=None):
    today = today or date.today()
    k, sc = portfolio["kpis"], portfolio["status_counts"]
    buf = BytesIO()
    doc = _doc(buf, "SupplyMind inventory report")
    s = _header("Inventory report", f"Generated {today.isoformat()}. Covers {k['analysed']} of {k['sku_count']} SKUs that have been analysed.", is_demo)
    s.append(_kv_table([
        ("Total expected cost", _fmt(k["total_cost"])), ("Units to reorder now", _fmt(k["reorder_units"])),
        ("Urgent SKUs", str(sc.get("Reorder now", 0))), ("Warning SKUs", str(sc.get("Low", 0) + sc.get("On order", 0))),
        ("Healthy SKUs", str(sc.get("Healthy", 0))), ("Average forecast certainty", f"{k['avg_certainty_pct']:.1f}%"),
    ]))
    alerts = portfolio["alerts"]
    action = [a for a in alerts if a["status"] != "Healthy"][:20]
    s.append(_p("What needs action", H2))
    if action:
        rows = [["SKU", "Status", "Stock now", "Reorder at", "Days left", "Order qty"]]
        for a in action:
            rows.append([a["sku"], STATUS_LABEL.get(a["status"], a["status"]), _fmt(a["stock_now"]), _fmt(a["reorder_point"]), f"{a['days_left'] if 'days_left' in a else a['days_of_cover']:.1f}", _fmt(a["order_qty"])])
        s.append(_table(rows, [WIDTH * f for f in (0.26, 0.15, 0.14, 0.15, 0.13, 0.17)], align_right=(2, 3, 4, 5),
                        status_col=1, status_values=[a["status"] for a in action]))
        s.append(_p("Days left is current stock divided by the forecast daily demand. Showing the 20 most urgent SKUs that are not healthy.", SMALL))
    else:
        s.append(_p("No SKUs need action. Every analysed SKU has comfortable stock.", BODY))
    if portfolio["top_orders"]:
        s.append(KeepTogether([_p("Largest recommended orders", H2), _bar_chart(portfolio["top_orders"], "qty"),
                               _p("Units per order, top SKUs.", SMALL)]))
    if experiments:
        s.append(CondPageBreak(90 * mm))
        s += _experiments_section(experiments)
    s.append(Spacer(1, 10))
    s.append(_p("Forecast certainty is 100% minus the typical plus-or-minus range of the forecast. The range is the 80% lower-to-upper bound, "
                "so actual demand should fall inside it about 8 days in 10.", SMALL))
    doc.build(s, onFirstPage=_footer("Inventory report"), onLaterPages=_footer("Inventory report"))
    return buf.getvalue()


def _experiments_section(exp):
    out = [_p("Experiment results", H2),
           _p("Every policy replayed on days it never saw: training used the first 70% of days, tuning the next 15%, and these results the last 15%. "
              "Lower cost is better. Ample storage has no limit; tight storage caps total stock at 85% of normal.", BODY), Spacer(1, 6)]
    for ds in exp["datasets"].values():
        names = [n for n in ds["policies"]["ample"] if not n.startswith("_")]
        rows = [["Policy", "Cost, ample", "Cost, tight", "Served, tight"]]
        for n in names:
            rows.append([n, _fmt(ds["policies"]["ample"][n]["total_cost"]), _fmt(ds["policies"]["tight"][n]["total_cost"]),
                         f"{ds['policies']['tight'][n]['fill_rate_pct']:.1f}%"])
        sv = ds["savings"]
        q = ds["forecast_quality"]
        gain = 100 * (q[0]["wape_pct"] - q[1]["wape_pct"]) / q[0]["wape_pct"]
        block = [_p(f"{ds['name']} ({ds['n_skus']} SKUs, {ds['split']['test_days']} test days)", ParagraphStyle("dsn", parent=BODY, fontName="Helvetica-Bold", spaceBefore=6)),
                 _table(rows, [WIDTH * f for f in (0.46, 0.17, 0.17, 0.20)], align_right=(1, 2, 3)),
                 _p(f"Best policy versus the simple rule: {sv['ample']['savings_pct']:.1f}% lower cost with ample storage ({sv['ample']['best_policy']}), "
                    f"{sv['tight']['savings_pct']:.1f}% with tight storage"
                    + (f" ({sv['tight']['best_policy']})." if sv["tight"]["savings_pct"] > 0 else " (the simple rule was already best)."), SMALL),
                 _p(f"The forecast model has {gain:.0f}% lower error than a plain 28-day average; its 80% range covered {q[1]['coverage_pct']:.1f}% of actual demand.", SMALL)]
        out.append(KeepTogether(block))
    out.append(_p("The decision-focused model and the RL correction are included for comparison. They did not clearly beat the squared-error forecast policy on these datasets.", SMALL))
    return out


# ------------------------------------------------------------------ one-SKU report
def sku_report(fc, plan, is_demo=False, today=None):
    today = today or date.today()
    sku = fc["sku"]
    m = fc["metrics"]
    split = fc["split_index"]
    buf = BytesIO()
    doc = _doc(buf, f"SupplyMind plan for {sku}")
    s = _header(f"Reorder plan: {sku}", f"Generated {today.isoformat()}. Forecast horizon {m['horizon_days']} days.", is_demo)
    status = plan["status"]
    s.append(_kv_table([
        ("Order quantity (units)", _fmt(plan["order_qty"])), ("Reorder point (units)", _fmt(plan["reorder_point"])),
        ("Safety stock (units)", _fmt(plan["safety_stock"])), ("Total expected cost", _fmt(plan["total_cost"])),
        ("Stock on hand now", _fmt(plan["stock_now"])), ("Days of cover", f"{plan['days_of_cover']:.1f}"),
        ("Stock status", STATUS_LABEL.get(status, status)), ("Supplier lead time", f"{plan['lead_time_days']} days"),
        ("Forecast certainty", f"{m['certainty_pct']:.1f}%"), ("Solver status", str(plan["solver_status"])),
    ]))
    s.append(KeepTogether([_p("Demand forecast with bounds", H2), _line_chart(fc["dates"], [
        ("Lower bound (80%)", fc["lower"], MUTED, True), ("Upper bound (80%)", fc["upper"], MUTED, True),
        ("Forecast", fc["forecast"], BRAND, False), ("Actual demand", fc["actual"], INK, False)])]))
    s.append(_p(f"Average forecast {m['avg_daily']:,} units per day; average bounds {m['lower_avg']:,} to {m['upper_avg']:,}. "
                f"Total forecast for the next {m['horizon_days']} days: {m['horizon_total']:,} units.", SMALL))
    rows = [["Date", "Lower bound", "Forecast", "Upper bound"]]
    for i in range(split, len(fc["dates"])):
        rows.append([fc["dates"][i], _fmt(fc["lower"][i]), _fmt(fc["forecast"][i]), _fmt(fc["upper"][i])])
    s.append(_p("Daily bounds", H2))
    s.append(_table(rows, [WIDTH * f for f in (0.31, 0.23, 0.23, 0.23)], align_right=(1, 2, 3), font=8))
    flat = lambda v: [v] * len(plan["dates"])  # noqa: E731
    s.append(KeepTogether([_p("Projected stock" if plan.get("projected") else "Stock on hand", H2), _line_chart(plan["dates"], [
        ("Stock", plan["stock"], BRAND, False), ("Reorder point", flat(plan["reorder_point"]), WARN, True), ("Safety stock", flat(plan["safety_stock"]), BAD, True)],
        y_title="Units in stock")]))
    s.append(_p(("Stock from today if you follow this plan and demand matches the forecast. " if plan.get("projected") else "") +
                f"Orders placed in this chart: {plan['orders_placed']}. Demand served: {plan['projected_service_pct']:.1f}%.", SMALL))
    ex = plan.get("explanation")
    if ex and ex.get("explanation"):
        s.append(_p("Why this plan", H2))
        s.append(_p(ex["explanation"]))
        chk = ex.get("check")
        if chk:
            s.append(_p("Number check: all %d figures match the optimizer output." % chk["checked"] if chk["ok"]
                        else "Number check: these figures do not match the optimizer output, so verify them: " + ", ".join(f"{x:g}" for x in chk["unmatched"]) + ".", SMALL))
    doc.build(s, onFirstPage=_footer(f"Plan for {sku}"), onLaterPages=_footer(f"Plan for {sku}"))
    return buf.getvalue()
