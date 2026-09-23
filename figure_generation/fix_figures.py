"""Vector-level label/legend fixes for all 9 figure PDFs.

Approach: edit the existing PDFs in place (originals backed up in
02_figures_original/) with PyMuPDF: replace/add axis labels (with units,
superscripts rendered as raised small runs), move overlapping annotations,
relabel time-axis ticks to seconds, and add missing legend entries.
All positions were measured from the PDFs themselves (text spans +
vector drawings); nothing else in the figures is touched.
"""
import os
import re
import shutil

import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "02_figures")
BAK = os.path.join(HERE, "02_figures_original")

REG_PATH = r"C:\Windows\Fonts\pala.ttf"      # Palatino Linotype regular
BOL_PATH = r"C:\Windows\Fonts\palab.ttf"     # Palatino Linotype bold
FREG = pymupdf.Font(fontfile=REG_PATH)
FBOL = pymupdf.Font(fontfile=BOL_PATH)

MINUS = "\u2212"  # −


def col(c):
    return (((c >> 16) & 255) / 255.0, ((c >> 8) & 255) / 255.0, (c & 255) / 255.0)


def get_spans(page):
    out = []
    for b in page.get_text("rawdict")["blocks"]:
        if b["type"] != 0:
            continue
        for l in b["lines"]:
            for s in l["spans"]:
                chars = s["chars"]
                t = "".join(ch["c"] for ch in chars).strip()
                if not t:
                    continue
                out.append(dict(text=t, bbox=pymupdf.Rect(s["bbox"]), size=s["size"],
                                color=s["color"], font=s["font"],
                                origin=chars[0]["origin"], dir=l["dir"]))
    return out


def find(spans, text, region=None, size=None, all_matches=False):
    res = []
    for s in spans:
        if s["text"] != text:
            continue
        r = s["bbox"]
        if region is not None:
            if not (region[0] - 1 <= r.x0 and r.x1 <= region[2] + 1
                    and region[1] - 1 <= r.y0 and r.y1 <= region[3] + 1):
                continue
        if size is not None and abs(s["size"] - size) > 0.35:
            continue
        res.append(s)
    if not res:
        raise LookupError(f"span not found: {text!r} region={region} size={size}")
    if all_matches:
        return res
    return res[0]


class Editor:
    """Collects redactions AND deferred insertions; on save(), applies the
    redactions first, then draws the new content (so new text can never be
    removed by the redaction of the old text it replaces)."""

    def __init__(self, path):
        self.doc = pymupdf.open(path)
        self.page = self.doc[0]
        self.spans = get_spans(self.page)
        self.redactions = []
        self.deferred = []
        self.log = []

    def span(self, text, region=None, size=None, all_matches=False):
        return find(self.spans, text, region, size, all_matches)

    def remove(self, span, pad=0.4):
        r = span["bbox"]
        self.redactions.append(pymupdf.Rect(r.x0 - pad, r.y0 - pad, r.x1 + pad, r.y1 + pad))
        self.log.append(f"  redact '{span['text']}' @ {tuple(round(v,1) for v in r)}")

    def _font(self, bold):
        return ("PalB", BOL_PATH, FBOL) if bold else ("PalR", REG_PATH, FREG)

    @staticmethod
    def parse(text):
        """'speed (m s^{−1})' -> [('speed (m s', False), ('−1', True), (')', False)]"""
        parts, i = [], 0
        for m in re.finditer(r"\^\{([^}]*)\}", text):
            if m.start() > i:
                parts.append((text[i:m.start()], False))
            parts.append((m.group(1), True))
            i = m.end()
        if i < len(text):
            parts.append((text[i:], False))
        return parts

    def label_width(self, text, size):
        f = FREG
        w = 0.0
        for t, sup in self.parse(text):
            w += f.text_length(t, fontsize=size * (0.7 if sup else 1.0))
            if sup:
                w += 0.12
        return w

    def label(self, pos, text, size, color=(0, 0, 0), bold=False,
              rotate=0, anchor="l"):
        """Queue a label. rotate=0: pos=(x, baseline_y), anchor l/c/r.
        rotate=90: pos=(baseline_x, y_center): text runs bottom-up."""
        def draw():
            fname, fpath, f = self._font(bold)
            W = self.label_width(text, size)
            x, y = pos
            if rotate == 0:
                if anchor == "c":
                    x -= W / 2
                elif anchor == "r":
                    x -= W
                for t, sup in self.parse(text):
                    s = size * (0.7 if sup else 1.0)
                    yy = y - 0.30 * size if sup else y
                    self.page.insert_text(pymupdf.Point(x, yy), t, fontsize=s,
                                          fontname=fname, fontfile=fpath, color=color)
                    x += f.text_length(t, fontsize=s) + (0.12 if sup else 0)
            elif rotate == 90:
                y = y + W / 2  # start at bottom, advance upward
                for t, sup in self.parse(text):
                    s = size * (0.7 if sup else 1.0)
                    xx = x - 0.30 * size if sup else x
                    self.page.insert_text(pymupdf.Point(xx, y), t, fontsize=s,
                                          fontname=fname, fontfile=fpath,
                                          color=color, rotate=90)
                    y -= f.text_length(t, fontsize=s) + (0.12 if sup else 0)
        self.deferred.append(draw)
        self.log.append(f"  insert '{text}' size={size} rot={rotate} at "
                        f"({pos[0]:.1f},{pos[1]:.1f}) anchor={anchor}")

    def replace_centered(self, span, text, size=None, extra_dx=0.0):
        """Remove span, re-insert new text centered on old span center,
        same baseline (horizontal text only)."""
        self.remove(span)
        sz = size or span["size"]
        cx = (span["bbox"].x0 + span["bbox"].x1) / 2 + extra_dx
        self.label((cx, span["origin"][1]), text, sz, color=col(span["color"]),
                   bold="Bol" in span["font"], anchor="c")

    def move(self, span, dx=0.0, dy=0.0):
        """Remove span, re-insert same text shifted by (dx, dy)."""
        self.remove(span)
        x, y = span["origin"]
        self.label((x + dx, y + dy), span["text"], span["size"],
                   color=col(span["color"]), bold="Bol" in span["font"],
                   rotate=0 if abs(span["dir"][0]) > 0.5 else 90)

    def relabel_tick(self, span, new_text):
        """Replace a tick label, keeping its center and baseline."""
        self.remove(span)
        cx = (span["bbox"].x0 + span["bbox"].x1) / 2
        self.label((cx, span["origin"][1]), new_text, span["size"],
                   color=col(span["color"]), bold="Bol" in span["font"], anchor="c")

    def line(self, p0, p1, color, width):
        def draw():
            self.page.draw_line(pymupdf.Point(*p0), pymupdf.Point(*p1),
                                color=color, width=width)
        self.deferred.append(draw)

    def circle(self, center, r, fill):
        def draw():
            self.page.draw_circle(pymupdf.Point(*center), radius=r, color=None, fill=fill)
        self.deferred.append(draw)

    def rect(self, rect, fill):
        def draw():
            self.page.draw_rect(pymupdf.Rect(*rect), color=None, fill=fill)
        self.deferred.append(draw)

    def save(self, out=None):
        for r in self.redactions:
            self.page.add_redact_annot(r)
        self.page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE,
                                   graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
                                   text=pymupdf.PDF_REDACT_TEXT_REMOVE)
        for draw in self.deferred:
            draw()
        target = out or self.doc.name
        tmp = target + ".tmp"
        self.doc.save(tmp, deflate=True, garbage=3)
        self.doc.close()
        os.replace(tmp, target)


BLUE = (0.0, 0.4470588, 0.6980392)   # #0072B2  human
ORANGE = (0.8352941, 0.3686275, 0.0)  # #D55E00  AV


# ----------------------------------------------------------------------
def fix_fig1():
    e = Editor(os.path.join(FIGS, "fig1.pdf"))
    print("fig1.pdf")
    # (b) y-axis label: speed regimes 0-5..>15 (m s^-1)
    e.label((168.5, 61.6), f"speed regime (m s^{{{MINUS}1}})", 8, rotate=90)
    # (d) x-axis label below rotated tick labels
    e.label((392.0, 266.3), f"speed regime (m s^{{{MINUS}1}})", 8, anchor="c")
    e.save()
    print("\n".join(e.log))


def fix_fig2():  # fig2.pdf
    e = Editor(os.path.join(FIGS, "fig2.pdf"))
    print("fig2.pdf")
    # (d) y label 'speed' -> with unit; keep baseline x, center on old center
    s = e.span("speed", region=(0, 185, 20, 218))
    cy = (s["bbox"].y0 + s["bbox"].y1) / 2
    e.remove(s)
    e.label((s["origin"][0], cy), f"speed (m s^{{{MINUS}1}})", 7, rotate=90)
    # (d) x label 'disturbance' -> with unit
    e.replace_centered(e.span("disturbance", region=(40, 255, 130, 275)),
                       f"disturbance (m s^{{{MINUS}1}})")
    # (h) move '0.95' up 8 pt so it clears the dashed gain=1 line
    e.move(e.span("0.95", region=(180, 300, 215, 322)), dy=-8.0)
    e.save()
    print("\n".join(e.log))


def fix_fig3():  # fig3.pdf
    e = Editor(os.path.join(FIGS, "fig3.pdf"))
    print("fig3.pdf")
    # (a) x label 'speed' -> 'speed regime (m s^-1)'
    s = e.span("speed", region=(60, 125, 120, 145))
    e.replace_centered(s, f"speed regime (m s^{{{MINUS}1}})")
    base_y = s["origin"][1]
    # (b) add x label, aligned with panel a's label row
    e.label((241.0, base_y), f"speed regime (m s^{{{MINUS}1}})", 8, anchor="c")
    # (c) move 'high'/'VSP' and 'speed'/'s.d.' spoke labels up 12 pt and
    # outward so the two-line blocks sit between the circle and the bottom
    # axis spine (y=116.3), clear of the theta tick numbers (y>=120.1) and
    # of the outer arc.  The redaction rectangles of the moved labels
    # overlap the '0.25'/'0.75' tick labels, so those two ticks are
    # redacted as well and redrawn at their original positions.
    for t, dx in [("high", -10.0), ("VSP", -10.0), ("speed", 12.0), ("s.d.", 12.0)]:
        e.move(e.span(t, region=(340, 100, 440, 130)), dx=dx, dy=-12.0)
    for t in ["0.25", "0.75"]:
        s = e.span(t, region=(340, 115, 450, 132))
        e.remove(s)
        e.label((s["origin"][0], s["origin"][1]), t, s["size"],
                color=col(s["color"]), anchor="l")
    # (e) move regime labels left 5 pt
    regs = [e.span(t, region=(165, 160, 200, 245)) for t in ["0-5", "5-10", "10-15", ">15"]]
    for s in regs:
        e.move(s, dx=-5.0)
    # (e) add rotated y-axis label
    e.label((172.4, 193.1), f"speed regime (m s^{{{MINUS}1}})", 8, rotate=90)
    e.save()
    print("\n".join(e.log))


def fix_fig4():  # fig4.pdf
    e = Editor(os.path.join(FIGS, "fig4.pdf"))
    print("fig4.pdf")
    tlab = f"speed (m s^{{{MINUS}1}})"
    # panels b and c: ticks 20/40 -> elapsed seconds 122/244; label 'time (s)'
    for xlo, xhi in [(185, 280), (335, 430)]:
        e.relabel_tick(e.span("20", region=(xlo, 112, xhi, 128)), "122")
        e.relabel_tick(e.span("40", region=(xlo, 112, xhi, 128)), "244")
        e.replace_centered(e.span("time step", region=(xlo, 122, xhi, 142)), "time (s)")
    # colorbar labels (horizontal, below the colorbar tick column)
    e.label((303.0, 114.5), tlab, 5, anchor="c")
    e.label((457.3, 114.5), tlab, 5, anchor="r")
    # (e) left y label: speed s.d. with unit
    s = e.span("speed s.d.", region=(165, 170, 185, 215))
    cx = (s["bbox"].x0 + s["bbox"].y1 - s["bbox"].y1)  # placeholder, use y-center below
    cy = (s["bbox"].y0 + s["bbox"].y1) / 2
    e.remove(s)
    e.label((s["origin"][0], cy), f"speed s.d. (m s^{{{MINUS}1}})", 8, rotate=90)
    # (f) x label for 0% / 50% ticks
    e.label((392.5, 262.5), "AV penetration", 8, anchor="c")
    e.save()
    print("\n".join(e.log))


def fix_fig5():  # fig5.pdf
    e = Editor(os.path.join(FIGS, "fig5.pdf"))
    print("fig5.pdf")
    # (c) add 'AV' legend entry above 'human' (line + marker + text)
    e.line((368.6, 96.2), (379.6, 96.2), ORANGE, 1.5)
    e.circle((374.1, 96.2), 3.0, ORANGE)
    e.label((384.0, 97.75), "AV", 5.5)
    # (d) x label 'disturbance' -> with unit
    e.replace_centered(e.span("disturbance", region=(80, 250, 140, 270)),
                       f"disturbance (m s^{{{MINUS}1}})")
    # (e) ticks '20%'->'20' etc., label 'penetration' -> 'penetration (%)'
    for old, new in [("20%", "20"), ("50%", "50"), ("100%", "100")]:
        e.relabel_tick(e.span(old, region=(200, 238, 330, 258)), new)
    e.replace_centered(e.span("penetration", region=(200, 250, 290, 270)),
                       "penetration (%)")
    # (f) x-axis name 'criterion' (below the rotated column tick labels,
    # which descend to y~260), y-axis name 'finding'
    e.label((411.0, 267.5), "criterion", 8, anchor="c")
    e.label((325.0, 190.0), "finding", 8, rotate=90)
    e.save()
    print("\n".join(e.log))


def fix_ed1():  # ed1_bias.pdf
    e = Editor(os.path.join(FIGS, "ed1_bias.pdf"))
    print("ed1_bias.pdf")
    dist = f"disturbance (m s^{{{MINUS}1}})"
    # panels a, c, e x labels
    for reg in [(140, 120, 200, 142), (75, 250, 135, 272), (370, 250, 432, 272)]:
        e.replace_centered(e.span("disturbance", region=reg), dist)
    # (a) y label 'noise' -> with unit
    s = e.span("noise", region=(20, 50, 40, 80))
    cy = (s["bbox"].y0 + s["bbox"].y1) / 2
    e.remove(s)
    e.label((s["origin"][0], cy), f"noise (m s^{{{MINUS}1}})", 7.5, rotate=90)
    e.save()
    print("\n".join(e.log))


def fix_ed2():  # ed2_controller.pdf
    e = Editor(os.path.join(FIGS, "ed2_controller.pdf"))
    print("ed2_controller.pdf")
    pos = "position (vehicle index)"
    # panels a, d, f x labels
    for reg in [(60, 122, 105, 142), (60, 248, 105, 268), (355, 248, 405, 268)]:
        e.replace_centered(e.span("position", region=reg), pos)
    # panels b, c: time ticks + labels
    for xlo, xhi in [(175, 265), (322, 412)]:
        e.relabel_tick(e.span("20", region=(xlo, 112, xhi, 128)), "122")
        e.relabel_tick(e.span("40", region=(xlo, 112, xhi, 128)), "244")
        e.replace_centered(e.span("time step", region=(xlo, 122, xhi, 142)), "time (s)")
    # colorbar labels
    tlab = f"speed (m s^{{{MINUS}1}})"
    e.label((290.5, 114.5), tlab, 5, anchor="c")
    e.label((445.5, 114.5), tlab, 5, anchor="r")
    # (a) legend for the two chain-gain curves
    e.line((43.0, 25.5), (54.0, 25.5), BLUE, 1.5)
    e.label((57.0, 27.05), "all-human", 6)
    e.line((43.0, 33.5), (54.0, 33.5), ORANGE, 1.5)
    e.label((57.0, 35.05), "all-AV", 6)
    # (d) y label 'speed s.d.' -> with unit
    s = e.span("speed s.d.", region=(0, 168, 20, 218))
    cy = (s["bbox"].y0 + s["bbox"].y1) / 2
    e.remove(s)
    e.label((s["origin"][0], cy), f"speed s.d. (m s^{{{MINUS}1}})", 7.5, rotate=90)
    e.save()
    print("\n".join(e.log))


def fix_ed3():  # ed3_sensitivity.pdf
    e = Editor(os.path.join(FIGS, "ed3_sensitivity.pdf"))
    print("ed3_sensitivity.pdf")
    jc = f"jerk cap (m s^{{{MINUS}3}})"
    # panels a and c: y label 'jerk cap' -> with unit
    for reg in [(24, 45, 44, 85), (24, 172, 44, 212)]:
        s = e.span("jerk cap", region=reg)
        cy = (s["bbox"].y0 + s["bbox"].y1) / 2
        e.remove(s)
        e.label((s["origin"][0], cy), jc, 7.5, rotate=90)
    # (b) ticks '20%'->'20' etc. + x label
    for old, new in [("20%", "20"), ("50%", "50"), ("100%", "100")]:
        e.relabel_tick(e.span(old, region=(355, 112, 450, 130)), new)
    e.label((404.0, 138.5), "penetration (%)", 7.5, anchor="c")
    e.save()
    print("\n".join(e.log))


def fix_ed4():  # ed4_validation.pdf
    e = Editor(os.path.join(FIGS, "ed4_validation.pdf"))
    print("ed4_validation.pdf")
    mj = f"median jerk (m s^{{{MINUS}3}})"
    # (a) y label with unit
    s = e.span("median jerk", region=(0, 40, 20, 90))
    cy = (s["bbox"].y0 + s["bbox"].y1) / 2
    e.remove(s)
    e.label((s["origin"][0], cy), mj, 7.5, rotate=90)
    # (a) colorbar label (left of the colorbar strip)
    e.label((301.5, 65.5), mj, 5.5, rotate=90)
    # (b) x label with unit
    e.replace_centered(e.span("95th p. jerk", region=(380, 122, 440, 142)),
                       f"95th p. jerk (m s^{{{MINUS}3}})")
    # (c) y label with unit
    s = e.span("95th p. jerk", region=(30, 165, 55, 220))
    cy = (s["bbox"].y0 + s["bbox"].y1) / 2
    e.remove(s)
    e.label((s["origin"][0], cy), f"95th p. jerk (m s^{{{MINUS}3}})", 7.5, rotate=90)
    # (b) y-axis title for the speed-bin stripes
    e.label((345.2, 66.8), f"speed (m s^{{{MINUS}1}})", 7.5, rotate=90)
    # (b) color legend for the two density bands (top-right; the bands
    # stay below y=42 for x>400, so two stacked rows fit above them)
    e.rect((406.0, 21.9, 411.0, 24.9), BLUE)
    e.label((413.5, 24.95), "Human (perc.)", 5.5)
    e.rect((406.0, 28.9, 411.0, 31.9), ORANGE)
    e.label((413.5, 31.95), "AV", 5.5)
    e.save()
    print("\n".join(e.log))


def relabel_vehicle_ticks():
    """Heat-map rows correspond to platoon vehicles 5,15,25,35,45,55
    (sampled every 10 positions), not indices 0-5: relabel the y ticks
    of the space-time heat maps in fig4.pdf and ed2_controller.pdf,
    right-aligned on the original tick column."""
    panels = [
        ("fig4.pdf", [(183.0, 190.0, 15.0, 110.0, 187.5)]),          # panel b
        ("fig4.pdf", [(331.5, 338.5, 15.0, 110.0, 336.1)]),          # panel c
        ("ed2_controller.pdf", [(171.0, 178.0, 15.0, 110.0, 175.5)]),  # panel b
        ("ed2_controller.pdf", [(319.5, 326.5, 15.0, 110.0, 324.1)]),  # panel c
    ]
    new = ["5", "15", "25", "35", "45", "55"]
    for pdf, regs in panels:
        e = Editor(os.path.join(FIGS, pdf))
        print(f"{pdf}: vehicle ticks")
        for (x0, x1, y0, y1, redge) in regs:
            ticks = sorted([s for s in e.spans
                            if x0 <= s["bbox"].x0 and s["bbox"].x1 <= x1
                            and y0 <= s["bbox"].y0 and s["bbox"].y1 <= y1],
                           key=lambda s: s["bbox"].y0)
            assert len(ticks) == 6, f"expected 6 ticks, got {len(ticks)}"
            for s, lab in zip(ticks, new):
                e.remove(s)
                e.label((redge, s["origin"][1]), lab, s["size"],
                        color=col(s["color"]), anchor="r")
        e.save()


def fill_ed1_e():
    """Fill the empty panel e of ed1_bias.pdf with the estimator-calibration
    curve (est./true gain vs disturbance at the operating noise level),
    drawn from ED1_bias_full in the source-data workbook.  Axis mapping
    measured from the PDF: x = 379.8 + 69.2*(log10(sigma_d)+1),
    y = 151.9 + (1.05 - v)*356."""
    import math
    import pandas as pd
    e = Editor(os.path.join(FIGS, "ed1_bias.pdf"))
    print("ed1_bias.pdf panel e fill")
    wb = os.path.join(HERE, "04_source_data", "figure_source_data.xlsx")
    df = pd.read_excel(wb, "ED1_bias_full")
    s = df[(df.noise == 0.15) & (df.L == 150) & (df.g_true == 1.0)].sort_values("sigma_d")
    PINK = (0.8, 0.4745098, 0.6549020)
    px = lambda sd: 379.8 + 69.2 * (math.log10(sd) + 1.0)
    py = lambda v: 151.9 + (1.05 - v) * 356.0
    pts = [(px(sd), py(v)) for sd, v in zip(s.sigma_d, s.bias_ratio)]
    # dotted analysis-threshold line at sigma_d = 0.3 (style as in panel c)
    e.page.draw_line(pymupdf.Point(px(0.3), 145.9), pymupdf.Point(px(0.3), 240.3),
                     color=(0, 0, 0), width=0.8, dashes="[.8 1.32] 0")
    # curve + markers
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        e.page.draw_line(pymupdf.Point(x0, y0), pymupdf.Point(x1, y1),
                         color=PINK, width=1.2)
    for x, y in pts:
        e.page.draw_circle(pymupdf.Point(x, y), radius=2.0, color=None, fill=PINK)
    # small legend, style matched to panel d
    e.page.draw_line(pymupdf.Point(358.0, 153.0), pymupdf.Point(369.0, 153.0),
                     color=PINK, width=1.2)
    e.label((372.0, 155.1), "\u03c3 = 0.15", 5.5)
    e.save()
    print("  curve points:", [(round(x, 1), round(y, 1)) for x, y in pts])


if __name__ == "__main__":
    for fn in (fix_fig1, fix_fig2, fix_fig3, fix_fig4, fix_fig5,
               fix_ed1, fix_ed2, fix_ed3, fix_ed4):
        fn()
    print("\nAll figures updated.")
