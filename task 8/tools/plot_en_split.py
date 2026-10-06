"""Draw split distribution charts using the bundled ReportLab plotting library."""
import json
import base64
import io
import re
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont

from reportlab.graphics import renderSVG
from reportlab.graphics.charts.barcharts import HorizontalBarChart
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.lib.colors import HexColor

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'outputs/en_split'
REPORT = json.loads((DEST / 'distribution.json').read_text(encoding='utf-8'))
SPLITS = ('train', 'dev', 'test')
COLORS = tuple(map(HexColor, ('#2563eb', '#f59e0b', '#059669')))
INK, MUTED, GRID = map(HexColor, ('#172554', '#475569', '#e2e8f0'))


def label(drawing, x, y, text, size=12, color=INK):
    drawing.add(String(x, y, text, fontName='Helvetica', fontSize=size, fillColor=color))


def legend(drawing, y):
    for index, (split, color) in enumerate(zip(SPLITS, COLORS)):
        x = 180 + index * 290
        drawing.add(Rect(x, y - 2, 12, 12, fillColor=color, strokeColor=None))
        label(drawing, x + 20, y, f'{split.title()}  n={REPORT["distributions"][split]["rows"]:,}', 13)


def panel(drawing, labels, prefix, y, height, title, value_max, left=180, width=880):
    chart = HorizontalBarChart()
    chart.x, chart.y, chart.width, chart.height = left, y, width, height
    # ReportLab categories progress bottom-to-top.
    categories = list(reversed(labels))
    counts = [[REPORT['distributions'][split]['features'].get(prefix + cat, 0)
               for cat in categories] for split in SPLITS]
    chart.data = [[100 * count / REPORT['distributions'][split]['rows'] for count in series]
                  for split, series in zip(SPLITS, counts)]
    chart.categoryAxis.categoryNames = categories
    chart.categoryAxis.labels.fontName = 'Helvetica'
    chart.categoryAxis.labels.fontSize = 11
    chart.categoryAxis.labels.fillColor = INK
    chart.categoryAxis.labels.dx = -10
    chart.categoryAxis.strokeColor = GRID
    chart.categoryAxis.tickLeft = 0
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = value_max
    chart.valueAxis.valueStep = 10 if value_max > 40 else 5
    chart.valueAxis.labels.fontSize = 10
    chart.valueAxis.labels.fillColor = MUTED
    chart.valueAxis.strokeColor = GRID
    chart.valueAxis.visibleGrid = True
    chart.valueAxis.gridStrokeColor = GRID
    chart.valueAxis.gridStrokeWidth = .5
    chart.barSpacing, chart.groupSpacing = 2, 13
    chart.barLabelFormat = 'values'
    chart.barLabelArray = [[str(count) for count in series] for series in counts]
    chart.barLabels.fontSize = 9
    chart.barLabels.fillColor = INK
    chart.barLabels.nudge = 6
    chart.barLabels.boxAnchor = 'w'
    for series, color in enumerate(COLORS):
        chart.bars[series].fillColor = color
        chart.bars[series].strokeColor = None
    drawing.add(chart)
    label(drawing, left, y + height + 24, title, 15)
    label(drawing, left, y - 40, 'Share of all rows in each split (%)  |  Numbers beside bars: sample counts', 11, MUTED)


def save(name, drawing):
    path = DEST / (name + '.svg')
    renderSVG.drawToFile(drawing, str(path))
    path.write_text(path.read_text(encoding='utf-8').replace('font-family: Helvetica;', 'font-family: Arial;'), encoding='utf-8')
    # Embed glyph images so exported charts use the same legible font on hosts
    # without an installed SVG font provider. The quantitative marks stay vector.
    ns = 'http://www.w3.org/2000/svg'
    ET.register_namespace('', ns)
    ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
    root = ET.fromstring(path.read_text(encoding='utf-8'))
    for parent in root.iter():
        for element in list(parent):
            if element.tag != '{' + ns + '}text':
                continue
            text = ''.join(element.itertext())
            if not text:
                continue
            style = element.get('style', '')
            size = float(re.search(r'font-size:\s*([\d.]+)px', style).group(1))
            font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', round(size * 3))
            bounds = font.getbbox(text, anchor='ls')
            width, height = bounds[2] - bounds[0] + 4, bounds[3] - bounds[1] + 4
            glyph = Image.new('RGBA', (width, height))
            values = re.search(r'fill:\s*rgb\(([^)]+)\)', style).group(1).split(',')
            fill = tuple(round(float(v.strip().strip('%')) * 2.55) for v in values)
            ImageDraw.Draw(glyph).text((2 - bounds[0], 2 - bounds[1]), text, font=font, fill=fill, anchor='ls')
            buffer = io.BytesIO()
            glyph.save(buffer, format='PNG')
            image = ET.Element('{' + ns + '}image', {
                'x': str(float(element.get('x', 0)) + (bounds[0] - 2) / 3),
                'y': str(float(element.get('y', 0)) + (bounds[1] - 2) / 3),
                'width': str(width / 3), 'height': str(height / 3),
                'aria-label': text,
                '{http://www.w3.org/1999/xlink}href': 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode(),
            })
            if element.get('transform'):
                image.set('transform', element.get('transform'))
            parent.insert(list(parent).index(element), image)
            parent.remove(element)
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)


def main():
    overview = Drawing(1200, 940)
    overview.add(Rect(0, 0, 1200, 940, fillColor=HexColor('#ffffff'), strokeColor=None))
    label(overview, 30, 902, 'EN label distribution: train / dev / test', 23)
    label(overview, 30, 874, 'Temporary 80/10/10 split; video and duplicate-comment groups kept together', 13, MUTED)
    legend(overview, 840)
    panel(overview, ['none', 'individual', 'group'], 'scope:', 610, 180, 'Target scope (mutually exclusive)', 60)
    panel(overview, ['l', 'g', 'b', 't', 'q', 'i', 'a', 'nb', 'lgbtqia+'],
          'identity:', 90, 420, 'Identity support (multi-label; one sample may count in several codes)', 35)
    label(overview, 30, 18, 'i: 14 samples from one video group; a: 1 sample. Both retained in train.', 12, MUTED)
    save('distribution_overview', overview)

    features = REPORT['distributions']
    labels = sorted({key[7:] for split in SPLITS for key in features[split]['features']
                     if key.startswith('target:')},
                    key=lambda value: (-sum(features[s]['features'].get('target:' + value, 0) for s in SPLITS), value))
    height = len(labels) * 48 + 180
    exact = Drawing(1420, height)
    exact.add(Rect(0, 0, 1420, height, fillColor=HexColor('#ffffff'), strokeColor=None))
    label(exact, 30, height - 42, 'Complete target-label distribution', 23)
    label(exact, 30, height - 70, 'Labels ordered by total support; counts shown beside bars, rates on the axis', 13, MUTED)
    legend(exact, height - 102)
    panel(exact, labels, 'target:', 65, len(labels) * 48, '', 55, left=340, width=920)
    save('target_labels', exact)

    other = Drawing(1200, 650)
    other.add(Rect(0, 0, 1200, 650, fillColor=HexColor('#ffffff'), strokeColor=None))
    label(other, 30, 612, 'EN auxiliary annotation distributions', 23)
    legend(other, 575)
    panel(other, ['no', 'yes_implicit', 'yes_explicit'], 'hate:', 345, 170, 'Hate speech', 60)
    panel(other, ['no', 'yes'], 'stereotype:', 90, 140, 'Stereotype', 65)
    save('annotation_labels', other)
    print('Saved distribution_overview.svg, target_labels.svg, annotation_labels.svg')


if __name__ == '__main__':
    main()
