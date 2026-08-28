# pyright: basic
import json
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
ASSETS = REPORTS / "assets"
QA = REPORTS / "qa"
OUT_PPTX = REPORTS / "multi_stl_gvbf_experiment_report.pptx"
MANIFEST = REPORTS / "multi_stl_gvbf_experiment_manifest.json"

W, H = 1920, 1080
EMU_W, EMU_H = 12192000, 6858000
DPI = (144, 144)

COLORS = {
    "ink": "#0F172A",
    "ink2": "#1E293B",
    "panel": "#F8FAFC",
    "paper": "#F1F5F9",
    "muted": "#64748B",
    "grid": "#CBD5E1",
    "white": "#FFFFFF",
    "accent": "#F97316",
    "teal": "#0F766E",
    "blue": "#2563EB",
    "red": "#DC2626",
    "green": "#16A34A",
}

EXP_ORDER = [
    ("biflow", "biflow_w3s", "BiFlow 模式", "联合前向/后向轨迹与条件噪声建模", "#7C3AED"),
    ("flow", "flow_w3s", "Flow 模式", "确定性流轨迹用于风速序列预报", "#0EA5E9"),
    ("condiff", "condiff_w3s", "ConDiff 模式", "基于条件扩散的 GVBF 动力学校正", "#10B981"),
]
THRESHOLDS = [8.0, 13.9, 20.8]


def font(size, bold=False):
    base = "/tmp/opencode/fonts/LXGWWenKai-Regular.ttf"
    return ImageFont.truetype(base, size=size)

F = {
    "title": font(58, True),
    "subtitle": font(28, False),
    "h1": font(42, True),
    "h2": font(30, True),
    "h3": font(24, True),
    "body": font(24, False),
    "body_b": font(24, True),
    "small": font(18, False),
    "small_b": font(18, True),
    "metric": font(50, True),
    "tiny": font(15, False),
}


def hex_to_rgb(value):
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def draw_text(draw, xy, text, fnt, fill, max_width=None, line_gap=8, anchor=None):
    x, y = xy
    if max_width is None:
        draw.text((x, y), text, font=fnt, fill=fill, anchor=anchor)
        return draw.textbbox((x, y), text, font=fnt, anchor=anchor)
    # CJK-aware character-level wrapping
    lines = []
    cur = ""
    for ch in text:
        if ch == " " and cur and draw.textlength(cur + " ", font=fnt) > max_width:
            lines.append(cur)
            cur = ""
            continue
        test = cur + ch
        if draw.textlength(test, font=fnt) <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    line_h = fnt.getbbox("测试Ag")[3] - fnt.getbbox("测试Ag")[1] + line_gap
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=fill)
        y += line_h
    return (x, xy[1], x + max_width, y)


def rounded(draw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def load_experiment(slug, folder, label, tagline, color):
    root = ROOT / "work_dirs" / folder
    cfg = json.loads((root / "obj_config.json").read_text())
    result = json.loads((root / "result.json").read_text())["w10"]
    metrics = json.loads((root / "metrics.json").read_text())
    epochs = []
    for key, value in metrics.items():
        if not key.isdigit():
            continue
        epoch = int(key)
        train = value.get("train", {}).get("loss")
        valid = value.get("valid", {}).get("loss")
        epochs.append({"epoch": epoch, "train_loss": train, "valid_loss": valid, "time": value.get("time")})
    epochs.sort(key=lambda x: x["epoch"])
    valid_points = [p for p in epochs if p["valid_loss"] is not None]
    best = min(valid_points, key=lambda x: x["valid_loss"])
    ckpts = sorted((root / "model").glob("epoch*.ckpt"))
    pred = root / "vis" / "2017_0" / "pred_12.png"
    if not pred.exists():
        candidates = sorted((root / "vis").glob("*/pred_12.png")) or sorted((root / "vis").glob("*/pred_*.png"))
        pred = candidates[0] if candidates else None
    gifs = sorted((root / "vis").glob("*_browse.gif"))
    pngs = sorted((root / "vis").glob("*/pred_*.png"))
    return {
        "slug": slug,
        "folder": folder,
        "label": label,
        "tagline": tagline,
        "color": color,
        "config": {
            "load_model": cfg.get("load_model"),
            "model_config": cfg.get("model_config"),
            "gvbf_mode": cfg.get("gvbf_mode"),
            "gvbf_model_config": cfg.get("gvbf_model_config"),
            "dataset": cfg.get("dataset"),
            "data_config": cfg.get("data_config"),
            "total_seq": cfg.get("total_seq"),
            "img_size": cfg.get("img_size"),
            "in_category": cfg.get("in_category"),
            "out_category": cfg.get("out_category"),
            "threshold": cfg.get("threshold"),
            "learning_rate": cfg.get("learning_rate"),
            "batch_size": cfg.get("batch_size"),
            "epoch": cfg.get("epoch"),
            "seed": cfg.get("seed"),
        },
        "result": result,
        "epochs": epochs,
        "best_epoch": best["epoch"],
        "best_val_loss": best["valid_loss"],
        "final_valid_loss": valid_points[-1]["valid_loss"] if valid_points else None,
        "checkpoint": ckpts[0].name if ckpts else None,
        "prediction_image": str(pred.relative_to(ROOT)) if pred else None,
        "visual_counts": {"gifs": len(gifs), "pngs": len(pngs)},
    }


def load_all():
    items = [load_experiment(*spec) for spec in EXP_ORDER]
    for metric in ["mae", "rmse", "mse"]:
        ranked = sorted(items, key=lambda x: x["result"][metric])
        for idx, item in enumerate(ranked, start=1):
            item.setdefault("rank", {})[metric] = idx
    ranked_hss = sorted(items, key=lambda x: sum(x["result"]["hss"]) / 3, reverse=True)
    for idx, item in enumerate(ranked_hss, start=1):
        item.setdefault("rank", {})["hss_avg"] = idx
    MANIFEST.write_text(json.dumps(items, indent=2, ensure_ascii=False))
    return items


def fit_image(path, size, bg="#E2E8F0"):
    canvas = Image.new("RGB", size, hex_to_rgb(bg))
    if not path:
        return canvas
    img = Image.open(ROOT / path).convert("RGB")
    img.thumbnail(size, Image.Resampling.LANCZOS)
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def cover_image(path, size, bg="#E2E8F0"):
    canvas = Image.new("RGB", size, hex_to_rgb(bg))
    if not path:
        return canvas
    img = Image.open(ROOT / path).convert("RGB")
    scale = max(size[0] / img.width, size[1] / img.height)
    resized = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
    left = max(0, (resized.width - size[0]) // 2)
    top = max(0, (resized.height - size[1]) // 2)
    cropped = resized.crop((left, top, left + size[0], top + size[1]))
    canvas.paste(cropped, (0, 0))
    return canvas


def draw_metric_card(draw, box, label, value, accent, suffix="", inverse=False):
    fill = hex_to_rgb("#111827") if inverse else hex_to_rgb("#FFFFFF")
    outline = hex_to_rgb(accent)
    rounded(draw, box, 28, fill, outline, 3)
    x1, y1, _, _ = box
    draw.text((x1 + 28, y1 + 22), label, font=F["small_b"], fill=hex_to_rgb(accent))
    draw.text((x1 + 28, y1 + 62), f"{value}{suffix}", font=F["metric"], fill=hex_to_rgb("#FFFFFF" if inverse else "#0F172A"))


def draw_loss_chart(exp, path, width=900, height=360):
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    margin = 58
    plot = (margin, 28, width - 28, height - 48)
    train = [(p["epoch"], p["train_loss"]) for p in exp["epochs"] if p["train_loss"] is not None]
    valid = [(p["epoch"], p["valid_loss"]) for p in exp["epochs"] if p["valid_loss"] is not None]
    values = [v for _, v in train + valid]
    if not values:
        d.text((24, 120), "训练历史不可用", font=F["body"], fill=hex_to_rgb(COLORS["muted"]))
        img.save(path, dpi=DPI)
        return
    ymin, ymax = min(values), max(values)
    if math.isclose(ymin, ymax):
        ymin *= 0.95
        ymax *= 1.05
    pad = (ymax - ymin) * 0.08
    ymin -= pad
    ymax += pad
    x0, y0, x1, y1 = plot
    d.rectangle(plot, outline=hex_to_rgb(COLORS["grid"]), width=2)
    for i in range(1, 4):
        y = y0 + (y1 - y0) * i / 4
        d.line((x0, y, x1, y), fill=hex_to_rgb("#E2E8F0"), width=1)
    max_epoch = max(p["epoch"] for p in exp["epochs"])
    def xy(point):
        e, v = point
        x = x0 + (x1 - x0) * (e - 1) / max(1, max_epoch - 1)
        y = y1 - (y1 - y0) * (v - ymin) / (ymax - ymin)
        return x, y
    for series, color, name, yleg in [(train, COLORS["accent"], "训练", 14), (valid, exp["color"], "验证", 42)]:
        if len(series) > 1:
            pts = [xy(p) for p in series]
            d.line(pts, fill=hex_to_rgb(color), width=4)
        d.rounded_rectangle((width - 150, yleg, width - 130, yleg + 20), 4, fill=hex_to_rgb(color))
        d.text((width - 120, yleg - 2), name, font=F["tiny"], fill=hex_to_rgb(COLORS["ink"]))
    d.text((24, 8), "损失曲线", font=F["small_b"], fill=hex_to_rgb(COLORS["ink"]))
    d.text((x0, height - 35), "epoch 1", font=F["tiny"], fill=hex_to_rgb(COLORS["muted"]))
    d.text((x1 - 82, height - 35), f"epoch {max_epoch}", font=F["tiny"], fill=hex_to_rgb(COLORS["muted"]))
    d.text((12, y0 - 4), f"{ymax:.3f}", font=F["tiny"], fill=hex_to_rgb(COLORS["muted"]))
    d.text((12, y1 - 12), f"{ymin:.3f}", font=F["tiny"], fill=hex_to_rgb(COLORS["muted"]))
    img.save(path, dpi=DPI)


def draw_header(d, title, subtitle, accent):
    d.rectangle((0, 0, W, 150), fill=hex_to_rgb(COLORS["ink"]))
    d.rectangle((0, 0, 22, H), fill=hex_to_rgb(accent))
    d.text((82, 34), title, font=F["title"], fill=hex_to_rgb(COLORS["white"]))
    d.text((84, 104), subtitle, font=F["subtitle"], fill=hex_to_rgb("#CBD5E1"))


def draw_setup_slide(exp, idx):
    img = Image.new("RGB", (W, H), hex_to_rgb(COLORS["paper"]))
    d = ImageDraw.Draw(img)
    draw_header(d, f"{exp['label']}: 实验设置", exp["tagline"], exp["color"])
    cfg = exp["config"]
    rounded(d, (80, 205, 840, 930), 34, hex_to_rgb(COLORS["white"]), hex_to_rgb("#E2E8F0"), 2)
    d.text((120, 245), "实验配置", font=F["h1"], fill=hex_to_rgb(COLORS["ink"]))
    rows = [
        ("模式", cfg.get("gvbf_mode")),
        ("GVBF 模型配置", cfg.get("gvbf_model_config")),
        ("数据集", f"{cfg.get('dataset')} / {cfg.get('data_config')}"),
        ("变量", f"{cfg.get('in_category')} -> {cfg.get('out_category')}"),
        ("序列", f"{cfg.get('total_seq')[0]} 输入 + {cfg.get('total_seq')[1]} 预测帧"),
        ("网格尺寸", f"{cfg.get('img_size')[0]} x {cfg.get('img_size')[1]}"),
        ("优化器", f"lr={cfg.get('learning_rate')}, batch={cfg.get('batch_size')}, seed={cfg.get('seed')}"),
        ("最优检查点", exp.get("checkpoint")),
    ]
    y = 330
    for key, value in rows:
        d.text((124, y), key, font=F["small_b"], fill=hex_to_rgb(COLORS["muted"]))
        draw_text(d, (330, y - 4), str(value), F["body"], hex_to_rgb(COLORS["ink"]), max_width=455, line_gap=7)
        y += 66
    rounded(d, (900, 205, 1820, 540), 34, hex_to_rgb(COLORS["ink2"]), None)
    d.text((946, 250), "本实验测试内容", font=F["h1"], fill=hex_to_rgb(COLORS["white"]))
    bullets = [
        "三种模式均在 WeatherBench wind_3s 任务上进行。",
        "基于 12 帧历史风速数据预报未来 12 帧。",
        "评估指标：MAE/MSE/RMSE 及阈值指标（8.0, 13.9, 20.8）。",
    ]
    by = 325
    for b in bullets:
        d.ellipse((950, by + 8, 966, by + 24), fill=hex_to_rgb(exp["color"]))
        draw_text(d, (982, by), b, F["body"], hex_to_rgb("#E2E8F0"), max_width=760, line_gap=8)
        by += 72
    sample = cover_image(exp["prediction_image"], (860, 300))
    rounded(d, (900, 585, 1820, 930), 34, hex_to_rgb(COLORS["white"]), hex_to_rgb("#E2E8F0"), 2)
    img.paste(sample, (930, 650))
    d.text((930, 610), "代表性预测帧：2017_0 / pred_12", font=F["body_b"], fill=hex_to_rgb(COLORS["ink2"]))
    d.text((82, 1012), f"数据来源：work_dirs/{exp['folder']} | {exp['visual_counts']['pngs']} 张 PNG 帧, {exp['visual_counts']['gifs']} 个 GIF 浏览", font=F["small"], fill=hex_to_rgb(COLORS["ink2"]))
    path = ASSETS / f"slide_{idx:02d}_{exp['slug']}_setup.png"
    img.save(path, dpi=DPI)
    return path


def draw_results_slide(exp, idx):
    chart_path = ASSETS / f"{exp['slug']}_loss_chart.png"
    draw_loss_chart(exp, chart_path)
    img = Image.new("RGB", (W, H), hex_to_rgb(COLORS["paper"]))
    d = ImageDraw.Draw(img)
    draw_header(d, f"{exp['label']}: 实验结果", "测试指标与训练动态", exp["color"])
    r = exp["result"]
    cards = [
        ("MAE", f"{r['mae']:.3f}", "", 80, exp["rank"].get("mae")),
        ("RMSE", f"{r['rmse']:.3f}", "", 395, exp["rank"].get("rmse")),
        ("最优验证损失", f"{exp['best_val_loss']:.4f}", "", 710, None),
        ("最优 Epoch", str(exp["best_epoch"]), "", 1025, None),
    ]
    for label, val, _suffix, x, rank in cards:
        draw_metric_card(d, (x, 205, x + 285, 345), label, val, exp["color"])
        if rank:
            d.text((x + 30, 315), f"排名 {rank}/3（越低越好）", font=F["tiny"], fill=hex_to_rgb(COLORS["muted"]))
    rounded(d, (1350, 205, 1820, 345), 28, hex_to_rgb(COLORS["ink2"]), None)
    avg_hss = sum(r["hss"]) / 3
    d.text((1380, 225), "平均 HSS", font=F["small_b"], fill=hex_to_rgb("#CBD5E1"))
    d.text((1380, 265), f"{avg_hss:.3f}", font=F["metric"], fill=hex_to_rgb(COLORS["white"]))
    d.text((1585, 312), f"排名 {exp['rank']['hss_avg']}/3", font=F["small_b"], fill=hex_to_rgb("#E2E8F0"))

    rounded(d, (80, 390, 760, 920), 34, hex_to_rgb(COLORS["white"]), hex_to_rgb("#E2E8F0"), 2)
    d.text((120, 425), "阈值指标", font=F["h2"], fill=hex_to_rgb(COLORS["ink"]))
    d.text((120, 470), "风速阈值下的 CSI / POD / FAR / HSS", font=F["small"], fill=hex_to_rgb(COLORS["muted"]))
    headers = ["阈值", "CSI", "POD", "FAR", "HSS"]
    xcols = [125, 245, 365, 485, 605]
    y = 540
    for x, h in zip(xcols, headers):
        d.text((x, y), h, font=F["small_b"], fill=hex_to_rgb(exp["color"]))
    y += 46
    for i, thr in enumerate(THRESHOLDS):
        vals = [f"{thr:g}", f"{r['csi'][i]:.3f}", f"{r['pod'][i]:.3f}", f"{r['far'][i]:.3f}", f"{r['hss'][i]:.3f}"]
        if i % 2 == 0:
            rounded(d, (105, y - 8, 720, y + 38), 12, hex_to_rgb("#F8FAFC"), None)
        for x, v in zip(xcols, vals):
            d.text((x, y), v, font=F["body"], fill=hex_to_rgb(COLORS["ink"]))
        y += 60
    d.text((120, 825), "观察结论", font=F["h3"], fill=hex_to_rgb(COLORS["ink"]))
    if exp["slug"] == "flow":
        note = "三个实验中综合最优：MAE/RMSE 最低，平均 HSS 最高。"
    elif exp["slug"] == "condiff":
        note = "检测指标接近 Flow，但误差幅度仍然较高。"
    else:
        note = "验证集最优出现较早（epoch 4），最终测试指标弱于其他模式。"
    draw_text(d, (120, 860), note, F["body"], hex_to_rgb(COLORS["ink"]), max_width=585)

    rounded(d, (810, 390, 1820, 920), 34, hex_to_rgb(COLORS["white"]), hex_to_rgb("#E2E8F0"), 2)
    d.text((850, 425), "训练动态", font=F["h2"], fill=hex_to_rgb(COLORS["ink"]))
    chart = Image.open(chart_path).convert("RGB").resize((900, 360), Image.Resampling.LANCZOS)
    img.paste(chart, (865, 500))
    d.text((865, 875), f"最终验证损失：{exp['final_valid_loss']:.4f} | 检查点：{exp['checkpoint']}", font=F["body"], fill=hex_to_rgb(COLORS["ink2"]))
    d.text((82, 1012), f"指标来源：work_dirs/{exp['folder']}/result.json 与 metrics.json", font=F["small"], fill=hex_to_rgb(COLORS["ink2"]))
    path = ASSETS / f"slide_{idx:02d}_{exp['slug']}_results.png"
    img.save(path, dpi=DPI)
    return path


def make_contact_sheet(slides, path):
    thumbs = []
    for slide in slides:
        im = Image.open(slide).convert("RGB")
        im.thumbnail((600, 338), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (620, 380), "white")
        canvas.paste(im, (10, 10))
        d = ImageDraw.Draw(canvas)
        d.text((14, 352), slide.name, font=F["tiny"], fill=hex_to_rgb(COLORS["muted"]))
        thumbs.append(canvas)
    sheet = Image.new("RGB", (1240, 1140), hex_to_rgb("#E2E8F0"))
    for i, thumb in enumerate(thumbs):
        x = (i % 2) * 620
        y = (i // 2) * 380
        sheet.paste(thumb, (x, y))
    sheet.save(path, dpi=DPI)


def write_xml(zf, name, text):
    zf.writestr(name, text.strip().encode("utf-8"))


def slide_xml(image_rid="rId1"):
    return f'''
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      <p:pic>
        <p:nvPicPr><p:cNvPr id="2" name="Slide image"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>
        <p:blipFill><a:blip r:embed="{image_rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>
        <p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{EMU_W}" cy="{EMU_H}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr>
      </p:pic>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>
'''


def package_pptx(slides):
    with zipfile.ZipFile(OUT_PPTX, "w", zipfile.ZIP_DEFLATED) as zf:
        slide_overrides = "\n".join(f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>' for i in range(1, len(slides) + 1))
        write_xml(zf, "[Content_Types].xml", f'''
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
  {slide_overrides}
</Types>
''')
        write_xml(zf, "_rels/.rels", '''
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
''')
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        write_xml(zf, "docProps/core.xml", f'''
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Multi-STL GVBF Experiment Report</dc:title>
  <dc:creator>Sisyphus</dc:creator>
  <cp:lastModifiedBy>Sisyphus</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>
''')
        write_xml(zf, "docProps/app.xml", f'''
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>OpenCode generated report</Application><PresentationFormat>On-screen Show (16:9)</PresentationFormat><Slides>{len(slides)}</Slides><Company></Company>
</Properties>
''')
        sld_ids = "\n".join(f'<p:sldId id="{256+i}" r:id="rId{2+i}"/>' for i in range(len(slides)))
        write_xml(zf, "ppt/presentation.xml", f'''
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
  <p:sldIdLst>{sld_ids}</p:sldIdLst>
  <p:sldSz cx="{EMU_W}" cy="{EMU_H}" type="wide"/><p:notesSz cx="6858000" cy="9144000"/>
  <p:defaultTextStyle><a:defPPr><a:defRPr lang="en-US"/></a:defPPr></p:defaultTextStyle>
</p:presentation>
''')
        rels = ['<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>']
        rels.extend(f'<Relationship Id="rId{2+i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i+1}.xml"/>' for i in range(len(slides)))
        write_xml(zf, "ppt/_rels/presentation.xml.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + ''.join(rels) + '</Relationships>')
        write_xml(zf, "ppt/slideMasters/slideMaster1.xml", f'''
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst><p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>
''')
        write_xml(zf, "ppt/slideMasters/_rels/slideMaster1.xml.rels", '''
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>
''')
        write_xml(zf, "ppt/slideLayouts/slideLayout1.xml", '''
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
  <p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>
''')
        write_xml(zf, "ppt/slideLayouts/_rels/slideLayout1.xml.rels", '''
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>
''')
        write_xml(zf, "ppt/theme/theme1.xml", '''
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Multi-STL"><a:themeElements><a:clrScheme name="Multi-STL"><a:dk1><a:srgbClr val="0F172A"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="1E293B"/></a:dk2><a:lt2><a:srgbClr val="F8FAFC"/></a:lt2><a:accent1><a:srgbClr val="F97316"/></a:accent1><a:accent2><a:srgbClr val="0EA5E9"/></a:accent2><a:accent3><a:srgbClr val="10B981"/></a:accent3><a:accent4><a:srgbClr val="7C3AED"/></a:accent4><a:accent5><a:srgbClr val="2563EB"/></a:accent5><a:accent6><a:srgbClr val="64748B"/></a:accent6><a:hlink><a:srgbClr val="2563EB"/></a:hlink><a:folHlink><a:srgbClr val="7C3AED"/></a:folHlink></a:clrScheme><a:fontScheme name="DejaVu"><a:majorFont><a:latin typeface="DejaVu Sans"/></a:majorFont><a:minorFont><a:latin typeface="DejaVu Sans"/></a:minorFont></a:fontScheme><a:fmtScheme name="Default"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle/></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme></a:themeElements></a:theme>
''')
        for i, slide in enumerate(slides, start=1):
            write_xml(zf, f"ppt/slides/slide{i}.xml", slide_xml())
            write_xml(zf, f"ppt/slides/_rels/slide{i}.xml.rels", f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/slide{i}.png"/></Relationships>')
            zf.write(slide, f"ppt/media/slide{i}.png")


def verify(slides):
    assert len(slides) == 6, f"expected 6 slides, got {len(slides)}"
    assert OUT_PPTX.exists() and OUT_PPTX.stat().st_size > 0
    assert MANIFEST.exists() and MANIFEST.stat().st_size > 0
    with zipfile.ZipFile(OUT_PPTX) as zf:
        names = zf.namelist()
        assert len([n for n in names if n.startswith("ppt/slides/slide") and n.endswith(".xml")]) == 6
        assert len([n for n in names if n.startswith("ppt/media/slide") and n.endswith(".png")]) == 6
        bad = zf.testzip()
        assert bad is None, f"bad zip entry: {bad}"
    data = json.loads(MANIFEST.read_text())
    folders = {item["folder"] for item in data}
    assert folders == {"biflow_w3s", "flow_w3s", "condiff_w3s"}
    by_slug = {item["slug"]: item for item in data}
    for slug, folder in [("biflow", "biflow_w3s"), ("flow", "flow_w3s"), ("condiff", "condiff_w3s")]:
        source = json.loads((ROOT / "work_dirs" / folder / "result.json").read_text())["w10"]
        for metric in ["mae", "mse", "rmse"]:
            assert by_slug[slug]["result"][metric] == source[metric]


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    experiments = load_all()
    slides = []
    idx = 1
    for exp in experiments:
        slides.append(draw_setup_slide(exp, idx)); idx += 1
        slides.append(draw_results_slide(exp, idx)); idx += 1
    make_contact_sheet(slides, QA / "multi_stl_gvbf_experiment_report_contact_sheet.png")
    package_pptx(slides)
    verify(slides)
    print(OUT_PPTX)
    print(MANIFEST)
    print(QA / "multi_stl_gvbf_experiment_report_contact_sheet.png")

if __name__ == "__main__":
    main()
