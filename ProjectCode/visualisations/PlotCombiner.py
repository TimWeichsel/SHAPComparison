# visualisations/plot_panel_grid_narrow.py
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

def _get_text_size(draw, text, font):
    try:
        l,t,r,b = draw.textbbox((0,0), text, font=font); return r-l, b-t
    except Exception:
        try:
            l,t,r,b = font.getbbox(text); return r-l, b-t
        except Exception:
            w = draw.textlength(text, font=font)
            asc, desc = font.getmetrics()
            return int(w), int(asc+desc)

def _load_images(folder, model, letter, runtime):
    folder = Path(folder)
    order = ["Exact","Copula","Gaussian","Kernel","Sample"]
    def fname(m):
        return (f"Summary_{model}_{letter}_{int(runtime)}_{m}.png"
                if runtime is not None else f"Summary_{model}_{letter}_{m}.png")
    paths = []
    for m in order:
        p = folder/fname(m)
        if not p.exists():
            alt = folder/f"Summary_{model}_{letter}_{m}.png"
            if not alt.exists():
                raise FileNotFoundError(f"Nicht gefunden: {p} oder {alt}")
            p = alt
        paths.append(p)
    return order, [Image.open(p).convert("RGB") for p in paths]

def make_panel_grid_narrow(
    folder="SummaryPlots",
    model_name="RR",
    data_name_long="Abalone",
    runtime_sec=250,
    plot_area_height_px=900,   # kompakter
    slot_aspect=1.05,          # Breite/Höhe je Slot (kleiner = schmaler)
    row_gap_px=16,             # << kleiner Reihenabstand
    col_gap_px=10,             # enger Spaltenabstand
    global_title_font_size=34, # Gesamtüberschrift: passt laut dir
    bg="white",
    out_name=None,
):
    letter = data_name_long[0].upper()
    order, imgs = _load_images(folder, model_name, letter, runtime_sec)

    # Slotgröße (alle gleich, gestreckt)
    slot_h = (plot_area_height_px - row_gap_px) // 2
    slot_w = int(round(slot_h * slot_aspect))

    plot_area_w = slot_w * 3 + col_gap_px * 2
    plot_area_h = slot_h * 2 + row_gap_px

    # Gesamt-Titel (klein lassen)
    title_h = int(global_title_font_size * 1.2)
    from PIL import Image
    canvas = Image.new("RGB", (plot_area_w, plot_area_h + title_h), color=bg)

    title = f"Summary Plot: {data_name_long} - {model_name}"
    if runtime_sec is not None:
        title += f" (Running Time ≈ {int(runtime_sec)} seconds)"
    draw = ImageDraw.Draw(canvas)
    # Font robust laden
    for cand in ["DejaVuSans.ttf", "/Library/Fonts/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"]:
        try:
            font = ImageFont.truetype(cand, global_title_font_size); break
        except Exception:
            font = None
    if font is None: font = ImageFont.load_default()
    tw, th = _get_text_size(draw, title, font)
    draw.text(((canvas.width - tw)//2, (title_h - th)//2), title, fill="black", font=font)

    # Slot-Positionen (3 oben, 2 unten linksbündig)
    top_y = title_h
    bot_y = title_h + slot_h + row_gap_px
    top_xs = [0, slot_w + col_gap_px, 2*slot_w + 2*col_gap_px]
    bot_xs = [0, slot_w + col_gap_px, 2*slot_w + 2*col_gap_px]  # dritter unten bleibt leer

    placements = {
        "Exact":   (top_xs[0], top_y),
        "Copula":  (top_xs[1], top_y),
        "Gaussian":(top_xs[2], top_y),
        "Kernel":  (bot_xs[0], bot_y),
        "Sample":  (bot_xs[1], bot_y),
    }
    img_by_name = dict(zip(order, imgs))

    # Bilder STRETCHEN auf Slotgröße (keine Zusatzlabels)
    for name, (x, y) in placements.items():
        im = img_by_name[name]
        im_resized = im.resize((slot_w, slot_h), Image.LANCZOS)
        canvas.paste(im_resized, (x, y))

    if out_name is None:
        out_name = (f"Summary_{model_name}_{letter}_{int(runtime_sec)}_ALL_3x2_narrow.png"
                    if runtime_sec is not None else
                    f"Summary_{model_name}_{letter}_ALL_3x2_narrow.png")
    out_path = Path(folder)/out_name
    canvas.save(out_path, dpi=(300,300))
    print("✓ panel saved:", out_path)

if __name__ == "__main__":
    make_panel_grid_narrow(
        folder="SummaryPlots",
        model_name="XR",
        data_name_long="Wine",
        runtime_sec=300,
        plot_area_height_px=900,  # kompakter
        slot_aspect=1.00,         # noch schmaler? 0.95–1.05 probieren
        row_gap_px=0,            # << noch kleiner, falls gewünscht
        col_gap_px=10,
        global_title_font_size=34,
        bg="white"
    )
