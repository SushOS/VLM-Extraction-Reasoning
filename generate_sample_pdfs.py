from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PDF_SAMPLES = [
    {
        "name": "sample_invoice_alpha",
        "title": "Alpha Office Supplies",
        "doc_type": "Invoice",
        "date": "2026-04-12",
        "ref": "INV-24012",
        "rows": [("Printer Paper", "4", "$12.00"), ("Toner Cartridge", "1", "$85.00")],
        "total": "$133.00",
        "signature": True,
        "form_fields": [],
    },
    {
        "name": "sample_form_vendor",
        "title": "Vendor Registration Form",
        "doc_type": "Form",
        "date": "2026-04-15",
        "ref": "FORM-991",
        "rows": [("Company Name", "Northwind Traders", ""), ("Tax ID", "47-110291", ""), ("Bank Account", "", "")],
        "total": "",
        "signature": False,
        "form_fields": [
            {"field": "Company Name", "status": "filled", "value": "Northwind Traders"},
            {"field": "Tax ID", "status": "filled", "value": "47-110291"},
            {"field": "Bank Account", "status": "empty", "value": None},
        ],
    },
    {
        "name": "sample_receipt_cafe",
        "title": "Lakeview Cafe",
        "doc_type": "Receipt",
        "date": "2026-04-18",
        "ref": "RCPT-1138",
        "rows": [("Latte", "2", "$9.00"), ("Croissant", "1", "$4.50"), ("Tax", "", "$1.08")],
        "total": "$14.58",
        "signature": False,
        "form_fields": [],
    },
    {
        "name": "sample_agreement_service",
        "title": "Service Agreement",
        "doc_type": "Agreement",
        "date": "2026-04-19",
        "ref": "AGR-221",
        "rows": [("Provider", "BrightFix Solutions", ""), ("Client", "Mosaic Retail", ""), ("Monthly Fee", "", "$500.00")],
        "total": "$500.00",
        "signature": True,
        "form_fields": [],
    },
]


def build_key_value_pairs(sample: dict) -> list[dict[str, str]]:
    pairs = [
        {"field": "document_type", "value": sample["doc_type"]},
        {"field": "vendor_name", "value": sample["title"]},
        {"field": "date", "value": sample["date"]},
        {"field": "reference", "value": sample["ref"]},
    ]
    if sample["total"]:
        pairs.append({"field": "total_amount", "value": sample["total"]})
    return pairs


def build_line_items(sample: dict) -> list[dict[str, str | None]]:
    return [
        {"name": name, "quantity": qty or None, "price": amount or None}
        for name, qty, amount in sample["rows"]
    ]


def build_ground_truth(sample: dict) -> dict:
    return {
        "document_type": sample["doc_type"].lower(),
        "vendor_name": sample["title"],
        "date": sample["date"],
        "total_amount": sample["total"] or None,
        "currency": "USD" if sample["total"] else None,
        "key_value_pairs": build_key_value_pairs(sample),
        "line_items": build_line_items(sample),
        "form_fields": sample["form_fields"],
        "signature_present": "yes" if sample["signature"] else "no",
        "summary": f"{sample['doc_type']} for {sample['title']}",
    }


def try_import_reportlab():
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas
    except ImportError:
        return None
    return {
        "colors": colors,
        "A4": A4,
        "mm": mm,
        "canvas": canvas,
    }


def draw_signature_reportlab(pdf, x: float, y: float) -> None:
    pdf.setLineWidth(1.4)
    pdf.bezier(x, y, x + 15, y + 12, x + 40, y - 8, x + 60, y + 10)
    pdf.bezier(x + 60, y + 10, x + 75, y + 22, x + 95, y - 4, x + 120, y + 8)


def generate_pdf_with_reportlab(sample: dict, output_dir: Path, modules: dict) -> None:
    colors = modules["colors"]
    A4 = modules["A4"]
    mm = modules["mm"]
    canvas = modules["canvas"]
    path = output_dir / f"{sample['name']}.pdf"
    pdf = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4

    pdf.setTitle(sample["title"])
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(22 * mm, height - 28 * mm, sample["title"])

    pdf.setFont("Helvetica", 11)
    pdf.drawString(22 * mm, height - 38 * mm, f"Document Type: {sample['doc_type']}")
    pdf.drawString(22 * mm, height - 45 * mm, f"Date: {sample['date']}")
    pdf.drawString(22 * mm, height - 52 * mm, f"Reference: {sample['ref']}")

    top = height - 70 * mm
    pdf.setStrokeColor(colors.black)
    pdf.rect(20 * mm, top - 65 * mm, 170 * mm, 65 * mm)

    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(24 * mm, top - 8 * mm, "Field")
    pdf.drawString(94 * mm, top - 8 * mm, "Value / Qty")
    pdf.drawString(150 * mm, top - 8 * mm, "Amount")

    pdf.setFont("Helvetica", 11)
    y = top - 18 * mm
    for name, value, amount in sample["rows"]:
        pdf.drawString(24 * mm, y, name)
        pdf.drawString(94 * mm, y, value)
        pdf.drawString(150 * mm, y, amount)
        y -= 10 * mm

    if sample["total"]:
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(120 * mm, top - 76 * mm, f"Total: {sample['total']}")

    pdf.setFont("Helvetica", 10)
    pdf.drawString(22 * mm, 34 * mm, "Generated as a realistic PDF sample for testing PDF ingestion only.")
    if sample["signature"]:
        pdf.drawString(135 * mm, 42 * mm, "Authorized Signature")
        draw_signature_reportlab(pdf, 135 * mm, 36 * mm)

    pdf.showPage()
    pdf.save()


def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
    ]:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def draw_signature_pillow(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    draw.arc((x, y, x + 120, y + 40), start=12, end=170, fill="black", width=3)
    draw.arc((x + 60, y - 8, x + 180, y + 32), start=200, end=350, fill="black", width=3)


def generate_pdf_with_pillow(sample: dict, output_dir: Path) -> None:
    path = output_dir / f"{sample['name']}.pdf"
    image = Image.new("RGB", (1654, 2339), "white")
    draw = ImageDraw.Draw(image)
    title_font = get_font(40)
    body_font = get_font(24)
    header_font = get_font(26)

    draw.text((120, 120), sample["title"], fill="black", font=title_font)
    draw.text((120, 200), f"Document Type: {sample['doc_type']}", fill="black", font=body_font)
    draw.text((120, 245), f"Date: {sample['date']}", fill="black", font=body_font)
    draw.text((120, 290), f"Reference: {sample['ref']}", fill="black", font=body_font)

    draw.rectangle((110, 380, 1510, 900), outline="black", width=3)
    draw.text((140, 410), "Field", fill="black", font=header_font)
    draw.text((760, 410), "Value / Qty", fill="black", font=header_font)
    draw.text((1230, 410), "Amount", fill="black", font=header_font)

    y = 490
    for name, value, amount in sample["rows"]:
        draw.text((140, y), name, fill="black", font=body_font)
        draw.text((760, y), value, fill="black", font=body_font)
        draw.text((1230, y), amount, fill="black", font=body_font)
        y += 90

    if sample["total"]:
        draw.text((1160, 960), f"Total: {sample['total']}", fill="black", font=header_font)

    draw.text(
        (120, 2180),
        "Generated as a realistic PDF sample for testing PDF ingestion only.",
        fill="black",
        font=body_font,
    )
    if sample["signature"]:
        draw.text((1100, 2080), "Authorized Signature", fill="black", font=body_font)
        draw_signature_pillow(draw, 1080, 2130)

    image.save(path, "PDF", resolution=150.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/pdf_samples")
    parser.add_argument("--metadata-dir", default="results/pdf_samples_metadata")
    parser.add_argument("--ground-truth-dir", default="data/pdf_ground_truth")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    metadata_dir = Path(args.metadata_dir)
    ground_truth_dir = Path(args.ground_truth_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    ground_truth_dir.mkdir(parents=True, exist_ok=True)
    reportlab_modules = try_import_reportlab()

    for sample in PDF_SAMPLES:
        if reportlab_modules:
            generate_pdf_with_reportlab(sample, output_dir, reportlab_modules)
        else:
            generate_pdf_with_pillow(sample, output_dir)
        (metadata_dir / f"{sample['name']}.json").write_text(json.dumps(sample, indent=2))
        (ground_truth_dir / f"{sample['name']}_page_1.json").write_text(
            json.dumps(build_ground_truth(sample), indent=2)
        )


if __name__ == "__main__":
    main()
