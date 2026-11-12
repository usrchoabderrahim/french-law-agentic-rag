import csv
import time
from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image

PDF = Path("data/raw/ocr_test/bo_1955_original.pdf")
OUT = Path("data/reports")
PAGES = [42, 43, 45]

CONFIGS = [
    ("A_300_PSM3", 300, 3),
    ("B_400_PSM3", 400, 3),
    ("C_300_PSM11", 300, 11),
    ("D_300_PSM6", 300, 6),
]

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

OUT.mkdir(parents=True, exist_ok=True)

results = []
texts = []

with pymupdf.open(PDF) as pdf:

    for page_number in PAGES:

        page = pdf[page_number - 1]

        print(f"\nPAGE {page_number}")

        for name, dpi, psm in CONFIGS:

            start = time.perf_counter()

            pix = page.get_pixmap(dpi=dpi, alpha=False)

            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

            data = pytesseract.image_to_data(
                image,
                lang="fra",
                config=f"--oem 1 --psm {psm}",
                output_type=pytesseract.Output.DICT,
                timeout=180,
            )

            elapsed = time.perf_counter() - start

            tokens = []
            lines = {}

            for i, word in enumerate(data["text"]):

                word = word.strip()

                try:
                    conf = float(data["conf"][i])
                except (ValueError, TypeError):
                    continue

                if not word or conf < 0:
                    continue

                tokens.append((word, conf))

                key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])

                lines.setdefault(key, []).append(word)

            if not tokens:
                continue

            count = len(tokens)
            average = sum(c for _, c in tokens) / count
            low60 = sum(c < 60 for _, c in tokens)
            low40 = sum(c < 40 for _, c in tokens)

            result = {
                "configuration": name,
                "page": page_number,
                "dpi": dpi,
                "psm": psm,
                "words": count,
                "average_confidence": round(average, 2),
                "below_60_count": low60,
                "below_60_percent": round(low60 / count * 100, 2),
                "below_40_count": low40,
                "below_40_percent": round(low40 / count * 100, 2),
                "elapsed_seconds": round(elapsed, 2),
            }

            results.append(result)

            text = "\n".join(" ".join(words) for words in lines.values())

            texts.append(f"\n\nPAGE {page_number} - {name}\n{text}")

            print(
                f"{name}: confiance={average:.2f}% | "
                f"faible={low60/count*100:.2f}% | "
                f"temps={elapsed:.2f}s"
            )

detail = OUT / "ocr_targeted_detail_1955.csv"
summary = OUT / "ocr_targeted_summary_1955.csv"
text_file = OUT / "ocr_targeted_text_1955.txt"

with detail.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

summary_rows = []

for name, _, _ in CONFIGS:

    rows = [r for r in results if r["configuration"] == name]

    if not rows:
        continue

    total = sum(r["words"] for r in rows)

    summary_rows.append(
        {
            "configuration": name,
            "pages": len(rows),
            "total_words": total,
            "average_confidence": round(
                sum(r["average_confidence"] * r["words"] for r in rows) / total, 2
            ),
            "below_60_percent": round(sum(r["below_60_count"] for r in rows) / total * 100, 2),
            "total_seconds": round(sum(r["elapsed_seconds"] for r in rows), 2),
        }
    )

with summary.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    writer.writeheader()
    writer.writerows(summary_rows)

text_file.write_text("\n".join(texts), encoding="utf-8")

print("\n========== RESULTATS FINAUX ==========")

for row in summary_rows:
    print(row)

print("\nFichiers créés :")
print(detail.resolve())
print(summary.resolve())
print(text_file.resolve())
