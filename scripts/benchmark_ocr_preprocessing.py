import csv
import time
from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image, ImageEnhance, ImageOps

PDF = Path("data/raw/ocr_test/bo_1955_original.pdf")
PAGES = [1, 2, 3, 41, 42, 43, 44, 45, 55]

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

OUT = Path("data/reports")
OUT.mkdir(parents=True, exist_ok=True)


def preprocess(image):
    """Improve contrast and sharpness without aggressive filtering."""
    image = image.convert("L")
    image = ImageOps.autocontrast(image, cutoff=1)
    image = ImageEnhance.Contrast(image).enhance(1.35)
    image = ImageEnhance.Sharpness(image).enhance(1.5)
    return image


CONFIGS = [
    ("A_original_PSM3", lambda im: im, "--oem 1 --psm 3"),
    ("B_enhanced_PSM3", preprocess, "--oem 1 --psm 3"),
    ("C_enhanced_PSM4", preprocess, "--oem 1 --psm 4"),
]

if not PDF.exists():
    raise FileNotFoundError(PDF.resolve())

if "fra" not in pytesseract.get_languages(config=""):
    raise RuntimeError("Le modèle français fra est introuvable.")

results = []
suspects = []
all_text = []

print("BENCHMARK COMPARATIF OCR")
print("Document :", PDF.name)
print("Pages :", PAGES)
print("Configurations :", len(CONFIGS))
print()

with pymupdf.open(PDF) as pdf:

    for page_number in PAGES:

        print(f"\n========== PAGE {page_number} ==========")

        page = pdf[page_number - 1]
        pix = page.get_pixmap(dpi=300, alpha=False)

        original = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        for name, transform, config in CONFIGS:

            image = transform(original.copy())

            start = time.perf_counter()

            data = pytesseract.image_to_data(
                image, lang="fra", config=config, output_type=pytesseract.Output.DICT, timeout=180
            )

            elapsed = time.perf_counter() - start

            tokens = []
            lines = {}

            for i, word in enumerate(data["text"]):

                word = word.strip()

                try:
                    confidence = float(data["conf"][i])
                except (ValueError, TypeError):
                    continue

                if not word or confidence < 0:
                    continue

                tokens.append((word, confidence))

                key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])

                lines.setdefault(key, []).append(word)

                if confidence < 60:
                    suspects.append(
                        {
                            "configuration": name,
                            "page": page_number,
                            "word": word,
                            "confidence": round(confidence, 2),
                        }
                    )

            total = len(tokens)

            if total == 0:
                print(name, ": aucun mot reconnu")
                continue

            average = sum(c for _, c in tokens) / total
            low60 = sum(c < 60 for _, c in tokens)
            low40 = sum(c < 40 for _, c in tokens)

            text = "\n".join(" ".join(words) for words in lines.values())

            results.append(
                {
                    "configuration": name,
                    "page": page_number,
                    "words": total,
                    "average_confidence": round(average, 2),
                    "below_60_count": low60,
                    "below_60_percent": round(low60 / total * 100, 2),
                    "below_40_count": low40,
                    "below_40_percent": round(low40 / total * 100, 2),
                    "elapsed_seconds": round(elapsed, 2),
                }
            )

            all_text.append(
                f"\n\n{'=' * 60}\n"
                f"CONFIGURATION: {name} | PAGE: {page_number}\n"
                f"{'=' * 60}\n{text}"
            )

            print(
                f"{name}: {average:.2f}% confiance | "
                f"{low60 / total * 100:.2f}% faible | "
                f"{elapsed:.2f}s"
            )

# Rapport détaillé
detail_file = OUT / "ocr_comparison_1955.csv"

with detail_file.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

# Rapport comparatif par configuration
summary = []

for name, _, _ in CONFIGS:

    rows = [r for r in results if r["configuration"] == name]
    words = sum(r["words"] for r in rows)

    if not words:
        continue

    summary.append(
        {
            "configuration": name,
            "pages": len(rows),
            "total_words": words,
            "average_confidence": round(
                sum(r["average_confidence"] * r["words"] for r in rows) / words, 2
            ),
            "below_60_percent": round(sum(r["below_60_count"] for r in rows) / words * 100, 2),
            "total_seconds": round(sum(r["elapsed_seconds"] for r in rows), 2),
        }
    )

summary_file = OUT / "ocr_comparison_summary_1955.csv"

with summary_file.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    writer.writeheader()
    writer.writerows(summary)

# Mots suspects
suspect_file = OUT / "ocr_comparison_suspects_1955.csv"

with suspect_file.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=["configuration", "page", "word", "confidence"])
    writer.writeheader()
    writer.writerows(suspects)

# Textes reconnus
text_file = OUT / "ocr_comparison_text_1955.txt"
text_file.write_text("\n".join(all_text), encoding="utf-8")

print("\n========== COMPARAISON FINALE ==========")

for row in summary:
    print(row)

print("\nRapport détaillé :", detail_file.resolve())
print("Résumé :", summary_file.resolve())
print("Mots suspects :", suspect_file.resolve())
print("Textes OCR :", text_file.resolve())
print("\nBENCHMARK TERMINE.")
