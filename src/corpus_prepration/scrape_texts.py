import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from readability import Document


def extract_text_from_pdf(file_path):
    try:
        with fitz.open(file_path) as doc:
            return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        raise RuntimeError(f"PyMuPDF error: {e}")


def extract_text_from_file(file_path):
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_path)

    elif ext in [".html", ".xml"]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()

        doc = Document(html)
        summary = doc.summary()
        soup = BeautifulSoup(summary, "lxml")

        for tag in soup(["script", "style", "img", "nav", "footer", "header", "aside"]):
            tag.decompose()

        return soup.get_text(separator="\n", strip=True)

    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


def _extract_and_write(args):
    i, url, filename, input_dir, output_dir = args
    file_path = input_dir / filename
    out_path = output_dir / f"filename_{i}.txt"

    if not file_path.exists():
        print(f"[Skipped] Missing file: {file_path}")
        return (url, None)

    if out_path.exists():
        return (url, f"filename_{i}.txt")

    try:
        text = extract_text_from_file(file_path)
    except Exception as e:
        print(f"[Error] Failed to extract {file_path.name}: {e}")
        return (url, None)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[Extracted] {file_path.name} -> filename_{i}.txt")
    return (url, f"filename_{i}.txt")


def extract_all_texts(path_prefix, workers=8):
    urls_path = Path(path_prefix) / "urls.txt"
    mapping_path = Path(path_prefix) / "urls_to_downloaded_filesnames.json"
    input_dir = Path(path_prefix) / "downloaded_files"
    output_dir = Path(path_prefix) / "scraped_texts"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not urls_path.exists() or not mapping_path.exists():
        raise FileNotFoundError(
            "Missing urls.txt or urls_to_downloaded_filesnames.json"
        )

    with open(urls_path, "r") as f:
        urls = sorted([line.strip() for line in f if line.strip()])
    urls = sorted(urls)

    with open(mapping_path, "r") as f:
        url_to_filename = json.load(f)

    args_list = []
    for i, url in enumerate(urls):
        filename = url_to_filename.get(url)
        if filename:
            args_list.append((i, url, filename, input_dir, output_dir))
        else:
            print(f"[Skipped] No downloaded file for URL: {url}")

    with Pool(processes=workers) as pool:
        results = pool.map(_extract_and_write, args_list)

    # Save new mapping
    text_mapping = {url: txt_file for url, txt_file in results if txt_file}
    mapping_output_path = Path(path_prefix) / "urls_to_text_files.json"
    with open(mapping_output_path, "w", encoding="utf-8") as f:
        output_str = json.dumps(text_mapping, indent=2, ensure_ascii=False)
        f.write(output_str)
        f.write("\n")

    print(f"\n✅ Saved mapping: {mapping_output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract text from all downloaded files."
    )
    parser.add_argument(
        "--path_prefix", required=True, help="Directory containing downloaded_files"
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of worker processes"
    )
    args = parser.parse_args()

    extract_all_texts(args.path_prefix, workers=args.workers)


if __name__ == "__main__":
    main()
