import argparse
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import html2text
from bs4 import BeautifulSoup
from bs4.builder import XMLParsedAsHTMLWarning
from tqdm import tqdm

warnings.filterwarnings("error", category=XMLParsedAsHTMLWarning)


def process_url(i_url, html_dir, text_dir):
    i, url = i_url
    url = url.strip()
    json_file = os.path.join(html_dir, f"file_{i}.json")
    if not os.path.isfile(json_file):
        return None
    text_file = os.path.join(text_dir, f"file_{i}.txt")
    if os.path.isfile(text_file):
        return (url, text_file)

    try:
        with open(json_file, "r") as f:
            data = json.load(f)
        assert data["url"] == url, f"{data['url']} vs. {url}"
        html_content = data["content"]
        soup = BeautifulSoup(html_content, "lxml")
        for s in soup.select("script"):
            s.extract()
        body = soup.find("body")
        text_maker = html2text.HTML2Text()
        text_maker.ignore_links = True
        text_maker.ignore_mailto_links = True
        text_maker.ignore_images = True
        scraped_text = text_maker.handle(body.prettify() if body else soup.prettify())
    except Warning:
        soup = BeautifulSoup(html_content, "xml")
        scraped_text = soup.get_text("\n", strip=True)
    except Exception as e:
        print(f"[Url {url}] Failed to process due to: {e}")
        scraped_text = ""

    if scraped_text:
        try:
            os.makedirs(text_dir, exist_ok=True)
            with open(text_file, "w") as f:
                f.write(scraped_text)
                f.write("\n")
            return (url, text_file)
        except Exception as e:
            print(f"[Url {url}] Failed to write file: {e}")
    return None


def main(path_prefix):
    html_dir = os.path.join(path_prefix, "htmls")
    text_dir = os.path.join(path_prefix, "scraped_texts")
    urls_file = os.path.join(path_prefix, "urls.txt")
    output_mapping_file = os.path.join(path_prefix, "urls_to_text_files.json")

    try:
        from multiprocessing import set_start_method

        set_start_method("fork")
    except RuntimeError:
        pass

    with open(urls_file, "r") as f:
        urls = [line.strip() for line in f if line.strip()]
    urls = sorted(urls)

    urls_to_text_files = {}
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(process_url, (i, url), html_dir, text_dir)
            for i, url in enumerate(urls)
        ]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                url, text_file = result
                urls_to_text_files[url] = text_file

    with open(output_mapping_file, "w") as f:
        json.dump(urls_to_text_files, f)
        f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_prefix",
        type=str,
        required=True,
        help="Path prefix for input and output files",
    )
    args = parser.parse_args()
    main(args.path_prefix)
