import argparse
import json
import multiprocessing
import numpy as np
import os
from ftplib import FTP
from pathlib import Path
from urllib.parse import urlparse
from firecrawl import FirecrawlApp
from dotenv import load_dotenv

import requests
import urllib3
from datasets import load_dataset
from tqdm import tqdm

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from playwright.sync_api import sync_playwright

load_dotenv()
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")


def aggregate_urls(dataset, skip_multi_turn, skip_no_vote, output_path):
    urls = set()
    data = load_dataset(dataset, split="test")
    data_df = data.to_pandas()
    qids_with_urls = set()
    for index, row in tqdm(data_df.iterrows(), total=data_df.shape[0]):
        if skip_multi_turn and row["turn"] != 1:
            continue
        if skip_no_vote and not row["winner"]:
            continue
        for key in ["a", "b"]:
            web_search_traces = row[f"system_{key}_metadata"]["web_search_trace"]
            for web_search_trace in web_search_traces:
                if isinstance(web_search_trace, np.ndarray):  # 24k
                    for result in web_search_trace:
                        urls.add(result[1])
                        qids_with_urls.add(index)
                else:  # v1-7k
                    search_results = web_search_trace["search_results"]
                    if search_results is None:
                        continue
                    for result in search_results:
                        urls.add(result["url"])
                        qids_with_urls.add(index)
    print(len(qids_with_urls))
    print(len(urls))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for url in sorted(urls):
            f.write(f"{url}\n")


def load_mapping(path_prefix, use_firecrawl):
    file_name = (
        "urls_to_downloaded_filesnames.json"
        if not use_firecrawl
        else "firecrawl_urls_to_downloaded_filesnames.json"
    )
    mapping_path = os.path.join(path_prefix, file_name)
    if os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_mapping(mapping, path_prefix, use_firecrawl):
    file_name = (
        "urls_to_downloaded_filesnames.json"
        if not use_firecrawl
        else "firecrawl_urls_to_downloaded_filesnames.json"
    )
    mapping_path = os.path.join(path_prefix, file_name)
    os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
    with open(mapping_path, "w", encoding="utf-8") as f:
        output_str = json.dumps(mapping, indent=2, ensure_ascii=False)
        f.write(output_str)
        f.write("\n")


def get_session():
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/pdf,application/xhtml+xml,"
            "application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,*;q=0.8",
            # "Referer": "https://www.google.com"
        }
    )
    return session


def download_with_headless_browser(url, path_prefix):
    save_dir = os.path.join(path_prefix, "downloaded_files")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url, timeout=30000)
        content = page.content()
        browser.close()
    ext = ".html"
    base_name = base_name[:200]
    path, filename = find_unique_name(base_name, ext, save_dir)
    with open(path, "wb") as f:
        f.write(content)

    return (url, filename)


def find_unique_name(base_name, ext, save_dir):
    base_name = base_name[:200]
    filename = f"{base_name}{ext}"
    path = Path(save_dir) / filename
    i = 1
    while path.exists():
        filename = f"{base_name}_{i}{ext}"
        path = Path(save_dir) / filename
        i += 1
    return path, filename


def download_and_store_with_firecrawl(url, path_prefix):
    save_dir = os.path.join(path_prefix, "firecrawl_downloaded_files")
    os.makedirs(save_dir, exist_ok=True)
    try:
        parsed_url = urlparse(url)
        ftp_path = parsed_url.path
        base_name = Path(ftp_path).name or "file"
        ext = ".json"
        path, filename = find_unique_name(base_name, ext, save_dir)
        app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
        content = app.scrape_url(url, formats=["markdown", "html"])
        with open(path, "w", encoding="utf-8") as f:
            output_str = json.dumps(content, ensure_ascii=False)
            f.write(output_str)
            f.write("\n")

        return (url, filename)

    except Exception as e:
        print(f"[FireCrawl Error] Failed to download {url}: {e}")
        return (url, None)


def download_and_store(url, path_prefix):
    save_dir = os.path.join(path_prefix, "downloaded_files")
    os.makedirs(save_dir, exist_ok=True)
    parsed_url = urlparse(url)
    if parsed_url.scheme == "ftp":
        try:
            ftp = FTP(parsed_url.hostname)
            ftp.login()  # anonymous login

            ftp_path = parsed_url.path
            base_name = Path(ftp_path).name or "file"
            ext = Path(ftp_path).suffix or ""
            path, filename = find_unique_name(base_name, ext, save_dir)
            with open(path, "wb") as f:
                ftp.retrbinary(f"RETR {ftp_path}", f.write)

            ftp.quit()
            return (url, filename)

        except Exception as e:
            print(f"[FTP Error] Failed to download {url}: {e}")
            return (url, None)
    try:
        response = get_session().get(
            url, timeout=10, allow_redirects=True, verify=False
        )
        content_type = response.headers.get("Content-Type", "")
        base_name = Path(parsed_url.path).stem or "page"

        if "pdf" in content_type or url.lower().endswith(".pdf"):
            ext = ".pdf"
        elif "xml" in content_type or url.lower().endswith(".xml"):
            ext = ".xml"
        else:
            ext = ".html"

        path, filename = find_unique_name(base_name, ext, save_dir)
        with open(path, "wb") as f:
            f.write(response.content)

        return (url, filename)

    except Exception as e:
        try:
            return download_with_headless_browser(url, path_prefix)
        except:
            print(f"[Error] Failed to download {url}: {e}")
            return (url, None)


def main():
    parser = argparse.ArgumentParser(description="Download and process URLs.")
    parser.add_argument(
        "--path_prefix", required=True, help="Directory to save downloaded files"
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of worker processes to use"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="The search arena dataset name"
    )
    parser.add_argument(
        "--skip_multi_turn", action="store_true", help="skips multi-turn queries"
    )
    parser.add_argument(
        "--skip_no_vote", action="store_true", help="skips queries with no human vote"
    )
    parser.add_argument(
        "--use_firecrawl",
        action="store_true",
        help="Uses Firecrawl.dev for downloading urls",
    )
    args = parser.parse_args()

    urls_file = os.path.join(args.path_prefix, "urls.txt")
    print("Aggregating URLs...")
    aggregate_urls(args.dataset, args.skip_multi_turn, args.skip_no_vote, urls_file)
    with open(urls_file, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]
    urls = sorted(urls)

    existing_mapping = load_mapping(args.path_prefix, args.use_firecrawl)
    urls_to_download = [url for url in urls if url not in existing_mapping][:10]
    print(f"Found {len(urls_to_download)} new URLs to download...")

    with multiprocessing.Pool(args.workers) as pool:
        if args.use_firecrawl:
            results = pool.starmap(
                download_and_store_with_firecrawl,
                [(url, args.path_prefix) for url in urls_to_download],
            )
        else:
            results = pool.starmap(
                download_and_store,
                [(url, args.path_prefix) for url in urls_to_download],
            )

    new_mapping = {url: filename for url, filename in results if url and filename}
    merged_mapping = {**existing_mapping, **new_mapping}
    save_mapping(merged_mapping, args.path_prefix, args.use_firecrawl)

    for url, filename in new_mapping.items():
        print(f"[Downloaded] {url} -> {filename}")


if __name__ == "__main__":
    main()
