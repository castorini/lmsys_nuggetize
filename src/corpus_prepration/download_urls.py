import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import set_start_method

import requests
from datasets import load_dataset
from tqdm import tqdm

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:98.0) Gecko/20100101 Firefox/98.0"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}


def aggregate_urls(output_path):
    urls = set()
    data = load_dataset("lmarena-ai/search-arena-v1-7k", split="test")
    data_df = data.to_pandas()
    for index, row in tqdm(data_df.iterrows(), total=data_df.shape[0]):
        assert index == row["question_id"]
        if row["turn"] != 1:
            continue
        for key in ["a", "b"]:
            web_search_traces = row[f"system_{key}_metadata"]["web_search_trace"]
            for web_search_trace in web_search_traces:
                search_results = web_search_trace["search_results"]
                if search_results is None:
                    continue
                for result in search_results:
                    urls.add(result["url"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for url in sorted(urls):
            f.write(f"{url}\n")


def download_url(i_url, html_dir):
    i, url = i_url
    url = url.strip()
    file_path = os.path.join(html_dir, f"file_{i}.json")

    if os.path.isfile(file_path):
        return ("skipped", url)

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            with open(file_path, "w") as f:
                json.dump({"url": url, "content": response.text}, f)
                f.write("\n")
            return ("success", url)
        else:
            return ("status_not_ok", url)
    except Exception:
        return ("failed", url)


def main(path_prefix):
    urls_file = os.path.join(path_prefix, "urls.txt")
    html_dir = os.path.join(path_prefix, "htmls")
    log_file = os.path.join(path_prefix, "failed_scrapes.json")

    print("Aggregating URLs...")
    aggregate_urls(urls_file)

    with open(urls_file, "r") as f:
        urls = [line.strip() for line in f if line.strip()]
    urls = sorted(urls)

    try:
        set_start_method("fork")
    except RuntimeError:
        pass

    print("Starting downloads...")
    failed_urls = []
    status_not_ok = []

    os.makedirs(html_dir, exist_ok=True)
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(download_url, (i, url), html_dir)
            for i, url in enumerate(urls)
        ]
        for future in tqdm(as_completed(futures), total=len(futures)):
            status, url = future.result()
            if status == "failed":
                failed_urls.append(url)
            elif status == "status_not_ok":
                status_not_ok.append(url)

    print(f"Writing log to {log_file}")
    with open(log_file, "w") as f:
        json.dump({"failed_urls": failed_urls, "status_not_ok": status_not_ok}, f)
        f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_prefix", type=str, required=True, help="Path prefix for saving outputs"
    )
    args = parser.parse_args()

    main(args.path_prefix)
