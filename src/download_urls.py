# import wget
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import requests
from datasets import load_dataset
from tqdm import tqdm


def aggregate_urls():
    urls = set()
    data = load_dataset("lmarena-ai/search-arena-v1-7k", split="test")
    data_df = data.to_pandas()
    for index, row in tqdm(data_df.iterrows(), total=data_df.shape[0]):
        assert index == row["question_id"]
        if row["turn"] != 1:
            continue
        # For now only handle the urls, others are none anyways.
        for key in ["a", "b"]:
            web_search_traces = row[f"system_{key}_metadata"]["web_search_trace"]
            for web_search_trace in web_search_traces:
                search_results = web_search_trace["search_results"]
                if search_results is None:
                    continue
                for result in search_results:
                    urls.add(result["url"])

    with open("/mnt/users/s8sharif/search_arena/urls.txt", "w") as f:
        for url in urls:
            f.write(url)
            f.write("\n")


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:98.0) Gecko/20100101 Firefox/98.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
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


def download_url(i_url):
    i, url = i_url
    url = url.rstrip()
    file_path = f"/mnt/users/s8sharif/search_arena/htmls/file_{i}.json"

    if Path(file_path).is_file():
        return ("skipped", url)  # Already exists

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            with open(file_path, "w") as f2:
                json.dump({"url": url, "content": response.text}, f2)
                f2.write("\n")
            return ("success", url)
        else:
            return ("status_not_ok", url)
    except Exception:
        return ("failed", url)


if __name__ == "__main__":
    aggregate_urls()
    urls = []
    with open("/mnt/users/s8sharif/search_arena/urls.txt", "r") as f1:
        for line in f1:
            urls.append(line)
    urls = sorted(urls)

    from multiprocessing import set_start_method

    try:
        set_start_method("fork")
    except RuntimeError:
        pass

    failed_urls = []
    status_not_ok = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(download_url, (i, url)) for i, url in enumerate(urls)
        ]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                status, url = result
                if status == "failed":
                    failed_urls.append(url)
                elif status == "status_not_ok":
                    status_not_ok.append(url)

    # Save error logs
    with open("/mnt/users/s8sharif/search_arena/failed_scrapes.json", "w") as f3:
        json.dump({"failed_urls": failed_urls, "status_not_ok": status_not_ok}, f3)
        f3.write("\n")
