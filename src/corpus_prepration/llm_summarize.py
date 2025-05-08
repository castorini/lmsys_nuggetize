import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


def get_openai_api_key() -> Optional[str]:
    load_dotenv(dotenv_path=".env")
    return os.getenv("OPEN_AI_API_KEY") or os.getenv("OPENAI_API_KEY")


def get_paths(index: int, prefix: str):
    text_path = os.path.join(prefix, "scraped_texts", f"file_{index}.txt")
    summary_path = os.path.join(prefix, "summary_texts", f"file_{index}.txt")
    return text_path, summary_path


def process_url(i_url, path_prefix: str):
    i, url = i_url
    url = url.strip()
    text_path, summary_path = get_paths(i, path_prefix)

    if not os.path.isfile(text_path):
        return None
    if os.path.isfile(summary_path):
        return url, summary_path

    try:
        with open(text_path, "r") as f:
            text_scrape = f.read().strip()
        if len(text_scrape) < 200:
            return None

        client = OpenAI(api_key=get_openai_api_key())
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": f"""
Summarize the contents of following Webpage Scrape in its original language.
Be sure to keep all the important details.
DON'T include JavaScript code in your summary.

# Webpage Scrape:
{text_scrape[:40000]}
                """,
                }
            ],
        )
        summary_text = completion.choices[0].message.content
        if summary_text.startswith("I'm sorry,"):
            return None

    except Exception as e:
        print(f"[Url {i}] Processing failed: {e}")
        return None

    try:
        with open(summary_path, "w") as f:
            f.write(summary_text + "\n")
        return url, summary_path
    except Exception as e:
        print(f"[Url {i}] Failed to write summary: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Summarize web scrapes with OpenAI.")
    parser.add_argument(
        "--path_prefix",
        type=str,
        required=True,
        help="Base path where 'scraped_texts', 'summary_texts', and 'urls.txt' are stored.",
    )
    args = parser.parse_args()
    path_prefix = args.path_prefix

    from multiprocessing import set_start_method

    try:
        set_start_method("fork")
    except RuntimeError:
        pass

    urls_file = os.path.join(path_prefix, "urls.txt")
    out_json = os.path.join(path_prefix, "urls_to_summary_files.json")

    with open(urls_file, "r") as f:
        urls = sorted([line for line in f])

    urls_to_summary_files = {}
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(process_url, (i, url), path_prefix)
            for i, url in enumerate(urls)
        ]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                url, summary_file = result
                urls_to_summary_files[url] = summary_file

    with open(out_json, "w") as f:
        json.dump(urls_to_summary_files, f, indent=2)


if __name__ == "__main__":
    main()
