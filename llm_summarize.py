import json
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional
import os

def get_openai_api_key() -> Optional[str]:
    load_dotenv(dotenv_path=".env")
    openai_api_key = os.getenv("OPEN_AI_API_KEY") or os.getenv("OPENAI_API_KEY")
    return openai_api_key

# If you cannot summarize it for any reason, DON'T appologize nor explain; return the word "error" instead.

def process_url(i_url):
    i, url = i_url
    url = url.strip()
    text_file = f'/mnt/users/s8sharif/search_arena/scraped_texts/file_{i}.txt'
    if not Path(text_file).is_file():
        return None
    summary_file = f'/mnt/users/s8sharif/search_arena/summary_texts/file_{i}.txt'
    if Path(summary_file).is_file():
        return None
    try:
        with open(text_file, 'r') as f:
            text_scrape = f.read().strip()
        # Most likely some error.
        if len(text_scrape) < 200:
            return None
        client = OpenAI(api_key=get_openai_api_key())
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": 
f"""
Summarize the contents of following Webpage Scrape in its original language.
Be sure to keep all the inportant details. 
DON'T include javascipt code in your summary.

# Webpage Scrape:
{text_scrape[:40000]}   

"""
                }
            ]
        )
        summary_text = completion.choices[0].message.content
        print(f"\n\n\n\n{i}:")
        print(summary_text)
        if summary_text.startswith("I'm sorry,"):
            return None
    except Exception as e:
        print(f"[Url {i}] Failed to process due to: {e}")
        summary_text = ""

    if summary_text:
        try:
            with open(summary_file, 'w') as f:
                f.write(summary_text)
                f.write('\n')
            return (url, summary_file)
        except Exception as e:
            print(f"[Url {i}] Failed to write file: {e}")
    return None

if __name__ == '__main__':
    from multiprocessing import set_start_method
    try:
        set_start_method("fork")  # safer on Unix-like; fallback needed on Windows
    except RuntimeError:
        pass
    urls = []
    with open('/mnt/users/s8sharif/search_arena/urls.txt', 'r') as f1:
        for line in f1:
            urls.append(line)
    urls = sorted(urls)
    urls_to_summary_files = {}
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_url, (i, url)) for i, url in enumerate(urls)]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                url, summary_file = result
                urls_to_summary_files[url] = summary_file

with open('/mnt/users/s8sharif/search_arena/urls_to_summary_files.json', 'w') as f:
    json.dump(urls_to_summary_files, f)
    f.write('\n')