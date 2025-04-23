import json
from tqdm import tqdm
from pathlib import Path
from bs4 import BeautifulSoup
from concurrent.futures import ProcessPoolExecutor, as_completed
import html2text
from bs4.builder import XMLParsedAsHTMLWarning
import warnings
warnings.filterwarnings("error", category=XMLParsedAsHTMLWarning)


def process_url(i_url):
    i, url = i_url
    url = url.strip()
    json_file = f'/mnt/users/s8sharif/search_arena/htmls/file_{i}.json'
    if not Path(json_file).is_file():
        return None
    text_file = f'/mnt/users/s8sharif/search_arena/scraped_texts/file_{i}.txt'
    if Path(text_file).is_file():
        return None
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        assert data['url'] == url, f"{data['url']} vs. {url}"
        html_content = data['content']
        soup = BeautifulSoup(html_content, 'lxml')
        for s in soup.select('script'):
            s.extract()
        body = soup.find('body')
        text_maker = html2text.HTML2Text()
        text_maker.ignore_links = True
        text_maker.ignore_mailto_links = True
        text_maker.ignore_images = True
        scraped_text = text_maker.handle(body.prettify() if body else soup.prettify())
    except Warning:
        soup = BeautifulSoup(html_content, "xml")
        scraped_text = soup.get_text('\n', strip=True)
    except Exception as e:
        print(f"[Url {i}] Failed to process due to: {e}")
        scraped_text = ""

    if scraped_text:
        try:
            with open(text_file, 'w') as f:
                f.write(scraped_text)
                f.write('\n')
            return (url, text_file)
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
    urls_to_text_files = {}
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_url, (i, url)) for i, url in enumerate(urls)]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                url, text_file = result
                urls_to_text_files[url] = text_file

with open('/mnt/users/s8sharif/search_arena/urls_to_text_files.json', 'w') as f:
    json.dump(urls_to_text_files, f)
    f.write('\n')