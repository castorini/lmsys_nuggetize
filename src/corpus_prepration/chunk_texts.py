import argparse
import json
import os
from multiprocessing import Manager, Pool, Process

import pycountry
import spacy
from langdetect import detect
from tqdm import tqdm

# Initialize spaCy and tokenizer globally
nlp = spacy.load("xx_sent_ud_sm")
nlp.max_length = 2_000_000_000


def sentence_split(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def chunk_sentences(sentences, max_len, overlap):
    chunks = []
    for i in range(0, len(sentences), max_len - overlap):
        chunk = sentences[i : i + max_len]
        chunks.append(" ".join(chunk))
    return chunks


def get_language_name(text):
    try:
        lang_code = detect(text)
        if "-" in lang_code:
            lang_code = lang_code.split("-")[0]
        return pycountry.languages.get(alpha_2=lang_code).name
    except:
        return "Unknown"


def process_one(args):
    index, url, text_path, max_len, overlap = args
    if not os.path.isfile(text_path):
        return []

    try:
        with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
        if not text:
            return []

        language = get_language_name(text)
        sentences = sentence_split(text)
        chunks = chunk_sentences(sentences, max_len, overlap)

        results = []
        for chunk_index, chunk in enumerate(chunks):
            item = {
                "_id": f"{index}_{chunk_index}",
                "metadata": {"url": url, "language": language},
                "text": chunk,
            }
            results.append(json.dumps(item, ensure_ascii=False))
        return results

    except Exception as e:
        print(f"[Error] index {index} ({url}): {e}")
        return []


def writer_worker(queue, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        while True:
            item = queue.get()
            if item is None:
                break
            f.write(item + "\n")


def main(path_prefix, num_workers, max_len, overlap):
    urls_file = os.path.join(path_prefix, "urls.txt")
    text_dir = os.path.join(path_prefix, "scraped_texts")
    output_path = os.path.join(path_prefix, "urls_chunked_corpus.jsonl")

    with open(urls_file, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    args_list = [
        (i, url, os.path.join(text_dir, f"filename_{i}.txt"), max_len, overlap)
        for i, url in enumerate(urls)
    ]

    manager = Manager()
    queue = manager.Queue()
    writer = Process(target=writer_worker, args=(queue, output_path))
    writer.start()

    with Pool(num_workers) as pool:
        for results in tqdm(
            pool.imap_unordered(process_one, args_list), total=len(args_list)
        ):
            for line in results:
                queue.put(line)

    queue.put(None)
    writer.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_prefix",
        type=str,
        required=True,
        help="Path prefix for input/output files",
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of parallel workers"
    )
    parser.add_argument(
        "--max_len", type=int, default=10, help="Maximum number of sentences per chunk"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=2,
        help="Number of overlapping sentences between chunks",
    )
    args = parser.parse_args()
    main(args.path_prefix, args.num_workers, args.max_len, args.overlap)
