import argparse
import dataclasses
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from tqdm import tqdm
from datasets import load_dataset

from nuggetizer.core.metrics import calculate_nugget_scores
from nuggetizer.core.types import Document, Query, Request
from nuggetizer.models.nuggetizer import Nuggetizer

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--sampling_rate", type=float, default=1, help="Sampling rate for prompt groups")
parser.add_argument("--path_prefix", type=str, required=True, help="Output path prefix")
parser.add_argument("--url_to_txt_filepath", type=str, required=True, help="path to the file containing a mapping from urls to the names of the files containing the url scrape")
parser.add_argument("--max_workers", type=int, default=4, help="Number of parallel workers")
parser.add_argument("--model_name", type=str, default="gpt-4.1", help="The model name")
args = parser.parse_args()

SAMPLING_RATE = args.sampling_rate
PATH_PREFIX = args.path_prefix
MODEL_NAME = args.model_name


def get_completion(row, key):
    message = row[f"messages_{key}"][1]
    assert message["role"] == "assistant", f"expected 'assistant'; got: {message['role']}"
    return message["content"]

def get_prompt(row):
    message = row["messages_a"][0]
    assert message["role"] == "user"
    prompt = message["content"]
    message = row["messages_b"][0]
    assert message["role"] == "user"
    assert prompt == message["content"], "both LLMs should get the same prompt"
    return prompt

def get_urls(row):
    urls = set()
    for key in ["a", "b"]:
        web_search_traces = row[f"system_{key}_metadata"]["web_search_trace"]
        for web_search_trace in web_search_traces:
            search_results = web_search_trace["search_results"]
            if search_results is None:
                continue
            for result in search_results:
                urls.add(result["url"])
    return urls


def process_prompt_group(args_tuple):
    prompt, qids, urls, urls_to_text_files, data_df = args_tuple
    qid_to_skipped_urls = defaultdict(list)
    qid =  "-".join([str(qid) for qid in qids])
    query = Query(qid=qid, text=prompt)
    nuggetizer = Nuggetizer(model=MODEL_NAME, window_size=20, max_nuggets=60)
    try:
        if random.random() > SAMPLING_RATE:
            return {"skipped": {"sampling": list(qids)}}
        random.shuffle(urls)
        documents = []
        for url in urls:
            if url not in urls_to_text_files:
                qid_to_skipped_urls[qid].append(url)
                continue
            with open(urls_to_text_files[url], "r") as f:
                segment = f.read()
            documents.append(Document(docid=url, segment=segment[:32000]))
        if not documents:
            return {"skipped": {"zero_grounding": list(qids)}}

        request = Request(query=query, documents=documents)
        scored_nuggets = nuggetizer.create(request)
        if not scored_nuggets:
            raise ValueError("No nuggets created")

        os.makedirs(f"{PATH_PREFIX}/nuggets", exist_ok=True)
        with open(f"{PATH_PREFIX}/nuggets/requests_with_nuggets_{qid}.json", "w") as f:
            json.dump({
                "prompt": prompt,
                "request": dataclasses.asdict(request),
                "scored_nuggets": [dataclasses.asdict(sn) for sn in scored_nuggets],
            }, f)
            f.write("\n")

    except Exception as e:
        print(f"[{qid}] Nugget creation failed: {e}", flush=True)
        return {"skipped": {"nugget_creation": list(qids)}}

    results = []
    assignment_skipped_qids = []
    for qid in qids:
        try:
            #   
            row = data_df.iloc[qid] 
            assigned_nuggets = {}
            metrics = {}
            completions = {}
            for key in ["a", "b"]:
                completions[key] = get_completion(row, key)
                assigned_nuggets[key] = nuggetizer.assign(prompt, completions[key], scored_nuggets)
                nugget_list = [
                    {"text": n.text, "importance": n.importance, "assignment": n.assignment}
                    for n in assigned_nuggets[key]
                ]
                metrics[key] = calculate_nugget_scores(query.qid, nugget_list)

            os.makedirs(f"{PATH_PREFIX}/assignments", exist_ok=True)
            with open(f"{PATH_PREFIX}/assignments/assigned_nuggets_{qid}.json", "w") as f:
                result = {"question_id": qid, "winner": row["winner"]}
                for key in ["a", "b"]:
                    result[f"completion_{key}"] = completions[key]
                    result[f"assigned_nuggets_{key}"] = [
                        dataclasses.asdict(an) for an in assigned_nuggets[key]
                    ]
                    result[f"metrics_{key}"] = metrics[key].__dict__
                json.dump(result, f)
                f.write("\n")
            results.append({
                "question_id": qid,
                "winner": row["winner"],
                "metrics_a": metrics["a"].__dict__,
                "metrics_b": metrics["b"].__dict__,
            })
        except Exception as e:
            print(f"[{qid}] Assignment failed for qid {qid}: {e}",flush=True)
            assignment_skipped_qids.append(qid)

    output = {"skipped": {}}
    if results:
        output["results"] = results
    output["skipped"]["nugget_assignment"] = assignment_skipped_qids
    
    if qid_to_skipped_urls:
        output["qid_to_skipped_urls"] = qid_to_skipped_urls
    return output


def create_and_assign_nuggets_parallel():
    data = load_dataset("lmarena-ai/search-arena-v1-7k", split="test")
    data_df = data.to_pandas()
    prompt_to_urls = defaultdict(set)
    prompt_to_qids = defaultdict(list)
    skip_log = defaultdict(list)
    for index, row in tqdm(data_df.iterrows(), total=data_df.shape[0]):
        assert index == row["question_id"]
        if row["turn"] != 1:
            skip_log["multi_turn"].append(index)
            continue
        assert row["turn"] == 1, "Need to deal with multiturn later"
        prompt = get_prompt(row)
        prompt_to_qids[prompt].append(index)
        # For now only handle the urls, others are none anyways.
        for url in get_urls(row):
            prompt_to_urls[prompt].add(url)
    print(f"skipped {len(skip_log['multi_turn'])} multi_turn questions", flush=True)
    urls_to_text_files = json.load(open(args.url_to_txt_filepath))

    args_list = []
    for prompt in prompt_to_qids:
        args_list.append((prompt, prompt_to_qids[prompt],list(prompt_to_urls[prompt]), urls_to_text_files, data_df))

    all_results = []

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_prompt_group, args) for args in args_list[:100]]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if not result:
                continue
            for k, v in result["skipped"].items():
                skip_log[k].extend(v)
            if "results" in result:
                all_results.extend(result["results"])
            if "qid_to_skipped_urls" in result:
                skip_log["qid_to_skipped_urls"].extend([(k,v) for k,v in result["qid_to_skipped_urls"].items()])

    with open(f"{PATH_PREFIX}/results.jsonl", "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    with open(f"{PATH_PREFIX}/skips.json", "w") as f:
        json.dump(skip_log, f, indent=2)


if __name__ == "__main__":
    create_and_assign_nuggets_parallel()