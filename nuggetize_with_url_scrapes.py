from datasets import load_dataset
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import json
from nuggetizer.core.types import Query, Document, Request
from nuggetizer.models.nuggetizer import Nuggetizer
from nuggetizer.core.metrics import calculate_nugget_scores
from collections import defaultdict
import dataclasses


def get_prompt(row):
    message = row['messages_a'][0]
    assert message['role'] == 'user'
    prompt = message['content']
    message = row['messages_b'][0]
    assert message['role'] == 'user'
    assert prompt == message['content'], "both LLMs should get the same prompt"
    return prompt


def get_urls(row, counts):
    urls = set()
    for key in ['a', 'b']:
        web_search_traces = row[f'system_{key}_metadata']['web_search_trace']
        for web_search_trace in web_search_traces:
            if web_search_trace['context']:
                counts['context_count'] += 1
            if web_search_trace['query']:
                counts['query_count'] += 1
            if web_search_trace['scrape_results']:
                counts['scrape_count'] += 1
            search_results = web_search_trace['search_results']
            if search_results is None:
                continue
            for result in search_results:
                urls.add(result['url'])
    return urls

def get_completion(row, key):
    message = list(row[f'messages_{key}'])[1]
    assert message['role'] == 'assistant', f"expected 'assistant'; got: {message['role']}"
    return message['content']

def create_and_assign_nuggets():
    data = load_dataset("lmarena-ai/search-arena-v1-7k", split="test")
    data_df = data.to_pandas()
    prompt_to_urls= defaultdict(set)
    prompt_to_qids= defaultdict(list)
    multi_turn_skipped_qids = set()
    zero_grounding_skipped_qids = set()
    nugget_creation_exception_skipped_qids = set()
    nugget_assignment_exception_skipped_qids = set()
    qid_to_skipped_urls = defaultdict(list)
    counts = {'context_count': 0, 'query_count':0, 'scrape_count':0}
    for index, row in tqdm(data_df.iterrows(), total=data_df.shape[0]):
        assert index == row['question_id']
        if row['turn'] != 1:
            multi_turn_skipped_qids.add(index)
            continue
        assert row['turn'] == 1, "Need to deal with multiturn later"
        prompt = get_prompt(row)
        prompt_to_qids[prompt].append(index)
        # For now only handle the urls, others are none anyways.
        for url in get_urls(row, counts):
            prompt_to_urls[prompt].add(url)
    print(counts)
    print(f'skipped {len(multi_turn_skipped_qids)} multi_turn questions')
    urls_to_text_files = {}
    nuggetizer = Nuggetizer(model="gpt-4o-mini", window_size=20, max_nuggets=60)
    with open('/mnt/users/s8sharif/search_arena/urls_to_text_files.json', 'r') as f:
        urls_to_text_files = json.load(f)
    with open('/mnt/users/s8sharif/search_arena/with_retrieval_scrape/results.jsonl', 'w') as results_file:
        # Create the nuggets and assign them
        for i, prompt in enumerate(tqdm(prompt_to_qids)):
            qids = prompt_to_qids[prompt]
            request_qid = f'request_{i}_' + '_'.join([str(qid) for qid in qids])
            query =  Query(qid=request_qid, text=prompt)
            urls = list(prompt_to_urls[prompt])
            # Skip prompts with zero grounding urls
            if not urls or not any([url in urls_to_text_files for url in urls]):
                for qid in qids:
                    zero_grounding_skipped_qids.add(qid) 
                continue
            if random.random() > 0.02:
                continue
            try:
                random.shuffle(urls)
                documents = []
                for url in urls:
                    if url not in urls_to_text_files:
                        for qid in qids:
                            qid_to_skipped_urls[qid].append(url)
                        continue
                    with open(urls_to_text_files[url], 'r') as scraped_text_file:
                        segment = scraped_text_file.read()
                    # around 8k token per document
                    if len(segment) > 32000:
                        print(f"Trimmed the segment for url {url} and qids {qids} from {len(segment)} to 32k")
                    document = Document(docid=url, segment=segment[:32000])
                    documents.append(document)
                request = Request(query=query, documents=documents)
                scored_nuggets = nuggetizer.create(request)
                with open(f'/mnt/users/s8sharif/search_arena/with_retrieval_scrape/nuggets/requests_with_nuggets_{i}.json', 'w') as f:
                    json.dump({"prompt": prompt, "request": dataclasses.asdict(request), "scored_nuggets": [dataclasses.asdict(sn) for sn in scored_nuggets]}, f)
                    f.write('\n')
            except Exception as error:
                print(f"failed to nuggetize with error {error}")
                for qid in qids:
                    nugget_creation_exception_skipped_qids.add(qid)
            # assign 
            for qid in tqdm(qids):
                assigned_nuggets = {}
                metrics = {}
                completions = {}
                row = data_df.iloc[qid]
                # Assign nuggets to a specific document
                try:
                    for key in ["a", "b"]:
                        completions[key] = get_completion(row, key)
                        assigned_nuggets[key] = nuggetizer.assign(query.text, completions[key], scored_nuggets)
                        nugget_list = [
                                {
                                    'text': n.text,
                                    'importance': n.importance,
                                    'assignment': n.assignment
                                }
                                for n in assigned_nuggets[key]
                            ]
                        metrics[key] = calculate_nugget_scores(request.query.qid, nugget_list)
                    with open(f'/mnt/users/s8sharif/search_arena/with_retrieval_scrape/assignments/assigned_nuggets_{i}.json', 'w') as f2:
                        result = {'question_id': qid, 'winner': row['winner']}
                        for key in ["a", "b"]:
                            result[f'completion_{key}'] = completions[key]
                            result[f'assigned_nuggets_{key}'] = [dataclasses.asdict(an) for an in assigned_nuggets[key]]
                            result[f'metrics_{key}'] = metrics[key].__dict__
                        json.dump(result, f2)
                        f2.write('\n')
                    json.dump({
                        'question_id': qid,
                        'winner': row['winner'],
                        'metrics_a' : metrics['a'].__dict__,
                        'metrics_b' : metrics['b'].__dict__,
                    }, results_file)
                    results_file.write('\n')
                except Exception as error:
                    print(f"failed to assign nuggets with error {error}")
                    nugget_assignment_exception_skipped_qids.add(qid)
                    continue
    with open('/mnt/users/s8sharif/search_arena/with_retrieval_scrape/skips.json', 'w') as skipped_file:
        json.dump(
            {
                "zero_grounding": list(zero_grounding_skipped_qids),
                "nugget_creation": list(nugget_creation_exception_skipped_qids),
                "nugget_assignment":list(nugget_assignment_exception_skipped_qids),
                "multi_turn": list(multi_turn_skipped_qids),
                "qid_to_skipped_urls": qid_to_skipped_urls,
            },
            skipped_file
        )
        skipped_file.write('\n')


if __name__ == '__main__':
    create_and_assign_nuggets()    
