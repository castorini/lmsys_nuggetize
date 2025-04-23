import random
import json
import dataclasses
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from datasets import load_dataset
import pandas as pd
from nuggetizer.core.types import Query, Document, Request
from nuggetizer.models.nuggetizer import Nuggetizer
from nuggetizer.core.metrics import calculate_nugget_scores
from tqdm import tqdm

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--sampling_rate", type=float, default=0.005, help="Sampling rate for processing rows")
parser.add_argument("--path_prefix", type=str, default="/mnt/users/s8sharif/search_arena/with_response", help="Output path prefix")
parser.add_argument("--max_workers", type=int, default=4, help="Number of parallel workers")
parser.add_argument("--model_name", type=str, default="gpt-4.1", help="the model name, for now from gpt family only.")
args = parser.parse_args()

# Unpack args
SAMPLING_RATE = args.sampling_rate
PATH_PREFIX = args.path_prefix
MODEL_NAME= args.model_name

def get_prompt(row):
    message = row['messages_a'][0]
    assert message['role'] == 'user'
    prompt = message['content']
    message = row['messages_b'][0]
    assert message['role'] == 'user'
    assert prompt == message['content'], "both LLMs should get the same prompt"
    return prompt

def get_completion(row, key):
    message = row[f'messages_{key}'][1]
    assert message['role'] == 'assistant'
    return message['content']

def process_row(index_row_tuple):
    index, row = index_row_tuple
    if row['turn'] != 1:
        return {'skipped_reason': 'multi_turn', 'question_id': index}
    # Skip ties for now
    if row['winner'] not in ["model_a", "model_b"]:
        return {'skipped_reason': 'sampling', 'question_id': index}
    if random.random() > SAMPLING_RATE:
        return {'skipped_reason': 'sampling', 'question_id': index}

    try:
        query = Query(qid=index, text=get_prompt(row))
        documents = [
            Document(docid="a", segment=get_completion(row, "a")),
            Document(docid="b", segment=get_completion(row, "b")),
        ]
        if random.random() > 0.5:
            documents[0], documents[1] = documents[1], documents[0]
        request = Request(query=query, documents=documents)

        nuggetizer = Nuggetizer(model=MODEL_NAME, use_azure_openai=True)
        scored_nuggets = nuggetizer.create(request)

        with open(f'{PATH_PREFIX}/nuggets/requests_with_nuggets_{index}.json', 'w') as f:
            result_str = json.dumps({
                "question_id": index,
                "request": dataclasses.asdict(request),
                "scored_nuggets": [dataclasses.asdict(sn) for sn in scored_nuggets]
            }, ensure_ascii=False)
            f.write(result_str)
            f.write('\n')
    except Exception as e:
        print(f"[{index}] Nugget creation failed: {e}")
        return {'skipped_reason': 'nugget_creation', 'question_id': index}

    try:
        assigned_nuggets = {}
        metrics = {}
        completions = {}

        for key in ["a", "b"]:
            completions[key] = get_completion(row, key)
            assigned_nuggets[key] = nuggetizer.assign(query.text, completions[key], scored_nuggets)
            nugget_list = [
                {
                    'text': n.text,
                    'importance': n.importance,
                    'assignment': n.assignment
                } for n in assigned_nuggets[key]
            ]
            metrics[key] = calculate_nugget_scores(request.query.qid, nugget_list)

        with open(f'{PATH_PREFIX}/assignments/assigned_nuggets_{index}.json', 'w') as f2:
            result = {'question_id': index, 'winner': row['winner']}
            for key in ["a", "b"]:
                result[f'completion_{key}'] = completions[key]
                result[f'assigned_nuggets_{key}'] = [dataclasses.asdict(an) for an in assigned_nuggets[key]]
                result[f'metrics_{key}'] = metrics[key].__dict__
            result_str = json.dumps(result, ensure_ascii=False)
            f2.write(result_str)
            f2.write("\n")

        return {
            'question_id': index,
            'winner': row['winner'],
            'metrics_a': metrics['a'].__dict__,
            'metrics_b': metrics['b'].__dict__,
            'skipped_reason': None
        }

    except Exception as e:
        print(f"[{index}] Nugget assignment failed: {e}")
        return {'skipped_reason': 'nugget_assignment', 'question_id': index}

def create_and_assign_nuggets_parallel(max_workers):
    os.makedirs(f"{PATH_PREFIX}/nuggets", exist_ok=True)
    os.makedirs(f"{PATH_PREFIX}/assignments", exist_ok=True)
    data = load_dataset("lmarena-ai/search-arena-v1-7k", split="test")
    data_df = data.to_pandas()

    results = []
    skip_logs = {
        "nugget_creation": [],
        "nugget_assignment": [],
        "multi_turn": [],
    }

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_row, (index, row)): index
            for index, row in data_df.iterrows()
        }
        for future in tqdm(as_completed(futures)):
            result = future.result()
            if result.get("skipped_reason"):
                reason = result["skipped_reason"]
                skip_logs.setdefault(reason, []).append(result["question_id"])
            else:
                del result["skipped_reason"]
                results.append(result)

    print("done with all runs, saving aggregated results.")
    with open(f'{PATH_PREFIX}/results.jsonl', 'w') as results_file:
        for res in results:
            json.dump(res, results_file)
            results_file.write('\n')

    with open(f'{PATH_PREFIX}/skips.json', 'w') as skipped_file:
        json.dump(skip_logs, skipped_file)
        skipped_file.write('\n')

if __name__ == '__main__':
    create_and_assign_nuggets_parallel(max_workers=args.max_workers)
    