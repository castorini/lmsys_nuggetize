import random
import json
import dataclasses
from concurrent.futures import ProcessPoolExecutor, as_completed

from datasets import load_dataset
import pandas as pd
from nuggetizer.core.types import Query, Document, Request
from nuggetizer.models.nuggetizer import Nuggetizer
from nuggetizer.core.metrics import calculate_nugget_scores
from tqdm import tqdm


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
    if random.random() > 0.0005:
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

        nuggetizer = Nuggetizer(model="gpt-4o-mini")
        scored_nuggets = nuggetizer.create(request)
        with open(f'/mnt/users/s8sharif/search_arena/with_response/nuggets/requests_with_nuggets_{index}.json', 'w') as f:
            json.dump({
                "question_id": index,
                "request": dataclasses.asdict(request),
                "scored_nuggets": [dataclasses.asdict(sn) for sn in scored_nuggets]
            }, f)
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

        with open(f'/mnt/users/s8sharif/search_arena/with_response/assignments/assigned_nuggets_{index}.json', 'w') as f2:
            result = {'question_id': index, 'winner': row['winner']}
            for key in ["a", "b"]:
                result[f'completion_{key}'] = completions[key]
                result[f'assigned_nuggets_{key}'] = [dataclasses.asdict(an) for an in assigned_nuggets[key]]
                result[f'metrics_{key}'] = metrics[key].__dict__
            json.dump(result, f2)

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


def create_and_assign_nuggets_parallel(max_workers=4):
    data = load_dataset("lmarena-ai/search-arena-v1-7k", split="test")
    data_df = data.to_pandas()

    results = []
    skip_logs = {
        "nugget_creation": [],
        "nugget_assignment": [],
        "multi_turn": [],
    }

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_row, (index, row)): index for index, row in tqdm(data_df.iterrows(),total=data_df.shape[0])}
        for future in as_completed(futures):
            result = future.result()
            print(f"processed: {result['question_id']}")
            if result.get("skipped_reason"):
                reason = result["skipped_reason"]
                skip_logs.setdefault(reason, []).append(result["question_id"])
            else:
                del result["skipped_reason"]
                results.append(result)

    print("done with all runs, saving aggeregated results.")
    with open('/mnt/users/s8sharif/search_arena/with_response/results.jsonl', 'w') as results_file:
        for res in results:
            json.dump(res, results_file)
            results_file.write('\n')

    with open('/mnt/users/s8sharif/search_arena/with_response/skips.json', 'w') as skipped_file:
        json.dump(skip_logs, skipped_file)
        skipped_file.write('\n')


if __name__ == '__main__':
    create_and_assign_nuggets_parallel(max_workers=2)