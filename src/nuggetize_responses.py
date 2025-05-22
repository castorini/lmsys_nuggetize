import argparse
import dataclasses
import json
import os
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from datasets import load_dataset
from nuggetizer.core.metrics import calculate_nugget_scores
from nuggetizer.core.types import Document, Query, Request
from nuggetizer.models.nuggetizer import Nuggetizer
from tqdm import tqdm

from src.utils import get_prompt

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--sampling_rate",
    type=float,
    default=0.005,
    help="Sampling rate for processing rows",
)
parser.add_argument(
    "--path_prefix",
    type=str,
    default="/mnt/users/s8sharif/search_arena/with_response",
    help="Output path prefix",
)
parser.add_argument(
    "--max_workers", type=int, default=4, help="Number of parallel workers"
)
parser.add_argument(
    "--model_name",
    type=str,
    default="gpt-4.1",
    help="the model name, for now from gpt family only.",
)
parser.add_argument(
    "--retrieved_runfile",
    type=str,
    required=False,
    default="",
    help="the filepath to retrieved results in trec eval format.",
)
parser.add_argument(
    "--chunks_file",
    type=str,
    required=False,
    default="",
    help="the path to the jsonl file containing chunked scraped urls.",
)
parser.add_argument(
    "--max_chunks",
    type=int,
    required=False,
    default=50,
    help="the maximum number or retrieved chunks used for nugget creation",
)
args = parser.parse_args()

# Unpack args
SAMPLING_RATE = args.sampling_rate
PATH_PREFIX = args.path_prefix
MODEL_NAME = args.model_name
RETRIEVED_RUNFILE = args.retrieved_runfile
CHUNKS_FILE = args.chunks_file
MAX_CHUNKS = args.max_chunks


def get_completion(row, key):
    message = row[f"messages_{key}"][1]
    assert message["role"] == "assistant"
    return message["content"]


def parse_rank_file(retrieved_runfile):
    qid_to_doc_ids = defaultdict(list)
    with open(retrieved_runfile, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, _, doc_id, _, _, _ = parts
            qid_to_doc_ids[qid].append(doc_id)
    return qid_to_doc_ids


def load_retrieved_chunks(chunks_file):
    doc_id_to_chunk = {}
    with open(chunks_file, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            doc_id_to_chunk[data["_id"]] = data["text"]
    return doc_id_to_chunk


def process_row(index_row_tuple):
    index, row, retrieved_chunks = index_row_tuple
    if row["turn"] != 1:
        return {"skipped_reason": "multi_turn", "question_id": index}
    if random.random() > SAMPLING_RATE:
        return {"skipped_reason": "sampling", "question_id": index}

    try:
        query = Query(qid=index, text=get_prompt(row))
        documents = [
            Document(docid="a", segment=get_completion(row, "a")),
            Document(docid="b", segment=get_completion(row, "b")),
        ]
        if random.random() > 0.5:
            documents[0], documents[1] = documents[1], documents[0]
        for chunk_id, chunk in retrieved_chunks:
            documents.append(Document(docid=chunk_id, segment=chunk))
        request = Request(query=query, documents=documents)

        nuggetizer = Nuggetizer(model=MODEL_NAME, use_azure_openai=True)
        scored_nuggets = nuggetizer.create(request)
        if not scored_nuggets:
            raise ValueError("No nuggets were created.")

        with open(
            f"{PATH_PREFIX}/nuggets/requests_with_nuggets_{index}.json", "w"
        ) as f:
            result_str = json.dumps(
                {
                    "question_id": index,
                    "request": dataclasses.asdict(request),
                    "scored_nuggets": [dataclasses.asdict(sn) for sn in scored_nuggets],
                },
                ensure_ascii=False,
            )
            f.write(result_str)
            f.write("\n")
    except Exception as e:
        print(f"[{index}] Nugget creation failed: {e}")
        return {"skipped_reason": "nugget_creation", "question_id": index}

    try:
        assigned_nuggets = {}
        metrics = {}
        completions = {}

        for key in ["a", "b"]:
            completions[key] = get_completion(row, key)
            assigned_nuggets[key] = nuggetizer.assign(
                query.text, completions[key], scored_nuggets
            )
            nugget_list = [
                {"text": n.text, "importance": n.importance, "assignment": n.assignment}
                for n in assigned_nuggets[key]
            ]
            metrics[key] = calculate_nugget_scores(request.query.qid, nugget_list)

        with open(
            f"{PATH_PREFIX}/assignments/assigned_nuggets_{index}.json", "w"
        ) as f2:
            result = {"question_id": index, "winner": row["winner"]}
            for key in ["a", "b"]:
                result[f"completion_{key}"] = completions[key]
                result[f"assigned_nuggets_{key}"] = [
                    dataclasses.asdict(an) for an in assigned_nuggets[key]
                ]
                result[f"metrics_{key}"] = metrics[key].__dict__
            result_str = json.dumps(result, ensure_ascii=False)
            f2.write(result_str)
            f2.write("\n")

        return {
            "question_id": index,
            "winner": row["winner"],
            "metrics_a": metrics["a"].__dict__,
            "metrics_b": metrics["b"].__dict__,
            "skipped_reason": None,
        }

    except Exception as e:
        print(f"[{index}] Nugget assignment failed: {e}")
        return {"skipped_reason": "nugget_assignment", "question_id": index}


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
    qids_to_docids = parse_rank_file(RETRIEVED_RUNFILE) if RETRIEVED_RUNFILE else {}
    docid_to_chunk = load_retrieved_chunks(CHUNKS_FILE) if CHUNKS_FILE else {}
    qid_to_chunks = defaultdict(list)
    for qid, doc_ids in qids_to_docids.items():
        qid_to_chunks[int(qid)] = [
            (doc_id, docid_to_chunk[doc_id]) for doc_id in doc_ids[:MAX_CHUNKS]
        ]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_row, (index, row, qid_to_chunks[row["question_id"]])
            ): index
            for index, row in data_df.iterrows()
        }
        for future in tqdm(as_completed(futures), total=data_df.shape[0]):
            result = future.result()
            if result.get("skipped_reason"):
                reason = result["skipped_reason"]
                skip_logs.setdefault(reason, []).append(result["question_id"])
            else:
                del result["skipped_reason"]
                results.append(result)

    print("done with all runs, saving aggregated results.")
    with open(f"{PATH_PREFIX}/results.jsonl", "w") as results_file:
        for res in results:
            json.dump(res, results_file)
            results_file.write("\n")

    with open(f"{PATH_PREFIX}/skips.json", "w") as skipped_file:
        json.dump(skip_logs, skipped_file)
        skipped_file.write("\n")


if __name__ == "__main__":
    create_and_assign_nuggets_parallel(max_workers=args.max_workers)
