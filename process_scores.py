from collections import defaultdict
import json
import argparse

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--path_prefix", type=str, default="/mnt/users/s8sharif/search_arena/with_response", help="Input and Output path prefix")
args = parser.parse_args()

# Unpack args
PATH_PREFIX = args.path_prefix

stats = defaultdict(int)
data = []

with open(f"{PATH_PREFIX}/results.jsonl", 'r') as f:
    for l in f:
        data.append(json.loads(l))

for row in data:
    if 'tie' in row['winner']:
        stats['tie'] += 1
        continue
    if row['winner'] == 'model_a':
        first_stats = row['metrics_a']
        second_stats = row['metrics_b']
    else:
        assert row['winner'] == 'model_b'
        first_stats = row['metrics_b']
        second_stats = row['metrics_a']
    vital_diff = first_stats["vital_score"]-second_stats["vital_score"]
    if vital_diff >= 0:
        stats['vital_matches'] += 1
    else:
        stats['vital_inversions'] += 1
    all_diff = first_stats["all_score"]-second_stats["all_score"]
    if all_diff >= 0:
        stats['all_matches'] += 1
    else:
        stats['all_inversions'] += 1
    strict_vital_diff = first_stats["strict_vital_score"]-second_stats["strict_vital_score"]
    if strict_vital_diff >= 0:
        stats['strict_vital_matches'] += 1
    else:
        stats['strict_vital_inversions'] += 1
    strict_all_diff = first_stats["strict_all_score"]-second_stats["strict_all_score"]
    if strict_all_diff >= 0:
        stats['strict_all_matches'] += 1
    else:
        stats['strict_all_inversions'] += 1

print(stats)
with open(f"{PATH_PREFIX}/aggregated_stats.json", 'w') as f:
    json.dump(stats, f)
    f.write('\n')