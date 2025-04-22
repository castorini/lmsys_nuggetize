from collections import defaultdict
import json

stats = defaultdict(int)
data = []
with open("./output.jsonl", 'r') as f:
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