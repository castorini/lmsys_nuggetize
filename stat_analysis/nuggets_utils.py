import json

def load_results(files):
    data = []
    for file in files:
        with open(file, "r") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data