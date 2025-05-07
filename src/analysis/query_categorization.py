'''
export OMP_NUM_THREADS=1
export HF_HOME=/mnt/users/n3thakur/cache
export DATASETS_HF_HOME=/mnt/users/n3thakur/cache
export AZURE_OPENAI_API_VERSION="2024-12-01-preview"
export AZURE_OPENAI_ENDPOINT="xxxx"
export AZURE_OPENAI_API_KEY="xxxx"

python -m openai_categorization \
    --model_name_or_path gpt-4.1 \
    --train_dataset lmarena-ai/search-arena-v1-7k \
    --output_dir ./search-arena-v1-7k/categorization/ \
    --output_file categories.gpt-4.1.prompt.researchy.questions.temp.0.7.jsonl \
    --max_completion_tokens 512 \
    --temperature 0.7
'''
from .openai_client import OpenAIClient
from datasets import load_dataset
from tqdm.autonotebook import tqdm

import random
random.seed(42)

import ast
import json
import logging
import os
import argparse

PROMPT = """
Given the question: {question}
Instructions: Please output a python dictionary with fields scoring the question on the following criteria:
1. "ambiguous" : Int 0-10 to what extent is the intent of the question ambiguous (has more than one interpretation); 0 means no major ambiguity. Not to be confused with subjectiveness or incompleteness.
2. "incompleteness" : Int 0-10 indicating how difficult it is to determine the intent of the question, whether it is missing crucial context or details that ought to be specified in order to answer the question; 0 means the question is answerable and self-contained, 10 means the question is un-answerable because it is incomplete or under-specified.
3. "assumptive" : Int 0-10 the degree to which the question has built-in assumptions or biases (that are not offensive, which is point 8 below); 0 means no notable or unreasonable assumptions.
4. "multi-faceted" : Int 0-10 the degree to which the question has multiple facets or perspectives that need to be considered in order to answer it; 0 means the question is straightforward and has a single, undisputed answer.
5. "knowledge-intensive" : Int 0-10 the degree to which the question would require specialized knowledge (like textbooks, scholarly articles, etc) to provide a thorough and grounded answer; 0 means the answer is common knowledge or can be looked up instantly in common references, 10 means the questions probably entails a lot of work to find and analyze specialized knowledge.
6. "subjective" : Int 0-10 the degree to which the question is subjective, meaning an answer(s) exist, but there is no agreed-upon way to determine which one is better; 0 means the question is largely objective i.e. the overwhelming majority of people would agree on the answer if they knew it.
7. "reasoning-intensive" : Int 0-10 the degree to which the question requires reasoning to synthesize an answer; 0 means the question can be answered trivially e.g. by looking up a fact, referencing an encyclopedia or database, or using a calculator (once).
8. "harmful" : Int 0-10 to what extent the question could be interpreted as being harmful (physically or psychologically to oneself, others, or animals), offensive, overly biased, sexually explicit, or otherwise inappropriate for e.g. someone of the age of 12 to be exposed to. 

Note that the above criteria are not mutually exclusive, e.g. a question can be both subjective and knowledge-intensive, for example "is capitalism better than socialism" would be both. Make sure to output only the valid python dictionary without comments or other extraneous output."""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--output_file_save", type=str, required=False, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_completion_tokens", type=int, default=512)
    args = parser.parse_args()

    ### Download scifact.zip dataset and unzip the dataset
    hf_dataset = load_dataset(args.train_dataset, split="test")

    ### Load the filtered dataset query and positive passages as corpus
    print(f"Loading the test dataset ({args.train_dataset})): {len(hf_dataset)}")

    ### load the OpenAI client
    client = OpenAIClient(model_name_or_path=args.model_name_or_path)
    print(f"Using model: {args.model_name_or_path}")

    ### Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    output_filepath = os.path.join(args.output_dir, f"{args.output_file}")

    ### check if output file path exists
    finished_queries = set()
    if os.path.exists(output_filepath):
        print(f"Output file already exists: {output_filepath}")
        with open(output_filepath, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                finished_queries.add(data["query"])
    
    print(f"Loaded {len(finished_queries)} queries from the existing results file....")

    queries_to_dict = {}
    for row in tqdm(hf_dataset, total=len(hf_dataset), desc="Loading Dataset"): 
        query = row["messages_a"][0]['content'].strip()
        query_b = row["messages_b"][0]['content'].strip()
        assert query == query_b
        if query not in queries_to_dict:
            queries_to_dict[query] = [row["question_id"]]
        else:
            queries_to_dict[query].append(row["question_id"])
    
    print(f"Loaded {len(queries_to_dict)} unique queries....")

    ### Save the queries to a file
    
    with open(output_filepath, "a", encoding="utf-8") as f:
        for query, question_ids in tqdm(queries_to_dict.items(), total=len(queries_to_dict), desc="Processing Queries"):
            if query in finished_queries:
                continue
            
            output_text = None

            try:
                prompt = PROMPT.format(question=query)
                output = client.response(
                    prompt=prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_completion_tokens,
                    n=1,
                    disable_logging=True
                )

                output_text = output.choices[0].message.content
                if "python" in output_text:
                    output_text = output_text.replace("```python", "").replace("```", "")

                output_dict = ast.literal_eval(output_text.strip())

                for question_id in question_ids:
                    example = {
                        "question_id": question_id,
                        "query": query,
                        "categories": output_dict,
                    }        
                    ## save the example to the output directory
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
                    f.flush()

            except Exception as e:
                print(f"Error processing query: {query}, output: {output_text}")
                print(f"Error: {e}")
                continue

if __name__ == "__main__":
    main()