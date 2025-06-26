#!/bin/bash
export AZURE_OPENAI_API_VERSION="2024-12-01-preview"
export AZURE_OPENAI_API_BASE="xxxx" 
export AZURE_OPENAI_ENDPOINT="xxxx" # Same as base
export AZURE_OPENAI_API_KEY="xxxx"

DATASET="lmarena-ai/search-arena-24k" # or "lmarena-ai/search-arena-v1-7k"
PATH_PREFIX="" # ---> change it to your root path
VISUALIZATION_OUTPUT_DIR="${PATH_PREFIX}/visualization" # ---> change it to your figures dir
LANGUAGE=English
METRICS=all_score
THRESHOLD=0.07
CLASS_THRESHOLD=7

mkdir -p $PATH_PREFIX
mkdir -p $VISUALIZATION_OUTPUT_DIR

# Outputs aggregated results.jsonl and skips.json,
# in addition to dumping per query nuggets and assignments.
# Uses the two model completions and retrieved url contents as documents for nugget creation.
# For excluding the retrieved URLs (Figure 7), do not pass the retrieved_runfile and chunks_file args, and set the THRESHOLD to 0.1.
python -m src.nuggetize_responses \
  --sampling_rate 1 \
  --path_prefix $PATH_PREFIX \
  --dataset $DATASET \
  --retrieved_runfile  $PATH_PREFIX/runs/run.bge-m3.url_corpus.txt \
  --chunks_file $PATH_PREFIX/urls_chunked_corpus.jsonl \
  --skip_multi_turn \
  --skip_no_vote

# Outputs some auxiliary analysis files including the per_language_inversion_ids.json
python  -m src.analysis.process_results \
  --path_prefix $PATH_PREFIX \
  --inversion_metric $METRICS \
  --candidates_language $LANGUAGE \
  --inversion_threshold $THRESHOLD \
  --dataset $DATASET

# Outputs the categoryization results for each query 
python -m src.analysis.query_categorization \
  --model_name_or_path gpt-4.1 \
  --train_dataset $DATASET \
  --output_dir $PATH_PREFIX \
  --output_file query_categories_model_gpt-4.1_temp_0.7.jsonl \
  --max_completion_tokens 512 \
  --temperature 0.7 \
  --skip_multi_turn \
  --skip_no_vote

CATEGORIES_PATH=$PATH_PREFIX/query_categories_model_gpt-4.1_temp_0.7.jsonl

# Table 1 (side a): inversions by query category
python -m src.analysis.inversions_by_category \
  --path_prefix $PATH_PREFIX \
  --categories_path $CATEGORIES_PATH \
  --class_threshold $CLASS_THRESHOLD

# Table 1 (side b): inversions by language
python -m src.analysis.inversions_by_language \
  --path_prefix $PATH_PREFIX \
  --dataset $DATASET

# Figures 2 and 3
python -m src.visualization.distribution_density \
  --metrics $METRICS \
  --results_path $PATH_PREFIX/results.jsonl \
  --output_dir $VISUALIZATION_OUTPUT_DIR

# Figures 4, 5, 6
python -m src.visualization.confusion_matrix \
  --results_path $PATH_PREFIX/results.jsonl \
  --categories_path $CATEGORIES_PATH \
  --output_dir $VISUALIZATION_OUTPUT_DIR \
  --metrics $METRICS \
  --preference_threshold $THRESHOLD \
  --class_threshold $CLASS_THRESHOLD \
  --dataset $DATASET

# Figure 8: LLM-as-a-judge
# TODO(Nandan)

# Figure 9
python -m src.visualization.dataset_stats \
  --output_dir $VISUALIZATION_OUTPUT_DIR \
  --top_n_languages 5 \
  --dataset $DATASET \
  --skip_multi_turn \
  --skip_no_vote

# Figure 10
python -m src.visualization.category_histogram \
  --results_path $CATEGORIES_PATH \
  --output_dir $VISUALIZATION_OUTPUT_DIR/query_category \
  --dataset $DATASET \
  --skip_multi_turn \
  --skip_no_vote

# Table 2
python -m src.analysis.sample_queries_per_category \
  --path_prefix $PATH_PREFIX \
  --categories_path $CATEGORIES_PATH \
  --class_threshold $CLASS_THRESHOLD \
  --language $LANGUAGE \
  --dataset $DATASET
