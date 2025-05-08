#!/bin/bash
export AZURE_OPENAI_API_VERSION="2024-12-01-preview"
export AZURE_OPENAI_ENDPOINT="xxxx"
export AZURE_OPENAI_API_KEY="xxxx"

PATH_PREFIX="" # ---> change it to your root path
VISUALIZATION_OUTPUT_DIR=./figures
LANGUAGE=English
METRICS=all_score
THRESHOLD=0.1
CLASS_THRESHOLD=7

# Outputs aggregated results.jsonl and skips.json,
# in addition to dumping per query nuggets and assignments.
# Uses the two model compeletions as documents for nugget creation.
# python -m src.nuggetize_responses \
#   --sampling_rate 1 \
#   --path_prefix $PATH_PREFIX


# Outputs aggregated results.jsonl and skips.json,
# in addition to dumping per query nuggets and assignments.
# Uses the summary of scraped urls as documents for nugget creation.
# python -m src.nuggetize_with_web_search \
#   --sampling_rate 1 \
#   --path_prefix $PATH_PREFIX \
#   --url_to_txt_filepath "/mnt/users/s8sharif/search_arena/urls_to_summary_files.json"


# outputs some auxiliary analysis files including the per_language_inversion_ids.json
python  -m src.analysis.process_results \
  --path_prefix $PATH_PREFIX \
  --inversion_metric $METRICS \
  --candidates_language $LANGUAGE \
  --inversion_threshold $THRESHOLD

# Outputs the categoryization results for each query 
# python -m src.analysis.query_categorization \
#     --model_name_or_path gpt-4.1 \
#     --train_dataset lmarena-ai/search-arena-v1-7k \
#     --output_dir $PATH_PREFIX/search-arena-v1-7k/ \
#     --output_file categories.gpt-4.1.temp.0.7.jsonl \
#     --max_completion_tokens 512 \
#     --temperature 0.7

CATEGORIES_PATH=$PATH_PREFIX/search-arena-v1-7k/categories.gpt-4.1.temp.0.7.jsonl

# Table 1: inversions by query category
python -m src.analysis.inversions_by_category \
  --path_prefix $PATH_PREFIX \
  --categories_path $CATEGORIES_PATH \
  --class_threshold $CLASS_THRESHOLD

# Table 2: inversions by language
python -m src.analysis.inversions_by_language \
  --path_prefix $PATH_PREFIX

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
  --class_threshold $CLASS_THRESHOLD

# Figure 7
python -m src.visualization.dataset_stats \
  --output_dir $VISUALIZATION_OUTPUT_DIR \
  --top_n_languages 5

# Figure 8
python -m src.visualization.category_histogram \
    --results_path $PATH_PREFIX/search-arena-v1-7k/categories.gpt-4.1.temp.0.7.jsonl \
    --output_dir $VISUALIZATION_OUTPUT_DIR/query_category \
    --hf_dataset lmarena-ai/search-arena-v1-7k \
    --filter_single_turn

# Table 3
python -m src.analysis.sample_queries_per_category \
  --path_prefix $PATH_PREFIX \
  --categories_path $CATEGORIES_PATH \
  --class_threshold $CLASS_THRESHOLD \
  --language $LANGUAGE
