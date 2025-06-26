#!/bin/bash

DATASET="lmarena-ai/search-arena-24k" # or "lmarena-ai/search-arena-v1-7k"
PATH_PREFIX="" # ---> change it to your root path
USE_FIRECRAWL=false # firecrawl fails to download lots of urls. Other than download urls, the rest of the script assumes this is false.
if [ "$USE_FIRECRAWL" = true ]; then
    DOWNLOADED_FILE_PATH="${PATH_PREFIX}/firecrawl_downloaded_files"
else
    DOWNLOADED_FILE_PATH="${PATH_PREFIX}/downloaded_files"
fi

# download urls
if [ ! -d "$DOWNLOADED_FILE_PATH" ]; then
    echo "Creating directory: $DOWNLOADED_FILE_PATH"
    mkdir -p "$DOWNLOADED_FILE_PATH"
fi

download_count=$(ls $DOWNLOADED_FILE_PATH | wc -l)
echo $download_count
prev_count=$download_count
# Retries previously failed downloads, stops when no new url is successfully downloaded in an iteration.
while true;
do
    if [ "$USE_FIRECRAWL" = true ]; then
        python -m src.corpus_prepration.download_urls \
            --path_prefix $PATH_PREFIX \
            --dataset $DATASET \
            --skip_multi_turn \
            --skip_no_vote \
            --use_firecrawl
    else
        python -m src.corpus_prepration.download_urls \
            --path_prefix $PATH_PREFIX \
            --dataset $DATASET \
            --skip_multi_turn \
            --skip_no_vote
    fi

    download_count=$(ls $DOWNLOADED_FILE_PATH | wc -l)
    if [ $download_count = $prev_count ];
    then
        break
    fi
    prev_count=$download_count
    echo $prev_count
done

# Extract text
python -m src.corpus_prepration.scrape_texts \
    --path_prefix $PATH_PREFIX
scrape_count=$(ls $PATH_PREFIX/scraped_texts | wc -l)
echo $scrape_count

# Chunk extracted texts
python -m spacy download xx_sent_ud_sm
python -m src.corpus_prepration.chunk_texts \
    --path_prefix $PATH_PREFIX

# Query prepration
python -m src.corpus_prepration.prepare_retrieval_queries \
    --path_prefix $PATH_PREFIX \
    --dataset $DATASET \
    --skip_multi_turn \
    --skip_no_vote

# Corpus indexing 
python -m src.corpus_prepration.encode_urls_corpus \
    --path-prefix $PATH_PREFIX \
    --device cuda \
    --batch-size 96 \
    --model-name BAAI/bge-m3 \
    --dimension 1024

# Retrieval
python -m src.corpus_prepration.retrieve_chunks \
    --path-prefix $PATH_PREFIX \
    --device cuda \
    --encoder BAAI/bge-m3 \
    --batch-size 128 \
    --hits 100 \
    --threads 8
