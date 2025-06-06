#!/bin/bash

PATH_PREFIX="" # ---> change it to your root path

# download urls
if [ ! -d "$PATH_PREFIX/downloaded_files" ]; then
    echo "Creating directory: $PATH_PREFIX/downloaded_files"
    mkdir -p "$PATH_PREFIX/downloaded_files"
fi

download_count=$(ls $PATH_PREFIX/downloaded_files | wc -l)
echo $download_count
prev_count=$download_count
# Retries previously failed downloads, stops when no new url is successfully downloaded in an iteration.
while true;
do
    python -m src.corpus_prepration.download_urls \
        --path_prefix $PATH_PREFIX
    download_count=$(ls $PATH_PREFIX/downloaded_files | wc -l)
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

# Corpus indexing 
python -m src.corpus_prepration.encode_urls_corpus \
    --path-prefix $PATH_PREFIX \
    --device cuda \
    --batch-size 96 \
    --model-name BAAI/bge-m3 \
    --dimension 1024

# Query prepration
python -m src.corpus_prepration.prepare_retrieval_queries \
    --path_prefix $PATH_PREFIX

# Retrieval
python -m src.corpus_prepration.retrieve_chunks \
    --path-prefix $PATH_PREFIX \
    --device cuda \
    --encoder BAAI/bge-m3 \
    --batch-size 128 \
    --hits 100 \
    --threads 8
