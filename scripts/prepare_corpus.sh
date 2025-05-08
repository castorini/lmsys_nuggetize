#!/bin/bash

PATH_PREFIX=/mnt/users/s8sharif/search_arena

# download urls
urls_count=$(wc -l < $PATH_PREFIX/urls.txt)
echo $urls_count
download_count=$(ls $PATH_PREFIX/htmls | wc -l)
echo $download_count
prev_count=$download_count
while [ $download_count -lt $urls_count ];
do
    python -m src.corpus_prepration.download_urls \
        --path_prefix $PATH_PREFIX
    download_count=$(ls $PATH_PREFIX/htmls | wc -l)
    if [ $download_count = $prev_count ];
    then
        break
    fi
    prev_count=$download_count
    echo $prev_count
done

# extract text
python -m src.corpus_prepration.scrape_texts \
    --path_prefix $PATH_PREFIX
scrape_count=$(ls $PATH_PREFIX/scraped_texts | wc -l)
echo $scrape_count

python -m src.corpus_prepration.llm_summarize \
    --path_prefix $PATH_PREFIX
summary_count=$(ls $PATH_PREFIX/summary_texts | wc -l)
echo $summary_count