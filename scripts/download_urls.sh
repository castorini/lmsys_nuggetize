#!/bin/bash
urls_count=$(wc -l < /mnt/users/s8sharif/search_arena/urls.txt)
echo $urls_count

download_count=$(ls /mnt/users/s8sharif/search_arena/htmls | wc -l)
echo $download_count

prev_count=$download_count
while [ $download_count -lt $urls_count ];
do
    python download_urls.py
    download_count=$(ls /mnt/users/s8sharif/search_arena/htmls | wc -l)
    if [ $download_count = $prev_count ];
    then
        break
    fi
    prev_count=$download_count
    echo $prev_count
done