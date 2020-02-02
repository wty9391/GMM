#!/usr/bin/env bash

advertisers="yoyi_sample"
root="./dataset/make-yoyi-data/original-data"
for advertiser in $advertisers; do
    echo "run [python ../yoyi_dataset_encode.py $root/$advertiser/train.yzx.txt $root/$advertiser/test.yzx.txt ../result/$advertiser]"
    mkdir -p ../result/$advertiser/log/yoyi_dataset_encode
    python ../yoyi_dataset_encode.py $root/$advertiser/train.yzx.txt $root/$advertiser/test.yzx.txt ../result/$advertiser\
        1>"../result/$advertiser/log/yoyi_dataset_encode/1.log" 2>"../result/$advertiser/log/yoyi_dataset_encode/2.log"&
done


