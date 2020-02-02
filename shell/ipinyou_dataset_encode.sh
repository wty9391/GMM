#!/usr/bin/env bash
advertisers="1458 2261 2997 3386 3476 2259 2821 3358 3427 all"
root="../dataset/make-ipinyou-data/"
for advertiser in $advertisers; do
    mkdir -p ../result/$advertiser/log/dataset_encode
    echo "run [python ../ipinyou_dataset_encode.py $root/$advertiser/train.log.txt $root/$advertiser/test.log.txt $root/$advertiser/featindex_no_ip.txt ../result/$advertiser]"
    python ../ipinyou_dataset_encode.py $root/$advertiser/train.log.txt $root/$advertiser/test.log.txt $root/$advertiser/featindex_no_ip.txt ../result/$advertiser\
        1>"../result/$advertiser/log/dataset_encode/1.log" 2>"../result/$advertiser/log/dataset_encode/2.log"&
done