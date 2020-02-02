#!/usr/bin/env bash
advertisers="1458 2261 2997 3386 3476 2259 2821 3358 3427 all"
learning_rate=0.0001
for advertiser in $advertisers; do
    mkdir -p ../result/$advertiser/log/DLF
    echo "run [python ../baseline/run_DLF.py $learning_rate $advertiser]"
    python ../baseline/run_DLF.py $learning_rate $advertiser \
        1>"../result/$advertiser/log/DLF/1.log" 2>"../result/$advertiser/log/DLF/2.log"&
done
