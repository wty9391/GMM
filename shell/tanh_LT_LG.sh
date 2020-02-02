#!/usr/bin/env bash
advertisers="1458 2261 2997 3386 3476 2259 2821 3358 3427 all"
for advertiser in $advertisers; do
    mkdir -p ../result/$advertiser/log/tanh_LT_LG
    echo "run [python ../baseline/run_tanh_LT_LG.py ../result/$advertiser]"
    python ../baseline/run_tanh_LT_LG.py ../result/$advertiser \
        1>"../result/$advertiser/log/tanh_LT_LG/1.log" 2>"../result/$advertiser/log/tanh_LT_LG/2.log"&
done
