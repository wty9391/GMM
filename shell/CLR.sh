#!/usr/bin/env bash
advertisers="1458 2261 2997 3386 3476 2259 2821 3358 3427 all"
for advertiser in $advertisers; do
    mkdir -p ../result/$advertiser/log/CLR
    echo "run [python ../baseline/run_CLR.py ../result/$advertiser 0]"
    python ../baseline/run_CLR.py ../result/$advertiser 0\
        1>"../result/$advertiser/log/CLR/1.log" 2>"../result/$advertiser/log/CLR/2.log"&
done
