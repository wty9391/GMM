#!/usr/bin/env bash
advertisers="1458 2261 2997 3386 3476 2259 2821 3358 3427 all"
for advertiser in $advertisers; do
    mkdir -p ../result/$advertiser/log/DCL
    echo "run [python ../baseline/run_DCL.py $advertiser]"
    python ../baseline/run_DCL.py $advertiser 0\
        1>"../result/$advertiser/log/DCL/1.log" 2>"../result/$advertiser/log/DCL/2.log"&
done
