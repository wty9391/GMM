#!/usr/bin/env bash
advertisers="1458 2261 2997 3386 3476 2259 2821 3358 3427 all"
for advertiser in $advertisers; do
    mkdir -p ../result/$advertiser/log/CGMM
    echo "run [python ../run_softmax_censored_gaussian_mixture.py ../result/$advertiser 0]"
    python ../run_softmax_censored_gaussian_mixture.py ../result/$advertiser 0\
        1>"../result/$advertiser/log/CGMM/1.log" 2>"../result/$advertiser/log/CGMM/2.log"&
done
