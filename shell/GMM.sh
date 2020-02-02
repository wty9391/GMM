#!/usr/bin/env bash
advertisers="1458 2261 2997 3386 3476 2259 2821 3358 3427 all"
#Ks="1 2 3 5 7 10 12 15"
Ks="10"
for advertiser in $advertisers; do
    for k in $Ks; do
        mkdir -p ../result/$advertiser/log/GMM/$k
        echo "run [python ../run_softmax_gaussian_mixture.py ../result/$advertiser $k 0]"
        python ../run_softmax_gaussian_mixture.py ../result/$advertiser $k 0\
            1>"../result/$advertiser/log/GMM/$k/1.log" 2>"../result/$advertiser/log/GMM/$k/2.log"&
    done
done
