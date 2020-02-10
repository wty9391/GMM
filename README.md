# A Discriminative Gaussian Mixture Model to Fit Market Price Distribution for Demand-Side Platforms
This is a repository of experiment code supporting [A Discriminative Gaussian Mixture Model to Fit Market Price Distribution for Demand-Side Platforms]().

For any problems, please report the issues here.

## Quirk Start

### Prepare Dataset
Before run the demo, please first check the GitHub project [make iPinYou data](https://github.com/wnzhang/make-ipinyou-data) for pre-processing the [iPinYou dataset](http://data.computational-advertising.org).
Or you can download the processed dataset from this [link](https://pan.baidu.com/s/1bjeROrEuxouy9Mhfd1vrCw) with extracting code `h12c`.

Then, please create a folder named `dataset`, and put the dataset in it.
The file tree looks like this:
```
GMM
│───README.md
│
└───dataset
│   └───make-ipinyou-data
│       │   1458
│       │   2259
│       │   ...
...
```

### Encode Dataset
Please run the following code to encode the dataset
```bash
cd ./shell
bash ./ipinyou_dataset_encode.sh
```
You can find the running logs in this directory `/result/$advertiser/log/dataset_encode`

### Run GMM and CGMM
Please run the following code to train and evaluate the GMM and CGMM
```bash
bash ./GMM.sh
bash ./CGMM.sh
```
You can find the running logs in this directories `/result/$advertiser/log/GMM` and `/result/$advertiser/log/CGMM`


### Run Baselines
Please run the following code to train and evaluate the baselines
```bash
bash ./tanh_LT_LG.sh
bash ./CLR.sh
bash ./SM.sh
bash ./DCL
bash ./DLF
```
You can find the running logs in these directories `/result/$advertiser/log/tanh_LT_LG`, `/result/$advertiser/log/CLR`,
`/result/$advertiser/log/SM`, `/result/$advertiser/log/DCL` and `/result/$advertiser/log/DLF`.

The code of DCL and DLF is forked from this [repository](https://github.com/rk2900/DLF).







