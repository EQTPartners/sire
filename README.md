<img src="./picture/sire_logo.png" alt="SiRE" height="70"/>

# Simulation-informed Revenue Extrapolation (SiRE) with confidence estimate for scaleup companies using scarce timeseries data


Investment professionals rely on extrapolating company revenue into the future (i.e. revenue forecast) to approximate the valuation of scaleups (private companies in a high-growth stage) and inform their investment decision. This task is manual and empirical, leaving the forecast quality heavily dependent on the investment professionals’ experiences and insights. Furthermore, financial data on scaleups is typically proprietary, costly and scarce, ruling out the wide adoption of data-driven approaches. 

We propose an algorithm **SiRE (Simulation-informed Revenue Extrapolation)** that generates fine-grained long-term revenue predictions on small datasets and short time-series.
SiRE fulfills eight important practical requirements: 
1. it performs well in multiple business sectors;
2. it works on small datasets with only a few hundred scaleups; 
3. the extrapolation can commence from short revenue time-series; 
4. it should produce a fine-grained time-series of at least three-year length; 
5. each predicted revenue point should come with a confidence estimation; 
6. it does not require any alternative data other than sector information;
7. the model can be timely and effortlessly adapted to any data change. 
8. the prediction is explainable.

## Usage

The source code and two datasets (transformed and anonymized) are already bundled in this repo. 
To run SiRE on both datasets, follow the steps below.

- To isolate the Python runtime for SiRE experimentation, install a virtual environment such as [Anaconda](https://www.anaconda.com/). We hereby assume that you are using Anaconda. The following steps should be run in the project folder `sire` from a Terminal.

- Create a virtual Python environment by running:
```bash
$ conda create --name sire python=3.8
$ conda activate sire
```

- Install the dependent libraries.
```bash
$ pip install -r requirements.txt
```

- Start Python Notebook:
```bash
$ jupyter notebook
```

To experiment SiRE on [SapiQ](data/sapiq.json) dataset, run through the notebook named [`sire_sapiq.ipynb`](sire_sapiq.ipynb).

To experiment SiRE on [ARR129](data/arr129.json) dataset, run through the notebook named [`sire_arr129.ipynb`](sire_arr129.ipynb).

## Example Forecast

Following is the forecast (from 2018-05-01) visualization for company in ARR129 with ID `5cf74ff84639c79510ef8491d4725967336ad3627216a50553f030f9c51d93bc`.

<img src="./picture/example_viz.png" alt="drawing" width="500"/>

The corresponding predicted revenue points with 95% confidence interval (`95CI`) look like below.
|    | DATE     |   MEAN |   `95CI`_low |   `95CI`_high |   MIN |   MAX |
|---:|:--------------------|---------------:|-------------------------:|-------------------------:|--------------:|--------------:|
|  0 | 2018-05-01 |        52.6052 |                  51.6823 |                  53.5282 |       50.6054 |       54.8298 |
|  1 | 2018-06-01 |        55.3628 |                  54.0489 |                  56.6767 |       52.6138 |       58.5401 |
|  2 | 2018-07-01 |        57.8639 |                  56.1423 |                  59.5856 |       54.3638 |       62.0497 |
|  3 | 2018-08-01 |        60.2249 |                  58.0926 |                  62.3571 |       55.9818 |       65.4445 |
|  4 | 2018-09-01 |        62.5506 |                  60.0036 |                  65.0977 |       57.5778 |       68.819  |
|  5 | 2018-10-01 |        64.9284 |                  61.9523 |                  67.9044 |       59.2417 |       72.2662 |
|  6 | 2018-11-01 |        67.4283 |                  63.9957 |                  70.8609 |       61.0437 |       75.8727 |
|  7 | 2018-12-01 |        70.1062 |                  66.1761 |                  74.0362 |       63.0244 |       79.7138 |
|  8 | 2019-01-01 |        73.0071 |                  68.528  |                  77.4861 |       65.2085 |       83.8526 |
|  9 | 2019-02-01 |        76.1667 |                  71.082  |                  81.2514 |       67.6073 |       88.3364 |
| 10 | 2019-03-01 |        79.616  |                  73.8686 |                  85.3634 |       70.2262 |       93.1938 |
| 11 | 2019-04-01 |        83.3822 |                  76.9178 |                  89.8467 |       73.0796 |       98.4444 |
| 12 | 2019-05-01 |        87.4905 |                  80.2587 |                  94.7224 |       76.1945 |      104.091  |
| 13 | 2019-06-01 |        91.9663 |                  83.9186 |                 100.014  |       79.6082 |      110.137  |
| 14 | 2019-07-01 |        96.8336 |                  87.9192 |                 105.748  |       83.3569 |      116.879  |
| 15 | 2019-08-01 |       102.117  |                  92.2778 |                 111.955  |       87.4871 |      124.982  |
| 16 | 2019-09-01 |       107.839  |                  97.0088 |                 118.668  |       92.0415 |      133.812  |
| 17 | 2019-10-01 |       114.022  |                 102.124  |                 125.921  |       97.045  |      143.432  |
| 18 | 2019-11-01 |       120.692  |                 107.633  |                 133.751  |      102.516  |      153.902  |
| 19 | 2019-12-01 |       127.874  |                 113.546  |                 142.201  |      108.453  |      165.266  |
| 20 | 2020-01-01 |       135.593  |                 119.877  |                 151.308  |      114.86   |      177.557  |
| 21 | 2020-02-01 |       143.871  |                 126.639  |                 161.104  |      121.739  |      190.775  |
| 22 | 2020-03-01 |       152.735  |                 133.858  |                 171.612  |      129.099  |      204.875  |
| 23 | 2020-04-01 |       162.21   |                 141.575  |                 182.845  |      136.84   |      219.772  |
| 24 | 2020-05-01 |       172.332  |                 149.844  |                 194.82   |      144.549  |      235.377  |
| 25 | 2020-06-01 |       183.145  |                 158.726  |                 207.564  |      152.722  |      251.624  |
| 26 | 2020-07-01 |       194.699  |                 168.276  |                 221.122  |      161.392  |      268.535  |
| 27 | 2020-08-01 |       207.049  |                 178.539  |                 235.559  |      170.614  |      286.235  |
| 28 | 2020-09-01 |       220.241  |                 189.552  |                 250.929  |      180.467  |      304.86   |
| 29 | 2020-10-01 |       234.302  |                 201.329  |                 267.276  |      191.038  |      324.585  |
| 30 | 2020-11-01 |       249.223  |                 213.84   |                 284.605  |      202.373  |      345.567  |
| 31 | 2020-12-01 |       264.929  |                 226.993  |                 302.865  |      214.456  |      367.884  |
| 32 | 2021-01-01 |       281.286  |                 240.629  |                 321.942  |      227.181  |      391.532  |
| 33 | 2021-02-01 |       298.059  |                 254.498  |                 341.621  |      240.334  |      416.363  |
| 34 | 2021-03-01 |       314.914  |                 268.262  |                 361.567  |      253.592  |      442.045  |
| 35 | 2021-04-01 |       331.436  |                 281.516  |                 381.356  |      266.558  |      468.107  |

## Cite This Work
```bibtex
@inproceedings{cao-etal-2022-sire,
    title = "Simulation-Informed Revenue Extrapolation with Confidence Estimate for Scaleup Companies Using Scarce Time Series Data",
    author = "Cao, Lele  and
      Horn, Sonja  and
      von Ehrenheim, Vilhelm  and
      Stahl, Richard Anselmo  and
      Landgren, Henrik",
    booktitle = "Proceedings of the 31st ACM International Conference on Information and Knowledge Management",
    month = oct,
    year = "2022",
    address = "Online and Atlanta, Georgia, USA",
    publisher = "Association for Computing Machinery (ACM)",
    pages = "1--10",
}
```