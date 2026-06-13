# Locked Results

## Segmentation

| Run | Split | n | Dice | IoU | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | validation | 114 | 0.670688 | 0.537722 | 0.661313 | 0.758906 |
| raw | test | 115 | 0.632329 | 0.496682 | 0.645789 | 0.729766 |
| ROI10 | validation | 114 | 0.670791 | 0.538289 | 0.662263 | 0.756408 |
| ROI10 | test | 115 | 0.635262 | 0.500053 | 0.651698 | 0.729273 |

## Full Inference And GNC

| Metric | Value |
| --- | ---: |
| Full inference cases | 39,530 |
| Ports | 49 |
| Port-month observations | 4,797 |
| Country-month observations | 2,323 |
| Countries | 23 |
| China without Hong Kong months | 110 |

## MAD Portfolio Optimization

| Portfolio | Treasury Sharpe |
| --- | ---: |
| Best GNC-informed MAD | 0.915243 |
| Historical-mean no-GNC baseline | 0.830063 |
| Always-positive baseline | 0.768995 |
| Matched equal-weight benchmark | 0.548530 |

Best GNC-informed configuration:

```text
distributed_lag, h=1, raw signal, L=5
```

Other locked winner metrics:

| Metric | Value |
| --- | ---: |
| Raw compound return | 1.567429 |
| Equal-weight raw compound return | 0.910660 |
| Raw max drawdown | -0.194416 |
| Equal-weight raw max drawdown | -0.235974 |
| Mean turnover | 0.525705 |
