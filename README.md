# KG-CTF

This is the official implementation of [Accurate Coupled Tensor Factorization with Knowledge Graph] (BigData 2024). 
The codes for the implementation of KG-CTF (Knowledge Graph based Coupled Tensor Factorization) are included in `src/`.

![Overview]([KG-CTF.pdf](https://github.com/snudatalab/KG-CTF/blob/main/KG-CTF.pdf))

## Abstract

How can we accurately decompose a temporal irregular tensor along with a related knowledge graph tensor? The PARAFAC2 decomposition is widely used for analyzing irregular tensors composed of matrices with varying row sizes. Recent advancements in PARAFAC2 methods primarily focus on capturing dynamic features that change over time, since data irregularities often arise from temporal fluctuations. However, these methods often neglect static features, such as knowledge information, which remain constant over time. 
In this paper, we propose KG-CTF (Knowledge Graph-based Coupled Tensor Factorization), a coupled tensor factorization method designed to capture both dynamic and static features within an irregular tensor. To incorporate knowledge graph tensors as static features, KG-CTF couples an irregular temporal tensor with a knowledge graph tensor that share a common axis. Additionally, KG-CTF employs a relational regularization to capture relationships among the factor matrices of the knowledge graph tensor. For accelerated convergence of the factor matrices, KG-CTF utilizes momentum update techniques. Extensive experiments show that KG-CTF reduces error rates by up to 1.64× compared to existing PARAFAC2 methods.

## Requirements
We recommend using the following versions of packages:
- numpy==1.24.4
- tqdm==4.66.5
- pandas==2.0.3
- scikit-learn==1.3.2
- scipy==1.10.1

## Data Overview
We use six datasets.
Download the datasets from the official links.
|        **Dataset**        |                  **Link**                   |
|:-------------------------:|:-------------------------------------------:|
|       **SP500**        |           `https://www.kaggle.com/datasets/camnugent/sandp500`           |
|       **NASDAQ**        |           `https://www.kaggle.com/datasets/paultimothymooney/stock-market-data`           |
|       **NYSE**        |           `https://www.kaggle.com/datasets/paultimothymooney/stock-market-data`           |
|       **CHINA-STOCK**        |           `https://pypi.org/project/yfinance/`           |
|       **KOREA-STOCK**        |           `https://pypi.org/project/yfinance/`           |
|       **JAPAN-STOCK**        |           `https://pypi.org/project/yfinance/`           |

We also open-source StockKG, a large-scale knowledge graph that includes stock information from four major countries: South Korea, the United States, Japan, and China. For the knowledge graph dataset, we construct triple-form knowledge graphs related to all stocks present in the six stock datasets by utilizing the [ICKG](https://github.com/xiaohui-victor-li/FinDKG) model. The StockKG dataset comprises 89,822 entities, including 14,019 stock entities, and 15 relations.

## How to Run
You can run the demo script in the directory by the following code. The datasets for demo are available [[here]](https://drive.google.com/file/d/1-6AksJC0c4mHRoihVc_-hjbcF1M15hYZ/view?usp=drive_link).
```
python main.py
```

## References

If you use this code, please cite the following paper.
~~~
@inproceedings{lee2024accurate,
  author    = {SeungJoo Lee and
               Yong-chan Park and
               U Kang},
  title     = {Accurate Coupled Tensor Factorization with Knowledge Graph},
  booktitle = {Big Data},
  year      = {2024}
}
~~~

