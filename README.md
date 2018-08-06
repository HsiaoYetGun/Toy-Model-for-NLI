# Toy-Model-for-NLI

My toy model for natural language inference task. This code is implemented by TensorFlow.

# Details

I utlized biLSTM and a structured self attention to encode both premise and hypothesis as 2D-representation, then use a decomposable attention to capture the important information between premise and hypothesis. Finally, I employed a method like ESIM to combine these information.

# Model Architecture

![model](https://github.com/HsiaoYetGun/Toy-Model-for-NLI/fig/model.jpg)

# References

1. **[A Structured Self-Attentive Sentence Embedding](https://arxiv.org/abs/1703.03130)** proposed by Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen Zhou, Yoshua Bengio (ICLR 2017)
2. **[A Decomposable Attention Model for Natural Language Inference](https://arxiv.org/abs/1606.01933)** proposed by Aparikh, Oscart, Dipanjand, Uszkoreit. (EMNLP 2016)
3. **[Enhanced LSTM for Natural Language Inference](https://arxiv.org/abs/1609.06038)** proposed by Qian Chen, Xiaodan Zhu, Zhenhua Ling, Si Wei, Hui Jiang, Diana Inkpen. (ACL 2017)

# Dataset

The dataset used for this task is [Stanford Natural Language Inference (SNLI)](https://nlp.stanford.edu/projects/snli/). Pretrained [GloVe embeddings](https://nlp.stanford.edu/projects/glove/) obtained from common crawl with 840B tokens used for words.

# Requirements

- Python>=3
- NumPy
- TensorFlow>=1.8

# Usage

Download dataset from [Stanford Natural Language Inference](https://nlp.stanford.edu/projects/snli/), then move `snli_1.0_train.jsonl`, `snli_1.0_dev.jsonl`, `snli_1.0_test.jsonl` into `./SNLI/raw data`.

```com
# move dataset to the right place
mkdir -p ./SNLI/raw\ data
mv snli_1.0_*.jsonl ./SNLI/raw\ data
```

Data preprocessing for convert source data into an easy-to-use format.

```python
python3 Utils.py
```

Default hyper-parameters have been stored in config file in the path of `./config/config.yaml`.

Training model:

```python
python3 Train.py
```

Test model:

```python
python3 Test.py
```

# Results

Fune-tuning ...