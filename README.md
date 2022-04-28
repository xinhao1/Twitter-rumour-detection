# NLP pj

## Requirements
- python 3.7
- [PyTorch](https://pytorch.org/)
- [Transformers](https://huggingface.co/transformers/)

```angular2html
pip install -r requirements.txt
```

## Wandb versioning (Optional)
```angular2html
wandb login
```

## Crawl tweet

```angular2html
cd data
mkdir train_tweet
mkdir dev_tweet
mkdir analysis_tweet
cd ..
nohup python src/crawl_tweet.py >crawl.out 2>&1 &
```

## Train and Test for competition

```angular2html
nohup bash exp/train.sh >train.out 2>&1 &
```

## Analysis Tweet

1. modify the dataset.py pl_model.py to generate the analysis_data (see the commented code)
2. set the ckpt in test.sh
3. predict the analysis_tweet data
```angular2html
nohup bash exp/test.sh >test.out 2>&1 &
```
4. move the analysis_data to project root path
5. run the analysis_tweet.ipynb
