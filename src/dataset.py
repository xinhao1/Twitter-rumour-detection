#!/usr/bin/env python
import json
from torch.utils.data import Dataset
import os
import time
import torch


class TrainDataset(Dataset):

    def __init__(
            self,
            tokenizer,
            max_length,
            n_obs=None,
    ):
        super().__init__()

        train_ids = open("data/train.data.txt", "r")
        train_labels = open("data/train.label.txt", "r")
        self.train_data = []
        self.train_label = []
        for train_ids_str, label in zip(train_ids.readlines(), train_labels.readlines()):
            train_ids_list = train_ids_str.strip().split(",")
            temp_json_list = []
            if not os.path.exists("data/train_tweet/" + train_ids_list[0] + ".json"):
                continue
            for train_id in train_ids_list:
                train_path = "data/train_tweet/" + train_id + ".json"
                if os.path.exists(train_path):
                    temp_json_list.append(json.load(open(train_path, "r")))
            # sort according to time
            temp_json_list = sorted(temp_json_list, key=lambda x: time.mktime(time.strptime(x["created_at"], '%Y-%m-%dT%H:%M:%S.%fZ')))
            self.train_data.append(temp_json_list)
            self.train_label.append(0 if label.strip() == "nonrumour" else 1)
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.n_obs = n_obs

    def __len__(self):
        if self.n_obs is None:
            return len(self.train_label)
        else:
            return self.n_obs

    def __getitem__(self, index):
        return self.train_data[index], self.train_label[index]

    def collate_fn(self, batch):
        input_text = []
        labels = []
        for x, label in batch:
            x_text = []
            for y in x:
                x_text.append(preprocess(y["text"]))
            input_text.append(self.tokenizer.sep_token.join(x_text))
            labels.append(label)

        src_text = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )

        batch_encoding = dict()
        batch_encoding["input_text"] = src_text.input_ids
        batch_encoding["attn_text"] = src_text.attention_mask
        batch_encoding["label"] = torch.LongTensor(labels)

        return batch_encoding


class DevDataset(Dataset):

    def __init__(
            self,
            tokenizer,
            max_length,
            n_obs=None,
    ):
        super().__init__()

        dev_ids = open("data/dev.data.txt", "r")
        dev_labels = open("data/dev.label.txt", "r")
        self.dev_data = []
        self.dev_label = []
        for dev_ids_str, label in zip(dev_ids.readlines(), dev_labels.readlines()):
            dev_ids_list = dev_ids_str.strip().split(",")
            temp_json_list = []
            if not os.path.exists("data/dev_tweet/" + dev_ids_list[0] + ".json"):
                continue
            for dev_id in dev_ids_list:
                dev_path = "data/dev_tweet/" + dev_id + ".json"
                if os.path.exists(dev_path):
                    temp_json_list.append(json.load(open(dev_path, "r")))
            # sort according to time
            temp_json_list = sorted(temp_json_list, key=lambda x: time.mktime(time.strptime(x["created_at"], '%Y-%m-%dT%H:%M:%S.%fZ')))
            self.dev_data.append(temp_json_list)
            self.dev_label.append(0 if label.strip() == "nonrumour" else 1)
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.n_obs = n_obs

    def __len__(self):
        if self.n_obs is None:
            return len(self.dev_label)
        else:
            return self.n_obs

    def __getitem__(self, index):
        return self.dev_data[index], self.dev_label[index]

    def collate_fn(self, batch):
        input_text = []
        labels = []
        for x, label in batch:
            x_text = []
            for y in x:
                x_text.append(preprocess(y["text"]))
            input_text.append(self.tokenizer.sep_token.join(x_text))
            labels.append(label)

        src_text = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )

        batch_encoding = dict()
        batch_encoding["input_text"] = src_text.input_ids
        batch_encoding["attn_text"] = src_text.attention_mask
        batch_encoding["label"] = torch.LongTensor(labels)

        return batch_encoding


class TestDataset(Dataset):

    def __init__(
        self,
        tokenizer,
        max_length,
        n_obs=None,
    ):
        super().__init__()

        test_ids = open("data/test.data.txt", "r")
        # test_ids = open("data/covid.data.txt", "r")
        self.test_data = []
        for test_ids_str in test_ids.readlines():
            test_ids_list = test_ids_str.strip().split(",")
            # if not os.path.exists("data/analysis_tweet/" + test_ids_list[0] + ".json"):
            #     continue
            temp_json_list = []
            for test_id in test_ids_list:
                # test_path = "data/analysis_tweet/" + test_id + ".json"
                test_path = "data/tweet-objects/" + test_id + ".json"
                if os.path.exists(test_path):
                    temp_json_list.append(json.load(open(test_path, "r")))

            # sort according to time
            temp_json_list = sorted(temp_json_list, key=lambda x: time.mktime(time.strptime(x["created_at"], '%a %b %d %H:%M:%S +0000 %Y')))
            # temp_json_list = sorted(temp_json_list, key=lambda x: time.mktime(time.strptime(x["created_at"], '%Y-%m-%dT%H:%M:%S.%fZ')))
            self.test_data.append(temp_json_list)

        self.max_length = max_length
        self.tokenizer = tokenizer
        self.n_obs = n_obs

    def __len__(self):
        if self.n_obs is None:
            return len(self.test_data)
        else:
            return self.n_obs

    def __getitem__(self, index):
        return self.test_data[index]

    def collate_fn(self, batch):
        input_text = []
        for x in batch:
            x_text = []
            for y in x:
                x_text.append(preprocess(y["text"]))
            input_text.append(self.tokenizer.sep_token.join(x_text))

        src_text = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )

        batch_encoding = dict()
        batch_encoding["input_text"] = src_text.input_ids
        batch_encoding["attn_text"] = src_text.attention_mask

        # origin_tweet = []
        # for x in batch:
        #     origin_tweet.append(json.dumps(x[0]))
        # batch_encoding["origin_tweet"] = origin_tweet
        return batch_encoding


def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)
