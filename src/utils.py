#!/usr/bin/env python
import pickle
import json


def pickle_save(obj, path):
    """pickle.dump(obj, path)"""
    with open(path, "wb") as f:
        return pickle.dump(obj, f)


def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, "w") as f:
        json.dump(content, f, indent=indent, **json_dump_kwargs)


def freeze_params(model, freeze_layer_count=8):
    """Set requires_grad=False for each of model.parameters()"""
    modules = [model.embeddings, *model.encoder.layer[:freeze_layer_count]]
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False
