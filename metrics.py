#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import torch
import torch.nn

eps = 1e-7


def evaluate_f1_ml(predict, truth):
    """
    F1-score for multi-label classification
    :param predict: (batch, labels)
    :param truth: (batch, labels)
    :return:
    """
    label_same = []
    label_predict = []
    label_truth = []
    label_f1 = []

    division = lambda x, y: (x * 1.0 / y) if y else 0
    f1 = lambda p, r: 2 * p * r / (p + r) if p + r else 0

    batch, label_size = predict.size()
    for i in range(label_size):
        cur_predict = predict[:, i]
        cur_truth = truth[:, i]

        predict_max = cur_predict.gt(0.5).long()
        cur_eq_num = (predict_max * cur_truth).sum().item()

        cur_predict_num = predict_max.sum().item()
        cur_truth_num = cur_truth.sum().item()

        cur_precision = division(cur_eq_num, cur_predict_num)
        cur_recall = division(cur_eq_num, cur_truth_num)
        cur_f1 = f1(cur_precision, cur_recall)

        label_same.append(cur_eq_num)
        label_predict.append(cur_predict_num)
        label_truth.append(cur_truth_num)
        label_f1.append(cur_f1)

    macro_f1 = sum(label_f1) / len(label_f1)
    micro_precision = division(sum(label_same), sum(label_predict))
    micro_recall = division(sum(label_same), sum(label_truth))
    micro_f1 = f1(micro_precision, micro_recall)

    return macro_f1, micro_f1, micro_precision, micro_recall, label_f1


def evaluate_hamming_loss(predict, truth):
    """
    Hamming Loss for multi-label evaluation
    :param predict:
    :param truth:
    :return:
    """
    predict_max = predict.gt(0.5).long()

    batch_eq_num = torch.ne(predict_max, truth).long().sum().item()
    batch_num, label_num = predict_max.shape

    return batch_eq_num * 1.0 / (batch_num * label_num)


def evaluate_one_error(predict, truth):
    _, max_label = predict.max(dim=-1)
    max_label = max_label.unsqueeze(-1)
    predict_max = torch.zeros_like(truth).scatter_(dim=-1, index=max_label, value=1)

    batch_eq_num = (predict_max * truth).sum().item()
    batch = truth.shape[0]

    return (batch - batch_eq_num) * 1.0 / batch