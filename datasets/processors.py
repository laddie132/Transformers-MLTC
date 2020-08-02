#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Han"
__email__ = "liuhan132@foxmail.com"

import json
import os

from tqdm import tqdm
from transformers.data import DataProcessor

from .utils import InputExample


class AAPDProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, 'text_train'),
                                     os.path.join(data_dir, 'label_train'), 'train')

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, 'text_val'),
                                     os.path.join(data_dir, 'label_val'), 'dev')

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, 'text_test'),
                                     os.path.join(data_dir, 'label_test'), 'test')

    def get_labels(self):
        """See base class."""
        return [
            "cmp-lg",
            "cond-mat.dis-nn",
            "cond-mat.stat-mech",
            "cs.AI",
            "cs.CC",
            "cs.CE",
            "cs.CG",
            "cs.CL",
            "cs.CR",
            "cs.CV",
            "cs.CY",
            "cs.DB",
            "cs.DC",
            "cs.DL",
            "cs.DM",
            "cs.DS",
            "cs.FL",
            "cs.GT",
            "cs.HC",
            "cs.IR",
            "cs.IT",
            "cs.LG",
            "cs.LO",
            "cs.MA",
            "cs.MM",
            "cs.MS",
            "cs.NA",
            "cs.NE",
            "cs.NI",
            "cs.PF",
            "cs.PL",
            "cs.RO",
            "cs.SC",
            "cs.SE",
            "cs.SI",
            "cs.SY",
            "math.CO",
            "math.IT",
            "math.LO",
            "math.NA",
            "math.NT",
            "math.OC",
            "math.PR",
            "math.ST",
            "nlin.AO",
            "physics.data-an",
            "physics.soc-ph",
            "q-bio.NC",
            "q-bio.QM",
            "quant-ph",
            "stat.AP",
            "stat.ME",
            "stat.ML",
            "stat.TH"
        ]

    def _create_examples(self, text_path, label_path, set_type):
        """Creates examples for the training and dev sets."""

        examples = []

        i = 0
        with open(text_path, 'r') as textf, open(label_path, 'r') as lf:
            for text, label in zip(textf, lf):
                if text != '' and label != '':
                    text = text.strip()
                    label = label.strip().split()

                    guid = "%s-%s" % (set_type, i)
                    text_a = text

                    examples.append(
                        InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
                i += 1

        return examples


class RMSCProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, 'rmsc.data.train.json'), 'train')

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, 'rmsc.data.valid.json'), 'dev')

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, 'rmsc.data.test.json'), 'test')

    def get_labels(self):
        """See base class."""
        return [
            "alternative",
            "britpop",
            "classical",
            "country",
            "darkwave",
            "electronic",
            "folk",
            "hiphop",
            "indie",
            "jazz",
            "jpop",
            "metal",
            "newage",
            "ost",
            "piano",
            "pop",
            "postpunk",
            "postrock",
            "punk",
            "r&b",
            "rock",
            "soul"
        ]

    def _create_examples(self, data_path, set_type):
        """Creates examples for the training and dev sets."""

        examples = []

        with open(data_path, 'r') as textf:
            data = json.load(textf)

        for i, sample in tqdm(enumerate(data), desc='loading data...'):
            guid = "%s-%s" % (set_type, i)
            label = sample['tags']
            text_a = '[SEP]'.join([item['comment'] for item in sample['all_short_comments']])

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        return examples


mltc_processors = {
    "aapd": AAPDProcessor,
    "rmsc": RMSCProcessor,
}

mltc_tasks_num_labels = {
    "aapd": 54,
    "rmsc": 22,
}
