#!/usr/bin/env python3

# This file is to:
# 1. Download PERSONA-CHAT data.
# 2. Reorganize raw data and convert it to json format file.

import parlai.core.build_data as build_data
# Path can solve the annoying problem with Windows and Mac/Linux.
from pathlib import Path
import os
from parlai.core.build_data import DownloadableFile
import logging
import json
import csv
import re

logger = logging.getLogger(__file__)

RESOURCES = [
    DownloadableFile(
        'http://parl.ai/downloads/personachat/personachat.tgz',
        'personachat.tgz',
        '507cf8641d333240654798870ea584d854ab5261071c5e3521c20d8fa41d5622',
    )
]

def dowload_dataset():
    """Download PERSONA-CHAT"""

    # Create directory.
    dpath = Path.cwd().joinpath('data', 'raw')

    if not build_data.built(dpath):
        logger.info("Download PERSONA-CHAT to ./data/raw")
        Path.mkdir(dpath, exist_ok=True)

        # Download the data.
        for downloadable_file in RESOURCES:
            downloadable_file.download_file(dpath)

        # Mark the data as built.
        build_data.mark_done(dpath)

def read_txt(filename, dic, k):
    """read txt file into dict format"""
    logger.info("Generate ./data/json/impression_%s.json"%str(dic))
    with open(filename) as f:
        dic[k] = []
        dialog_yours = {}
        dialog_yours["personality"] = []
        dialog_yours["utterances"] = []
        dialog_partner = {}
        dialog_partner['personality'] = []
        dialog_partner["utterances"] = []
        for line in f:
            # the dataset is from speaker2's view
            if line.startswith("1 your persona") and len(dialog_yours["personality"]) > 0:
                dic[k].append(dialog_yours.copy())
                dic[k].append(dialog_partner.copy())
                dialog_yours["personality"] = []
                dialog_yours["utterances"] = []
                dialog_partner['personality'] = []
                dialog_partner["utterances"] = []
            # delete the line number
            line = re.sub('^\d+\s', '', line)
            if line.startswith("your persona"):
                # This is the start of a new dialog
                # I split one dialog into 2 parts, including your dialog utterances and your personality
                # and your partner's utterances and his/her personality.
                dialog_yours['personality'].append(line[14:-1])
            elif line.startswith("partner's persona"):
                dialog_partner['personality'].append(line[19:-1])
            else:
                utterances = line.split("\t")
                # print(utterances[0])
                dialog_partner["utterances"].append(utterances[0])
                dialog_yours["utterances"].append(utterances[1])


def convert_to_json():
    """Convert txt file to json format."""
    logger.info("Convert raw data to json format to ./data/json")
    # I use train_both_xxx.txt and valid_both_xxx.txt to construct dataset to learn impression
    # utterances -> personality.
    dpath = Path.cwd().joinpath('data', 'json')
    Path.mkdir(dpath, exist_ok=True)
    impression_original = {}
    read_txt("data/raw/personachat/train_both_original.txt", impression_original, "train")
    read_txt("data/raw/personachat/valid_both_original.txt", impression_original, "valid")
    impression_revised = {}
    read_txt("data/raw/personachat/train_both_revised.txt", impression_revised, "train")
    read_txt("data/raw/personachat/valid_both_revised.txt", impression_revised, "valid")
    logger.info("Write into data/json/impression_original.json")
    with open("data/json/impression_original.json", 'w') as f:
        json.dump(impression_original, f)
    logger.info("Write into data/json/impression_revised.json")
    with open("data/json/impression_revised.json", 'w') as f:
        json.dump(impression_revised, f)

def contruct_cse():
    dpath = Path.cwd().joinpath('data', 'csv')
    Path.mkdir(dpath, exist_ok=True)
    with open("./data/json/impression_original.json") as f:
        impression_original = json.load(f)
    ori_train = impression_original["train"]
    with open("./data/csv/oriforsimcse.csv", 'w') as f:
        writer = csv.writer(f)
        # write header
        writer.writerow(['sent0', 'sent1', 'hard_neg'])
        for i, sample in enumerate(ori_train):
            utt = ""
            for utterance in sample["utterances"]:
                utt += utterance
            for persona in sample["personality"]:
                writer.writerows([[utt, persona]])

if __name__ == '__main__':
    contruct_cse()