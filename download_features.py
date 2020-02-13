

import os
import numpy as np
import requests
import argparse
import json
import time
import logging
import csv
import hashlib

from multiprocessing import Pool, Process, Value, Lock

from requests.exceptions import ConnectionError, ReadTimeout, TooManyRedirects, MissingSchema, InvalidURL


IMAGENET_URL_SYNSETS = 'http://www.image-net.org/api/text/imagenet.sbow.obtain_synset_list'
METADIR = os.path.join('./', 'metadata')
SBOX_LIST = os.path.join(METADIR, 'sbox_list.txt')
SBOX_DATADIR = os.path.join(METADIR, 'sbox_data')
# IMAGENET_SBOX_DOWNLOAD = 'http://www.image-net.org/api/download/imagenet.sbow.synset?wnid=%s'
IMAGENET_SBOX_DOWNLOAD = 'http://www.image-net.org/downloads/features/sbow/%s.sbow.mat'


parser = argparse.ArgumentParser(description='ImageNet image scraper')
parser.add_argument('-debug', default=False,type=lambda x: (str(x).lower() == 'true'))
# parser.add_argument('-scrape_only_flickr', default=True, type=lambda x: (str(x).lower() == 'true'))

args, args_other = parser.parse_known_args()

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()

fileHandler = logging.FileHandler("{0}/{1}.log".format('./', 'imagenet_meta.log'))
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)


if args.debug:
    rootLogger.setLevel(logging.DEBUG)
else:
    rootLogger.setLevel(logging.INFO)

if not os.path.exists(METADIR):
    os.mkdir(METADIR)

if not os.path.exists(SBOX_LIST):
    logging.info('need to download the list')
    time.sleep(0.05)
    resp = requests.get(IMAGENET_URL_SYNSETS)
    with open(SBOX_LIST, 'wb') as f:
        f.write(resp.content)
else:
    logging.info('using existing list')

os.makedirs(SBOX_DATADIR, exist_ok=True)
widlist = []
with open(SBOX_LIST, 'r') as f:
    for l in f:
        # logging.info(l)
        if len(l) == 0:
            continue
        widlist.append(l.strip())

cnt = 0;
for wid in widlist:
    cnt += 1
    fn = os.path.join(SBOX_DATADIR, wid+'.sbow.mat')
    if os.path.exists(fn):
        continue

    logging.info('downloading % 3d / %d' % (cnt, len(widlist),))
    d = IMAGENET_SBOX_DOWNLOAD % (wid,)
    logging.info(d)

    resp = requests.get(d)
    if resp.status_code != 200:
        logging.error('error when download ' + d)
        continue
    with open(fn, 'wb') as f:
        f.write(resp.content)
    time.sleep(0.5)

