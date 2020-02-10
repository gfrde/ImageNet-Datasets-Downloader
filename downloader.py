
'''

example:
python ImageNet-Datasets-Downloader/downloader.py -data_root=/media/georg/Datasets/ImageNet/images/ -number_of_classes 20000 -images_per_class 20000 -scrape_only_flickr false -ignoreImageCount


'''

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

parser = argparse.ArgumentParser(description='ImageNet image scraper')
parser.add_argument('-scrape_only_flickr', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-number_of_classes', default = 10, type=int)
parser.add_argument('-images_per_class', default = 10, type=int)
parser.add_argument('-data_root', default='' , type=str)
parser.add_argument('-use_class_list', default=False,type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('-class_list', default=[], nargs='*')
parser.add_argument('-debug', default=False,type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("-ignoreImageCount", help="ignores the number of images", action="store_true")
parser.add_argument("-dryrun", help="don't download only simulate", action="store_true")

parser.add_argument('-multiprocessing_workers', default=50, type=int)

args, args_other = parser.parse_known_args()

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()

fileHandler = logging.FileHandler("{0}/{1}.log".format('./', 'imagenet_scarper.log'))
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)


if args.debug:
    rootLogger.setLevel(logging.DEBUG)
    # logging.basicConfig(level=logging.DEBUG)
    # logging.basicConfig(filename='imagenet_scarper.log', level=logging.DEBUG)
else:
    rootLogger.setLevel(logging.INFO)
    # logging.basicConfig(level=logging.INFO)
    # logging.basicConfig(filename='imagenet_scarper.log', level=logging.INFO)

if len(args.data_root) == 0:
    logging.error("-data_root is required to run downloader!")
    exit()

logging.info('dest: ' + args.data_root)
if not os.path.isdir(args.data_root):
    logging.error('folder {dr} does not exist! please provide existing folder in -data_root arg!'.format(dr=args.data_root))
    exit()


IMAGENET_API_WNID_TO_URLS = lambda wnid: 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=%s' % (wnid,)

current_folder = os.path.dirname(os.path.realpath(__file__))

class_info_json_filename = 'imagenet_class_info.json'
class_info_json_filepath = os.path.join(current_folder, class_info_json_filename)

class_info_dict = dict()

with open(class_info_json_filepath) as class_info_json_f:
    class_info_dict = json.load(class_info_json_f)

logging.info('number of available classes: ' + str(len(class_info_dict)))
classes_to_scrape = []

if args.use_class_list == True:
   for item in args.class_list:
       classes_to_scrape.append(item)
       if item not in class_info_dict:
           logging.error('Class %s not found in ImageNet' % (item,) )
           exit()

elif args.use_class_list == False:
    potential_class_pool = []
    for key, val in class_info_dict.items():

        if args.scrape_only_flickr:
            if int(val['flickr_img_url_count']) * 0.9 > args.images_per_class:
                potential_class_pool.append(key)
            elif args.ignoreImageCount:
                potential_class_pool.append(key)
        else:
            if int(val['img_url_count']) * 0.8 > args.images_per_class:
                potential_class_pool.append(key)
            elif args.ignoreImageCount:
                potential_class_pool.append(key)

    if (len(potential_class_pool) < args.number_of_classes):
        logging.warning("With %s images per class there are %d to choose from." % (args.images_per_class, len(potential_class_pool), ) )
        logging.warning("Decrease number of classes or decrease images per class.")

        if not args.ignoreImageCount:
            exit(-1)

    picked_classes_idxes = np.random.choice(len(potential_class_pool), args.number_of_classes, replace = False)

    for idx in picked_classes_idxes:
        classes_to_scrape.append(potential_class_pool[idx])


print("Picked the following classes:")
print([ class_info_dict[class_wnid]['class_name'] for class_wnid in classes_to_scrape ])

imagenet_images_folder = os.path.join(args.data_root, 'imagenet_images')
if not os.path.isdir(imagenet_images_folder):
    os.mkdir(imagenet_images_folder)


scraping_stats = dict(
    all=dict(
        tried=0,
        success=0,
        time_spent=0,
    ),
    is_flickr=dict(
        tried=0,
        success=0,
        time_spent=0,
    ),
    not_flickr=dict(
        tried=0,
        success=0,
        time_spent=0,
    )
)

def add_debug_csv_row(row):
    with open('stats.csv', "a") as csv_f:
        csv_writer = csv.writer(csv_f, delimiter=",")
        csv_writer.writerow(row)

class MultiStats():
    def __init__(self):

        self.lock = Lock()

        self.stats = dict(
            all=dict(
                tried=Value('d', 0),
                success=Value('d',0),
                time_spent=Value('d',0),
            ),
            is_flickr=dict(
                tried=Value('d', 0),
                success=Value('d',0),
                time_spent=Value('d',0),
            ),
            not_flickr=dict(
                tried=Value('d', 0),
                success=Value('d', 0),
                time_spent=Value('d', 0),
            )
        )
    def inc(self, cls, stat, val):
        with self.lock:
            self.stats[cls][stat].value += val

    def get(self, cls, stat):
        with self.lock:
            ret = self.stats[cls][stat].value
        return ret

multi_stats = MultiStats()


if args.debug:
    row = [
        "all_tried",
        "all_success",
        "all_time_spent",
        "is_flickr_tried",
        "is_flickr_success",
        "is_flickr_time_spent",
        "not_flickr_tried",
        "not_flickr_success",
        "not_flickr_time_spent"
    ]
    add_debug_csv_row(row)


def add_stats_to_debug_csv():
    row = [
        multi_stats.get('all', 'tried'),
        multi_stats.get('all', 'success'),
        multi_stats.get('all', 'time_spent'),
        multi_stats.get('is_flickr', 'tried'),
        multi_stats.get('is_flickr', 'success'),
        multi_stats.get('is_flickr', 'time_spent'),
        multi_stats.get('not_flickr', 'tried'),
        multi_stats.get('not_flickr', 'success'),
        multi_stats.get('not_flickr', 'time_spent'),
    ]
    add_debug_csv_row(row)


def print_stats(cls, print_func):

    actual_all_time_spent = time.time() - scraping_t_start.value
    processes_all_time_spent = multi_stats.get('all', 'time_spent')

    if processes_all_time_spent == 0:
        actual_processes_ratio = 1.0
    else:
        actual_processes_ratio = actual_all_time_spent / processes_all_time_spent

    #print(f"actual all time: {actual_all_time_spent} proc all time {processes_all_time_spent}")

    print_func('STATS For class {cls}:'.format(cls=cls))
    print_func(' tried {tried} urls with'.format(tried=multi_stats.get(cls, "tried"))
               +' {success} successes'.format(success=multi_stats.get(cls, "success")) )

    if multi_stats.get(cls, "tried") > 0:
        print_func('{rate}% success rate for {cls} urls '.format(rate=100.0 * multi_stats.get(cls, "success")/multi_stats.get(cls, "tried"), cls=cls))
    if multi_stats.get(cls, "success") > 0:
        print_func('{secs} seconds spent per {cls} succesful image download'.format(secs=multi_stats.get(cls,"time_spent") * actual_processes_ratio / multi_stats.get(cls,"success"), cls=cls))


def is_ascii(s):
    try:
        s.encode().decode('ascii')
    except UnicodeDecodeError:
        return False
    return True
    # return all(ord(c) < 128 for c in s)


lock = Lock()
url_tries = Value('d', 0)
scraping_t_start = Value('d', time.time())
class_folder = ''
class_images = Value('d', 0)


def get_image(img_url):
    #print(f'Processing {img_url}')
    if len(img_url) <= 1:
        return
    cls_imgs = 0
    with lock:
        cls_imgs = class_images.value

    if cls_imgs >= args.images_per_class:
        return

    logging.debug(img_url)

    cls = ''

    if 'flickr' in img_url:
        cls = 'is_flickr'
    else:
        cls = 'not_flickr'
        if args.scrape_only_flickr:
            return

    img_name = img_url.split('/')[-1]
    img_name = img_name.split("?")[0]

    if img_name.find('..') != -1 or img_name.find('/') != -1 or img_name.find('\\') != -1 or len(img_name)>200 or not is_ascii(img_name):
        sha_1 = hashlib.sha1()
        sha_1.update(img_name.encode())
        s = sha_1.hexdigest()
        s += '.' + img_name.split('.')[-1]
        logging.info('renaming file: %s --> %s' % (img_name, s,))
        img_name = s


    t_start = time.time()

    def finish(status):
        t_spent = time.time() - t_start
        multi_stats.inc(cls, 'time_spent', t_spent)
        multi_stats.inc('all', 'time_spent', t_spent)

        multi_stats.inc(cls,'tried', 1)
        multi_stats.inc('all', 'tried', 1)

        if status == 'success':
            multi_stats.inc(cls,'success', 1)
            multi_stats.inc('all', 'success', 1)

        elif status == 'failure':
            pass
        else:
            logging.error('No such status {status}!!'.format(status=status))
            exit()
        return


    with lock:
        url_tries.value += 1
        if url_tries.value % 250 == 0:
            print('\nScraping stats:')
            print_stats('is_flickr', print)
            print_stats('not_flickr', print)
            print_stats('all', print)
            if args.debug:
                add_stats_to_debug_csv()

    if (len(img_name) <= 1):
        return finish('failure')

    localImgageName = img_name.lower()

    link_file_path = os.path.join(class_folder, localImgageName)
    if os.path.exists(link_file_path):
        with lock:
            class_images.value += 1
        logging.debug('file already downloaded: ' + localImgageName)
        return finish('success')

    relLink = '../../realfiles/' + localImgageName[0]
    mainImgFolder = os.path.join(class_folder, relLink)
    os.makedirs(mainImgFolder, exist_ok=True)
    img_file_path = os.path.join(mainImgFolder, localImgageName)
    rel_file_path = os.path.join(relLink, localImgageName)
    if os.path.exists(img_file_path):
        with lock:
            class_images.value += 1
        logging.debug('file already downloaded but not linked: ' + localImgageName)
        if not os.path.exists(link_file_path):
            os.symlink(rel_file_path, link_file_path)
        return finish('success')

    try:
        img_resp = requests.get(img_url, timeout = 1)
    except ConnectionError:
        logging.debug("Connection Error for url " + img_url)
        return finish('failure')
    except ReadTimeout:
        logging.debug("Read Timeout for url " + img_url)
        return finish('failure')
    except TooManyRedirects:
        logging.debug("Too many redirects " + img_url)
        return finish('failure')
    except MissingSchema:
        return finish('failure')
    except InvalidURL:
        return finish('failure')
    except UnicodeDecodeError as err:
        logging.warning('Unicode error for "%s": %s' % (img_url, str(err),) )
        return finish('failure')
    except UnicodeError as err:
        logging.warning('Unicode error for "%s": %s' % (img_url, str(err),) )
        return finish('failure')
    except requests.exceptions.ContentDecodingError as err:
        logging.warning('Content-problem error for "%s": %s' % (img_url, str(err),) )
        return finish('failure')
    except requests.exceptions.InvalidSchema as err:
        logging.warning('Invalid schema for "%s": %s' % (img_url, str(err),) )
        return finish('failure')
    except requests.exceptions.ChunkedEncodingError as err:
        logging.warning('Invalid chunks for "%s": %s' % (img_url, str(err),) )
        return finish('failure')

    if not 'content-type' in img_resp.headers:
        return finish('failure')

    if not 'image' in img_resp.headers['content-type']:
        logging.debug("Not an image")
        return finish('failure')

    if (len(img_resp.content) < 1000):
        return finish('failure')

    logging.debug(img_resp.headers['content-type'])
    logging.debug('image size ' + str(len(img_resp.content)))

    logging.debug('Saving image in ' + img_file_path)

    with open(img_file_path, 'wb') as img_f:
        img_f.write(img_resp.content)

        with lock:
            class_images.value += 1

        logging.debug('Scraping stats')
        print_stats('is_flickr', logging.debug)
        print_stats('not_flickr', logging.debug)
        print_stats('all', logging.debug)

        if not os.path.exists(link_file_path):
            os.symlink(rel_file_path, link_file_path)

        return finish('success')

cnt = 0
for class_wnid in classes_to_scrape:

    cnt += 1
    class_name = class_info_dict[class_wnid]["class_name"]
    logging.info('********************** (% 3d/% 4d)   Scraping images for class: %s' % (cnt, len(classes_to_scrape), class_name,) )
    url_urls = IMAGENET_API_WNID_TO_URLS(class_wnid)

    time.sleep(0.05)
    resp = requests.get(url_urls)

    class_folder = os.path.join(imagenet_images_folder, class_wnid + '___' + class_name)
    if not os.path.exists(class_folder):
        os.mkdir(class_folder)

    class_images.value = 0

    urls = [url.decode('utf-8') for url in resp.content.splitlines()]
    logging.info('number of images for class: ' + str(len(urls)))

    logging.info("  Multiprocessing workers: {w}".format(w=args.multiprocessing_workers))
    with Pool(processes=args.multiprocessing_workers) as p:
        p.map(get_image, urls)
