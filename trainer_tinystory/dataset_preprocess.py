import os
import json
import glob
import string
from tqdm import tqdm

import multiprocessing

DATA_CACHE_DIR = './'

def json2txt(num_shards=50):
    txt_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_txt")
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    json_file = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    if not os.path.exists(txt_dir):
        os.mkdir(txt_dir)

    for shard in tqdm(json_file[:]):
        shard_txt = shard.replace('.json', '.txt')
        shard_txt = shard_txt.replace('TinyStories_all_data', 'TinyStories_txt')
        shard_txt = shard_txt.replace('data', 'tinystory')
        with open(shard_txt, "w", encoding="utf-8") as of:
            with open(shard, 'r') as f:
                data = json.load(f)
            for example in data:
                text = example['story']
                text = text.strip()
                of.write(text + '\n')

def DeletePunctuation():
    txt_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_txt")
    txt_file = sorted(glob.glob(os.path.join(txt_dir, "*.txt")))

    delete_punctuation = str.maketrans('', '', string.punctuation)

    for shard in tqdm(txt_file[:]):
        shard_txt = shard.replace('.txt', '_wop.txt')
        with open(shard_txt, "w") as of:
            with open(shard, 'r') as f:
                data = f.read()
            no_punctuation = data.translate(delete_punctuation)
            translated_data = no_punctuation.replace('\n', '.\n')
            of.write(translated_data)

def txt2phonemes_unit(txt, phoneme):
    os.system('espeak -x -q -s 1000 -f {txt} >> {phoneme}'.format(txt=txt, phoneme=phoneme))


if __name__ == "__main__":
    json2txt()
    DeletePunctuation()

    text_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_txt")
    text_file = sorted(glob.glob(os.path.join(text_dir, "*_wop.txt")))
    phonemes_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_phonemes")
    phonemes_file = []

    if not os.path.exists(phonemes_dir):
        os.mkdir(phonemes_dir)

    for shard in text_file:
        tmp = shard.replace('TinyStories_txt', 'TinyStories_phonemes')
        tmp = tmp.replace('_wop.txt', '.txt')
        phonemes_file.append(tmp)

    phonemes_file = sorted(phonemes_file)
    pair = list(zip(text_file, phonemes_file))

    pool = multiprocessing.Pool()
    pool.starmap(txt2phonemes_unit, pair)
    pool.close()