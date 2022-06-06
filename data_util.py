import os
import tqdm
import shutil
from collections import defaultdict

import scipy.io

with open("imagenet_labels.txt", "r") as f:
    IMAGENET_CLASSES = list(map(str.rstrip, f.readlines()))

META = scipy.io.loadmat("meta.mat")

ILSVRC_TO_SYNID = {}
SYNSET_TO_INDEX = {}

with open("synset.txt", "r") as f:
    for idx, line in enumerate(f.readlines()):
        synset_tag = line.split(' ')[0]
        SYNSET_TO_INDEX[synset_tag] = idx

for i in range(1000):
    ilsvrc_id = META["synsets"][i][0][0][0][0]
    synset_tag = META["synsets"][i][0][1][0]
    good_id = SYNSET_TO_INDEX[synset_tag]
    ILSVRC_TO_SYNID[ilsvrc_id] = synset_tag


def generate_image_folder_format(folder_path: str, annotations_path: str, output_dir: str):
    with open(annotations_path, 'r') as f:
        annotations = list(map(int, f.readlines()))
    os.makedirs(output_dir, exist_ok=True)
    image_files = [x for x in os.listdir(folder_path) if x.endswith("JPEG")]
    for index, label in enumerate(tqdm.tqdm(annotations)):
        mapped_id = ILSVRC_TO_SYNID[label]
        image_output_dir = os.path.join(output_dir, mapped_id)
        os.makedirs(image_output_dir, exist_ok=True)
        shutil.copy(os.path.join(folder_path, image_files[index]), image_output_dir)


if __name__ == '__main__':
    generate_image_folder_format("validation", "ground_truth.txt", "imagenet_val")
