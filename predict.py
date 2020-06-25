"""
@author: Signatrix GmbH
Implementation of paradigm described in paper: Designing Network Design Spaces published by Facebook AI Research (FAIR)
"""

import os
import cv2
import torch
import argparse
import pandas as pd
from src.transforms import *
from src.regnet import RegNetY
from collections import OrderedDict
from torch.nn import Softmax
from src.config import TRAIN_IMAGE_SIZE, TEST_IMAGE_SIZE


def get_args():
    parser = argparse.ArgumentParser(
        description="Implementation of paradigm described in paper: Designing Network Design Spaces published by Facebook AI Research (FAIR)")

    parser.add_argument("-d", "--test_data_path", type=str, default="test", help="the root folder of test images")
    parser.add_argument("--model_path", type=str, help="trained model weights path")

    # These default parameters are for RegnetY 200MF
    parser.add_argument("--bottleneck_ratio", default=1, type=int)
    parser.add_argument("--group_width", default=8, type=int)
    parser.add_argument("--initial_width", default=24, type=int)
    parser.add_argument("--slope", default=36, type=float)
    parser.add_argument("--quantized_param", default=2.5, type=float)
    parser.add_argument("--network_depth", default=13, type=int)
    parser.add_argument("--stride", default=2, type=int)
    parser.add_argument("--se_ratio", default=4, type=int)

    args = parser.parse_args()
    return args

def rename_state_dict_keys(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_key = key.split("module.")[1]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

def preprocess(img):
    img = scale(TEST_IMAGE_SIZE, img)
    img = center_crop(TRAIN_IMAGE_SIZE, img)
    img = img.transpose(2, 0, 1) / 255
    img = color_norm(img, [0.406, 0.456, 0.485], [0.225, 0.224, 0.229])
    return img

def evaluate(opt, batch_size=1):
    filenames = os.listdir(opt.test_data_path)

    model = RegNetY(opt.initial_width, opt.slope, opt.quantized_param, opt.network_depth, opt.bottleneck_ratio,
                    opt.group_width, opt.stride, opt.se_ratio)

    if torch.cuda.is_available():
        model = model.cuda()

    checkpoint = torch.load(opt.model_path)
    state_dict = rename_state_dict_keys(checkpoint["state_dict"])
    model.load_state_dict(state_dict)
    model.eval()

    # load all images
    print('loading files')
    list_of_imgs = [ cv2.imread(os.path.join(opt.test_data_path, file)) for file in filenames ]

    # preprocess
    print('preprocessing')
    list_of_imgs = [ preprocess(img) for img in list_of_imgs ]
    images = np.stack(list_of_imgs, axis=0)
    images = torch.from_numpy(images)

    batches = []
    batch_filenames = []
    for i in range(0, len(images), batch_size):
        these_imgs = images[i:i+batch_size]
        batches.append(these_imgs)
        these_filenames = filenames[i:i+batch_size]
        batch_filenames.append(these_filenames)
    
    softmax = Softmax(1)
    print('starting inferences')
    output_dict = {}
    with torch.no_grad():
        for i, batch in enumerate(batches):
            batch = batch.float().cuda()
            logits = model(batch)
            outputs = softmax(logits)
            # print('softmax vector: {}'.format(outputs))
            _, preds = torch.max(outputs, 1)
            output_classes = preds.data.cpu().numpy()
            # print('output_classes: {}'.format(output_classes))

            # zero pad the labels
            output_classes=["%02d" % x for x in output_classes]

            # print('batch_filenames: {}'.format(batch_filenames[i]))
            for filename, output_cls in zip(batch_filenames[i], output_classes):
                output_dict[filename] = output_cls
    # print('output dict: {}'.format(output_dict))

    # output to csv
    output_df = pd.DataFrame(output_dict.items(), columns=['filename', 'category'])
    # print('output df:\n{}'.format(output_df))
    output_df.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    opt = get_args()
    evaluate(opt, batch_size=10)
