#!/usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import
from cmath import pi
__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

# To avoid costly TF1 -> Tf2 migration
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# /end

# import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

import argparse

from classes.dataset.Generator import *
from classes.model.pix2code import *
from classes.model_torch.model_wrapper import Pix2CodeWrapper


def run(input_path, output_path, is_memory_intensive=False, pretrained_model=None, pytoch=False):
    np.random.seed(1234)

    dataset = Dataset()
    dataset.load(input_path, generate_binary_sequences=True)
    dataset.save_metadata(output_path)
    dataset.voc.save(output_path)

    if not is_memory_intensive:
        dataset.convert_arrays()

        input_shape = dataset.input_shape
        output_size = dataset.output_size
        #print(len(dataset.input_images), len(dataset.partial_sequences), len(dataset.next_words))
        #print(dataset.input_images.shape, dataset.partial_sequences.shape, dataset.next_words.shape)
    else:
        gui_paths, img_paths = Dataset.load_paths_only(input_path)

        input_shape = dataset.input_shape
        output_size = dataset.output_size
        steps_per_epoch = dataset.size / BATCH_SIZE

        voc = Vocabulary()
        voc.retrieve(output_path)

        generator = Generator.data_generator(voc, gui_paths, img_paths, batch_size=BATCH_SIZE, generate_binary_sequences=True)

    ModelImpl = Pix2CodeWrapper if pytoch else pix2code
    model = ModelImpl(input_shape, output_size, output_path)

    if pretrained_model is not None:
        model.load_weights(pretrained_model)

    if not is_memory_intensive:
        model.fit(dataset.input_images, dataset.partial_sequences, dataset.next_words)
    else:
        model.fit_generator(generator, steps_per_epoch=steps_per_epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train model')

    parser.add_argument('-i', '--input-path', required=True, type=str, help='input path')
    parser.add_argument('-o', '--output-path', required=True, type=str, help='output path')
    parser.add_argument('-m', '--is-memory-intensive', required=False, action='store_true', help='is memory intensive')
    parser.add_argument('-w', '--pretrained-weights', required=False, type=str, help='pretrained weights')
    parser.add_argument('--pytorch', required=False, action='store_true', help='use pytorch model for training')

    args = parser.parse_args()

    run(args.input_path, args.output_path, args.is_memory_intensive, args.pretrained_weights, args.pytorch)