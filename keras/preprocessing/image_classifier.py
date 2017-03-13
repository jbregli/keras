'''
Contains a set of class/method that are built on top of keras
library.

'''
from .image import *
from tqdm import *
import numpy as np
import os
from skimage.exposure import adjust_log
from keras.preprocessing.image import *
from skimage import exposure
from PIL import Image, ImageDraw
import itertools
import random
import pandas as pd


class ClassifierImageGenerator(ImageDataGenerator):
    '''
    Customize keras implementation to add the preprocess
    of the paper on VGG16.
    '''

    def __init__(self, add_channel=False, *args, **kwargs):
        super(ClassifierImageGenerator, self).__init__(*args, **kwargs)
        self.add_channel = add_channel
        self.bmodel_preprocessing = bmodel_preprocessing
        if bmodel_preprocessing is not None:
            assert hasattr(bmodel, self.bmodel_preprocessing), '{} preprocessing  not available'.format(
                self.bmodel_preprocessing)

    def standardize(self, x):
        if self.rescale:
            x *= self.rescale
        # x is a single image, so it doesn't have image number at index 0
        img_channel_index = self.channel_index - 1
        grayscale = x.shape[img_channel_index] == 1
        if self.samplewise_center:
            if grayscale:
                raise ValueError(
                    'samplewise_center on a grey image does not make sense')
            x -= np.mean(x, axis=img_channel_index, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, axis=img_channel_index, keepdims=True) + 1e-7)

        if self.featurewise_center:
            x -= self.mean
        if self.featurewise_std_normalization:
            x /= (self.std + 1e-7)

        if self.zca_whitening:
            flatx = np.reshape(x, (x.size))
            whitex = np.dot(flatx, self.principal_components)
            x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))
        # Add our custome stuff
        # vgg prerpocessing
        if self.bmodel_preprocessing is not None:
            module = getattr(bmodel, self.bmodel_preprocessing.lower())
            s = x.shape
            x = np.expand_dims(x, axis=0)
            x = getattr(module, 'preprocess_input')(x)
            x = x.reshape(s)

        # add_channel to gray images
        if x.shape[img_channel_index] == 1:
            x = x.transpose(2, 0, 1)
            if self.add_channel:
                log = exposure.adjust_log(x)
                hist = exposure.equalize_adapthist(
                    x[0].astype('int16'))[np.newaxis]
                x = np.vstack((log, hist, x)).transpose(1, 2, 0)
            else:
                x = np.vstack(([x] * 3)).transpose(1, 2, 0)
        return x

    def flow_from_directory(self, directory,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_every=None,
                            save_prefix='',
                            save_format='jpeg'):
        return ClassifierDirectoryIterator(
            directory, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            dim_ordering=self.dim_ordering,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            save_every=save_every)


def try_until_success(func):
    def func_wrapper(*args, **kwargs):
        res = 0
        c = 0
        while res == 0:
            try:
                x = func(*args, **kwargs)
                res = 1
            except:
                c += 1
            if c > 10:
                raise Exception('Sorry, we try hard but did not get anything')
        return x
    return func_wrapper


class ClassifierDirectoryIterator(Iterator):
    '''
    Add a color model grayscale augmented and buil the image generator on top
    of that.

    WHY ?? Because if you specify the image is a grayscale then, the generator
    can output only single channel images.

    save_every is a probability to not save every images.
    '''

    def __init__(self, directory,
                 image_data_generator,
                 target_size=(256, 256),
                 color_mode='rgb',
                 dim_ordering='default',
                 classes=None,
                 class_mode='categorical',
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='jpeg',
                 save_every=None):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.save_every = save_every
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale', 'grayscale_augmented'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale", "grayscale_augmented".')
        self.color_mode = color_mode
        self.dim_ordering = dim_ordering
        if self.color_mode in ['rgb', 'grayscale_augmented']:
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}

        # first, count the number of samples and classes
        self.nb_sample = 0

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)
        self.nb_class = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        for subdir in classes:
            subpath = os.path.join(directory, subdir)
            for fname in sorted(os.listdir(subpath)):
                is_valid = False
                for extension in white_list_formats:
                    if fname.lower().endswith('.' + extension):
                        is_valid = True
                        break
                if is_valid:
                    self.nb_sample += 1
        print('Found %d images belonging to %d classes.' %
              (self.nb_sample, self.nb_class))

        # second, build an index of the images in the different class
        # subfolders
        self.filenames = []
        self.classes = np.zeros((self.nb_sample,), dtype='int32')
        i = 0
        for subdir in classes:
            subpath = os.path.join(directory, subdir)
            for fname in sorted(os.listdir(subpath)):
                is_valid = False
                for extension in white_list_formats:
                    if fname.lower().endswith('.' + extension):
                        is_valid = True
                        break
                if is_valid:
                    self.classes[i] = self.class_indices[subdir]
                    self.filenames.append(os.path.join(subdir, fname))
                    i += 1
        super(ClassifierDirectoryIterator, self).__init__(
            self.nb_sample, batch_size, shuffle, seed)

    # @try_until_success
    def get_batch_x(self, j):
        fname = self.filenames[j]
        grayscale = self.color_mode == 'grayscale' or self.color_mode == 'grayscale_augmented'
        img = load_img(os.path.join(self.directory, fname),
                       grayscale=grayscale, target_size=self.target_size)
        x = img_to_array(img, dim_ordering=self.dim_ordering)
        x = self.image_data_generator.random_transform(x)
        x = self.image_data_generator.standardize(x)
        return x

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(
                self.index_generator)
        # The transformation of images is not under thread lock so it can be
        # done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape)
        # build batch of image data
        for i, j in enumerate(index_array):
            x = self.get_batch_x(j)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                if self.save_every is not None:
                    p = np.random.rand()
                    if p > self.save_every:
                        break
                arr = np.copy(batch_x[i])
                img = array_to_img(arr, self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(
                                                                      1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        batch_y = None
        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype('float32')
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.nb_class), dtype='float32')
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x

        return batch_x, batch_y
