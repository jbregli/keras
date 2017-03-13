from boltons.iterutils import chunked_iter
from os.path import join as ojoin
import h5py
import json

from .image import Iterator


class H5FeatureIterator(Iterator):
    """
    mini-batch generator from a already shuffle h5 file.

    Note if you want shuffle, you have to do it when 
    you dump the feature. h5 format does not allow to slice 
    not contigus slice.
    """

    def __init__(self, feature_path,
                 split,
                 batch_size=None,
                 seed=None):
        self.feature_path = feature_path
        self.split = split
        self.hf = ojoin(feature_path, 'feature.h5')
        config = json.load(
            open(ojoin(feature_path, 'feature_config.json'), 'r'))
        self.__dict__.update(config)
        if batch_size is not None:
            self.batch_size = batch_size
        self.nb_sample = self.get_nsample(split)
        if self.batch_size > self.nb_sample:
            self.batch_size = self.nb_sample
            print('Set batch_size to {}'.format(self.nb_sample))
        # DONT CHANGE shuffle - reason in the docstring
        super(H5FeatureIterator, self).__init__(
            self.nb_sample, batch_size=self.batch_size, shuffle=False, seed=seed)

    def get_nsample(self, split):
        nsample = getattr(self, '{}_nb_sample'.format(split))
        return nsample

    def get(self, key, chunk):
        chunk = list(chunk)  # make sure this is a list
        with h5py.File(self.hf) as hf:
            if isinstance(hf[key], h5py.Dataset):
                arr = hf.get(key)[chunk]
            else:
                arr = dict.fromkeys(hf[key].keys())
                for key, value in hf[key].items():
                    arr[key] = value[chunk]
        return arr

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(
                self.index_generator)
        batch_x = self.get('X_{}'.format(self.split), index_array)
        batch_y = self.get('y_{}'.format(self.split), index_array)
        return batch_x, batch_y
