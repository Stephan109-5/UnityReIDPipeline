import functools
import os.path as osp
import pickle

import cv2
import numpy as np
import torch
import torchreid
import torch.nn.functional as F
from torchreid.reid.utils import FeatureExtractor
from torchreid.reid.utils import torchtools as torchtools_module

def init_reid_feature_extractor(model_name: str = 'osnet_x1_0', model_path: str = None) -> FeatureExtractor:
    model_name = model_name.lower()

    # Bypass pytorch 2.6+ checkpoint loading issues by monkey-patching torchtools.load_checkpoint
    def load_checkpoint_compat(fpath):
        if fpath is None:
            raise ValueError('File path is None')

        fpath = osp.abspath(osp.expanduser(fpath))
        if not osp.exists(fpath):
            raise FileNotFoundError('File is not found at "{}"'.format(fpath))

        map_location = None if torch.cuda.is_available() else 'cpu'
        try:
            checkpoint = torch.load(fpath, map_location=map_location, weights_only=False)
        except TypeError:
            checkpoint = torch.load(fpath, map_location=map_location)
        except UnicodeDecodeError:
            pickle.load = functools.partial(pickle.load, encoding='latin1')
            pickle.Unpickler = functools.partial(pickle.Unpickler, encoding='latin1')
            try:
                checkpoint = torch.load(fpath, pickle_module=pickle, map_location=map_location, weights_only=False)
            except TypeError:
                checkpoint = torch.load(fpath, pickle_module=pickle, map_location=map_location)
        except Exception:
            print('Unable to load checkpoint from "{}"'.format(fpath))
            raise

        return checkpoint

    original_torchtools_load_checkpoint = torchtools_module.load_checkpoint

    torchtools_module.load_checkpoint = load_checkpoint_compat
    try:
        return FeatureExtractor(model_name=model_name, model_path=model_path, device='cuda' if torch.cuda.is_available() else 'cpu')
    finally:
        torchtools_module.load_checkpoint = original_torchtools_load_checkpoint