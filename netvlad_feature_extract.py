import cv2
import h5py
import numpy as np
from pathlib import Path

from hfnet.models import get_model
from hfnet.settings import EXPER_PATH

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
tf.contrib.resampler  # import C++ op

from argparse import ArgumentParser

parser = ArgumentParser(description='Features Extract')
parser.add_argument(
    '-image-dir', '--image-dir',
    type=Path,
    help='image dir'
)

parser.add_argument(
    '-output', '--output',
    type=Path, default='output',
    help='Output folder for featues'
)

args=parser.parse_args()

checkpoint_path = Path(EXPER_PATH, 'vd16_pitts30k_conv5_3_vlad_preL2_intra_white/vd16_pitts30k_conv5_3_vlad_preL2_intra_white')
config = {'checkpoint_path':checkpoint_path, 'data': {'name': 'aachen', 'load_db': False, 'load_queries': True, 'resize_max': 960}, 'model': {'name': 'netvlad_original', 'local_descriptor_layer': 'conv3_3', 'image_channels': 1}, 'weights': 'vd16_pitts30k_conv5_3_vlad_preL2_intra_white/vd16_pitts30k_conv5_3_vlad_preL2_intra_white'}

globs=['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']
image_paths = []
for g in globs:
    image_paths += list(Path(args.image_dir).glob('**/'+g))
if len(image_paths) == 0:
    raise ValueError(f'Could not find any image in root: {args.image_dir}.')
image_paths = sorted(list(set(image_paths)))
image_paths = [i.relative_to(args.image_dir) for i in image_paths]
#print('image_paths:{}]'.format(image_paths))

global_feature_path=Path(args.output, 'global_features.h5')
global_feature_path.parent.mkdir(exist_ok=True, parents=True)
global_feature_file = h5py.File(str(global_feature_path), 'w')

with get_model(config['model']['name'])(
            data_shape={'image': [None, None, None, 3]},
            **config['model']) as net:
      if checkpoint_path is not None:
            net.load(str(checkpoint_path))

      keys = ['global_descriptor']
      cnt = 0
      for im_file in image_paths:
            image = cv2.imread(str(args.image_dir / im_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictions = net.predict({'image':image}, keys=keys)
            grp = global_feature_file.create_group(str(im_file))
            grp.create_dataset('global_descriptor', data=predictions['global_descriptor'])
            cnt += 1
            if cnt % 200 == 0 :
                  print('{} images processed'.format(cnt))

global_feature_file.close()
print('Finished exporting features.')
