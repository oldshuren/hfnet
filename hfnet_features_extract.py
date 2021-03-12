import cv2
import h5py
import numpy as np
from pathlib import Path

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


class HFNet:
    def __init__(self, model_path, outputs):
        self.session = tf.Session()
        self.image_ph = tf.placeholder(tf.float32, shape=(None, None, 3))

        net_input = tf.image.rgb_to_grayscale(self.image_ph[None])
        tf.saved_model.loader.load(
            self.session, [tag_constants.SERVING], str(model_path),
            clear_devices=True,
            input_map={'image:0': net_input})

        graph = tf.get_default_graph()
        self.outputs = {n: graph.get_tensor_by_name(n+':0')[0] for n in outputs}
        self.nms_radius_op = graph.get_tensor_by_name('pred/simple_nms/radius:0')
        self.num_keypoints_op = graph.get_tensor_by_name('pred/top_k_keypoints/k:0')
        
    def inference(self, image, nms_radius=4, num_keypoints=1000):
        inputs = {
            self.image_ph: image[..., ::-1].astype(np.float),
            self.nms_radius_op: nms_radius,
            self.num_keypoints_op: num_keypoints,
        }
        return self.session.run(self.outputs, feed_dict=inputs)

model_path = Path(EXPER_PATH, 'saved_models/hfnet')
outputs = ['global_descriptor', 'keypoints', 'local_descriptors', 'scores']
hfnet = HFNet(model_path, outputs)

globs=['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']
image_paths = []
for g in globs:
    image_paths += list(Path(args.image_dir).glob('**/'+g))
if len(image_paths) == 0:
    raise ValueError(f'Could not find any image in root: {root}.')
image_paths = sorted(list(set(image_paths)))
image_paths = [i.relative_to(args.image_dir) for i in image_paths]
#print('image_paths:{}]'.format(image_paths))

global_feature_path=Path(args.output, 'global_features.h5')
local_feature_path=Path(args.output, 'local_features.h5')
local_feature_path.parent.mkdir(exist_ok=True, parents=True)
global_feature_file = h5py.File(str(global_feature_path), 'w')
local_feature_file = h5py.File(str(local_feature_path), 'w')

mode = cv2.IMREAD_COLOR
cnt = 0
for im_file in image_paths:
    image = cv2.imread(str(args.image_dir / im_file), mode)
    db = hfnet.inference(image)
    grp = global_feature_file.create_group(str(im_file))
    grp.create_dataset('global_descriptor', data=db['global_descriptor'])
    grp = local_feature_file.create_group(str(im_file))
    grp.create_dataset('keypoints', data=db['keypoints'])
    grp.create_dataset('descriptors', data=db['local_descriptors'])
    grp.create_dataset('scores', data=db['scores'])
    cnt += 1
    if cnt % 200 == 0 :
        print('{} images processed'.format(cnt))

global_feature_file.close()
local_feature_file.close()
print('Finished exporting features.')
    


