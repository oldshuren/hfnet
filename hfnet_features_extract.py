import cv2
import h5py
import numpy as np
from pathlib import Path

from hfnet.settings import EXPER_PATH

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
tf.contrib.resampler  # import C++ op

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

def get_hfnet() :
    model_path = Path(EXPER_PATH, 'saved_models/hfnet')
    outputs = ['global_descriptor', 'keypoints', 'local_descriptors', 'scores']
    return HFNet(model_path, outputs)

def main(image_dir, output_dir, gen_local=True, nms_radius=4, keypoints=4096):
    globs=['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']
    image_paths = []
    for g in globs:
        image_paths += list(Path(image_dir).glob('**/'+g))
    if len(image_paths) == 0:
        raise ValueError(f'Could not find any image in root: {root}.')
    image_paths = sorted(list(set(image_paths)))
    image_paths = [i.relative_to(image_dir) for i in image_paths]
    #print('image_paths:{}]'.format(image_paths))

    global_feature_path=Path(output_dir, 'global_features.h5')
    global_feature_path.parent.mkdir(exist_ok=True, parents=True)
    global_feature_file = h5py.File(str(global_feature_path), 'w')
    if gen_local:
          local_feature_path=Path(output_dir, 'feats-superpoint.h5')
          local_feature_file = h5py.File(str(local_feature_path), 'w')

    hfnet = get_hfnet()
    mode = cv2.IMREAD_COLOR
    cnt = 0
    for im_file in image_paths:
        image = cv2.imread(str(image_dir / im_file), mode)
        db = hfnet.inference(image, nms_radius=nms_radius, num_keypoints=keypoints)
        grp = global_feature_file.create_group(str(im_file))
        grp.create_dataset('global_descriptor', data=db['global_descriptor'])
        if gen_local:
            grp = local_feature_file.create_group(str(im_file))
            grp.create_dataset('keypoints', data=db['keypoints'].astype(np.float32))
            grp.create_dataset('descriptors', data=db['local_descriptors'].T)
            grp.create_dataset('scores', data=db['scores'])
            size = image.shape[:2][::-1]
            grp.create_dataset('image_size', data=np.array(size))
        cnt += 1
        if cnt % 200 == 0 :
            print('{} images processed'.format(cnt))

    global_feature_file.close()
    if gen_local:
        local_feature_file.close()
    print('Finished exporting features.')

if __name__ == '__main__':
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

    parser.add_argument(
        '-gen-local', '--gen-local',
          action='store_true',
          help='Generate local feature'
    )

    parser.add_argument(
        '-nms_radius', '--nms_radius',
          type=int, default=4
    )
    parser.add_argument(
        '-keypoints', '--keypoints',
          type=int, default=4096
    )

    args=parser.parse_args()

    main(args.image_dir, args.output, gen_local=args.gen_local, nms_radius=args.nms_radius, keypoints=args.keypoints)
