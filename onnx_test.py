import warnings
warnings.filterwarnings('ignore')
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
from functools import partial

import mmcv
import torch.multiprocessing as mp
from torch.multiprocessing import Process, set_start_method

from mmdeploy.apis import (create_calib_input_data, extract_model,
                           get_predefined_partition_cfg, torch2onnx,
                           torch2torchscript, visualize_model)
from mmdeploy.apis.core import PIPELINE_MANAGER
from mmdeploy.apis.utils import to_backend
from mmdeploy.backend.sdk.export_info import export2SDK
from mmdeploy.utils import (IR, Backend, get_backend, get_calib_filename,
                            get_ir_config, get_partition_config,
                            get_root_logger, load_config, target_wrapper)


def parse_args():
    parser = argparse.ArgumentParser(description='Export model to backends.')
    parser.add_argument('deploy_cfg', help='deploy config path')
    parser.add_argument('model_cfg', help='model config path')
    parser.add_argument('checkpoint', help='model checkpoint path')
    parser.add_argument('img', help='image used to convert model model')
    parser.add_argument(
        '--test-img',
        default=None,
        type=str,
        nargs='+',
        help='image used to test model')
    parser.add_argument(
        '--work-dir',
        default=os.getcwd(),
        help='the dir to save logs and models')
    parser.add_argument(
        '--calib-dataset-cfg',
        help=('dataset config path used to calibrate in int8 mode. If not '
              'specified, it will use "val" dataset in model config instead.'),
        default=None)
    parser.add_argument(
        '--device', help='device used for conversion', default='cpu')
    parser.add_argument(
        '--log-level',
        help='set log level',
        default='INFO',
        choices=list(logging._nameToLevel.keys()))
    parser.add_argument(
        '--show', action='store_true', help='Show detection outputs')
    parser.add_argument(
        '--dump-info', action='store_true', help='Output information for SDK')
    parser.add_argument(
        '--quant-image-dir',
        default=None,
        help='Image directory for quantize model.')
    parser.add_argument(
        '--quant', action='store_true', help='Quantize model to low bit.')
    parser.add_argument(
        '--uri',
        default='192.168.1.1:60000',
        help='Remote ipv4:port or ipv6:port for inference on edge device.')
    args = parser.parse_args()
    return args


def create_process(name, target, args, kwargs, ret_value=None):
    logger = get_root_logger()
    logger.info(f'{name} start.')
    log_level = logger.level

    wrap_func = partial(target_wrapper, target, log_level, ret_value)

    process = Process(target=wrap_func, args=args, kwargs=kwargs)
    process.start()
    process.join()

    if ret_value is not None:
        if ret_value.value != 0:
            logger.error(f'{name} failed.')
            exit(1)
        else:
            logger.info(f'{name} success.')


def main():
    args = parse_args()
    set_start_method('spawn', force=True)
    logger = get_root_logger()
    log_level = logging.getLevelName(args.log_level)
    logger.setLevel(log_level)

    deploy_cfg_path = args.deploy_cfg
    model_cfg_path = args.model_cfg

    # load deploy_cfg
    deploy_cfg, model_cfg = load_config(deploy_cfg_path, model_cfg_path)

    ret_value = mp.Value('d', 0, lock=False)
    backend = get_backend(deploy_cfg)
    backend_files = [args.checkpoint]
    if args.test_img is None:
        args.test_img = args.img

    extra = dict(
        backend=backend,
        output_file=osp.join(args.work_dir, f'output_{backend.value}.jpg'),
        show_result=args.show)

    # get backend inference result, try render
    create_process(
        f'visualize {backend.value} model',
        target=visualize_model,
        args=(model_cfg_path, deploy_cfg_path, backend_files, args.test_img,
              args.device),
        kwargs=extra,
        ret_value=ret_value)
    logger.info('All process success.')


if __name__ == '__main__':
    main()

# python onnx_test.py \
#     mmdeploy/configs/mmdet/detection/detection_onnxruntime_dynamic.py \
#     mmdetection/configs/a_yu/ld_r18_gflv1_r101_fpn_coco_1x_student.py \
#     mmdeploy_model/r18_onnx/end2end.onnx \
#     img_brand/brands.jpg \
#     --work-dir mmdeploy_model/test \
#     --device cpu \
#     --dump-info
