# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco import build as build_coco

# For remote sensing dataset loading
from .coco_FSDA import build_xView2DOTA_DA, build_city_DA, build_GTAV10k2UCAS_AOD_DA, build_HRRSD2SSDD_DA



def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    if args.dataset_file == 'o365':
        from .o365 import build_o365_combine
        return build_o365_combine(image_set, args)
    if args.dataset_file == 'vanke':
        from .vanke import build_vanke
        return build_vanke(image_set, args)

    # For cross-satellite benchmark (xView→DOTA)
    if args.dataset_file == 'xView2DOTA':
        return build_xView2DOTA_DA(image_set, args)

    # For synthetic-to-real benchmark (GTAV10k→UCAS_AOD)
    if args.dataset_file == 'GTAV10k2UCAS_AOD':
        return build_GTAV10k2UCAS_AOD_DA(image_set, args)

    # For optical-to-SAR benchmark
    if args.dataset_file == 'HRRSD2SSDD':
        return build_HRRSD2SSDD_DA(image_set, args)

    # Add natural scenery
    if args.dataset_file == 'city':
        return build_city_DA(image_set, args)



    raise ValueError(f'dataset {args.dataset_file} not supported')
