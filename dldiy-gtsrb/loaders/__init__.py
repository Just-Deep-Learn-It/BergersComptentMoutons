import json

from loaders.gtsrb_loader import GTSRBLoader


def get_loader(args):
    """get_loader

    :param name:
    """
    return {
        'shanghaitech_loader' : ShanghaiTechLoader,
        # feel free to add new datasets here
    }[args.dataset]
