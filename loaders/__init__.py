import json

from loaders.shanghaitech_loader import ShanghaiTechLoader


def get_loader(args):
    """get_loader

    :param name:
    """
    return {
        'shanghaitech' : ShanghaiTechLoader,
        # feel free to add new datasets here
    }[args.dataset]
