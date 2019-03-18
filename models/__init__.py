from models.csrnet import csrnet

def get_model(args):
    model_instance = _get_model_instance(args.arch)

    print('Fetching model %s - %s ' % (args.arch, args.model_name))
    if args.arch == 'csrnet':
        model = model_instance(args.model_name, args.num_classes, args.input_channels, args.pretrained)
    else:
        raise 'Model {} not available'.format(args.arch)
    return model

def _get_model_instance(name):
    return {
        'csrnet': csrnet,
}[name]
