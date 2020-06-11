# -*- coding: utf-8 -*-


from .badnet import Parser_BadNet

class Parser_Hidden(Parser_BadNet):
    r"""Hidden Backdoor Attack Parser

    Attributes:
        name (str): ``'attack'``
        attack (str): ``'hidden'``
    """
    attack = 'hidden'


    @classmethod
    def add_argument(cls, parser):
        super().add_argument(parser)
        parser.add_argument('--poisoned_image_num', dest='poisoned_image_num', type=int,
                            help='the number of poisoned image, defaults to config[hidden][poisoned_image_num]=100')
        parser.add_argument('--poison_generation_iteration', dest='poison_generation_iteration', type=int,
                            help='the iteration of generating poisoned image, defaults to config[hidden][poison_generation_iteration]=5000')
        parser.add_argument('--poison_lr', dest='poison_lr', type=float,
                            help='the lr when generating poisoned image, defaults to config[hidden][poison_lr]=0.01')
        parser.add_argument('--preprocess_layer', dest='preprocess_layer', type=str,
                            help='the chosen specific layer that on which the feature space of source images patched by trigger is close to poisoned images, defaults to config[hidden][preprocess_layer]=features')
        parser.add_argument('--epsilon', dest='epsilon', type=int,
                            help='the threshold in pixel space to ensure the poisoned image is not visually distinguishable from the target image, defaults to config[hidden][epsilon]=16')
        parser.add_argument('--decay', dest='decay', type=bool,
                            help='specify whether the learning rate decays with iteraion times, defaults to config[hidden][decay]=True')
        parser.add_argument('--decay_iteration', dest='decay_iteration', type=int,
                            help='specify the number of iteration time interval, the learning rate will decays once, defaults to config[hidden][decay_iteration]=2000')
        parser.add_argument('--decay_ratio', dest='decay_ratio', type=float,
                            help='specify the learning rate decay proportion, defaults to config[hidden][decay_ratio]=0.95')