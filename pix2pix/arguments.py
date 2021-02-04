import argparse

class Arguments:
    def __init__(self):
        self._parser = argparse.ArgumentParser(description='Arguments for TreeGAN.')

        self._parser.add_argument('--gpu', type=int, default=1, help='GPU number to use.')
        # Dataset arguments
        self._parser.add_argument('--batch_size', type=int, default=8, help='Integer value for batch size.')
        self._parser.add_argument('--image_size', type=int, default=256, help='Integer value for number of points.')
        self._parser.add_argument('--input_nc', type=int, default=3, help='size of image height')
        self._parser.add_argument('--output_nc', type=int, default=3, help='size of image height')
        
        # Optimizer arguments
        self._parser.add_argument('--b1', type=int, default=0.5, help='Adam.')
        self._parser.add_argument('--b2', type=int, default=0.999, help='Adam.')
        self._parser.add_argument('--lr', type=float, default=2e-4, help='Adam : learning rate.')
        self._parser.add_argument('--decay_epoch', type=int, default=100, help="epoch from which to start lr decay")

        # Training arguments
        self._parser.add_argument('--epoch', type=int, default=1, help='Epoch to start training from.')
        self._parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs of training.')
        self._parser.add_argument('--data_path', type=str, default='/mnt/hdd/jongjin/data/data/edges2shoes/', help='Dataset path.')
        self._parser.add_argument('--ckpt_path', type=str, default='/mnt/hdd/jongjin/data/ckpt/', help='Network save path.')
        self._parser.add_argument('--result_path', type=str, default='/mnt/hdd/jongjin/data/result/', help='Generated results path.')
        self._parser.add_argument('--sample_path', type=str, default='/mnt/hdd/jongjin/data/sample/', help='Generated results path.')
        self._parser.add_argument('--log_path', type=str, default='/mnt/hdd/jongjin/log/', help='Generated results path.')

        
    def parser(self):
        return self._parser


