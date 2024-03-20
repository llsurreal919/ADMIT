from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--num_test', type=int, default=10000, help='how many test images to run')
        parser.add_argument('--num_test_channel', type=int, default=5, help='how many random channels for each image')
        parser.add_argument('--SNR', type=int, default=1, help='Signal to Noise Ratio')
        parser.add_argument('--train_SNR', type=int, default=1, help='Signal to Noise Ratio when traing')
        # parser.add_argument('--test_dir', type=str, default='./Pretrained', help='test dir')
        self.isTrain = False
        return parser
