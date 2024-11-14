from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--SNR', type=int, default=1, help='Signal to Noise Ratio')
        parser.add_argument('--train_SNR', type=int, default=1, help='Signal to Noise Ratio when traing')
        self.isTrain = False
        return parser
