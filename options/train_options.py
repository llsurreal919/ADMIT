from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=20480, help='frequency of saving the latest results')
        parser.add_argument('--print_freq', type=int, default=10240, help='frequency of ploting losses')
        parser.add_argument('--save_epoch_freq', type=int, default=40, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--pretrain', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        # training parameters
        parser.add_argument('--n_epochs_joint', type=int, default=200, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=200, help='number of epochs with lr decay')
        parser.add_argument('--n_epochs_fine', type=int, default=200, help='number of epochs for finetuning')
        parser.add_argument('--lr_joint', type=float, default=1e-4, help='initial learning rate')
        parser.add_argument('--lr_decay', type=float, default=5e-5, help='decayed learning rate')
        parser.add_argument('--lr_fine', type=float, default=1e-5, help='learning rate for fine-tuning')
        parser.add_argument('--temp_init', type=int, default=5, help='initial temperature for Gumbel-Softmax')
        parser.add_argument('--eta', type=float, default=0.015, help='decay factor for annealling')
        # parser.add_argument('--num_channel', type=float, default=4, help='bandwidth ratios, 4->1/12, 8->1/6')
        # parser.add_argument('--cpp', type=float, default=0, help='bandwidth ratios, 32->0.25, 1->0.3125, 2->0.375, 3->0.4375, 4->0.5')

        # test parameters
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        parser.add_argument('--num_test_channel', type=int, default=5, help='how many random channels for each image')
        parser.add_argument('--SNR', type=int, default=1, help='Signal to Noise Ratio')
          
        self.isTrain = True
        return parser
