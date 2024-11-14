import torch
import thop
from MDJSCC.models.MDJSCC_model import *
from ptflops import get_model_complexity_info
from models import create_model
from options.train_options import TrainOptions
from options.test_options import TestOptions

###需要将models/DynaAWGN_model.py 中的forward加上一个x对应于get_model_complexity_info的输入。之后直接将self.real_A代替为x
train_opt = TrainOptions().parse()
train_opt.band = 192
train_opt.name = 'cpp' + str(train_opt.band / 512) + '_Pre_' + '0.375'
model = create_model(train_opt)      # create a model given opt.model and other options
with torch.cuda.device(0):

    net = MDJSCCModel(train_opt)
    macs, params = get_model_complexity_info(net, (3, 512, 768), as_strings=True,
                                            print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))
