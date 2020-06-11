from trojanzoo.parser import Parser_Dataset, Parser_Model, Parser_Train, Parser_Seq
from trojanzoo.parser import Parser_Mark
from trojanzoo.parser.attack import Parser_BadNet
from trojanzoo.parser.attack import Parser_Hidden

from trojanzoo.dataset import Dataset
from trojanzoo.model import Model
from trojanzoo.attack import BadNet
from trojanzoo.utils.attack import Watermark
from trojanzoo.attack.backdoor.hidden import HiddenBackdoor
from trojanzoo.utils import save_tensor_as_img

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = Parser_Seq(Parser_Dataset(), Parser_Model(), Parser_Train(),
                        Parser_Mark(), Parser_BadNet(), Parser_Hidden())
    parser.parse_args()
    parser.get_module()

    dataset: Dataset = parser.module_list['dataset']
    model: Model = parser.module_list['model']
    optimizer, lr_scheduler, train_args = parser.module_list['train']
    mark: Watermark = parser.module_list['mark']
    attack: HiddenBackdoor = parser.module_list['attack']
    

    del train_args['epoch']

    # ------------------------------------------------------------------------ #
    attack.attack(optimizer=optimizer, lr_scheduler=lr_scheduler, **train_args)



