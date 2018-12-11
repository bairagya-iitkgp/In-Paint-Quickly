import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', True):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', False):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='')

parser.add_argument('--IMAGE_SIZE', dest='IMAGE_SIZE', default=128, type=int, help='input image size')
parser.add_argument('--LOCAL_SIZE', dest='LOCAL_SIZE', default=64, type=int, help='size of input to local discriminator')
parser.add_argument('--HOLE_MIN', dest='HOLE_MIN', default=24, type=int, help='minimun hole size')
parser.add_argument('--HOLE_MAX', dest='HOLE_MAX', default=48, type=int, help='maximum hole size')
parser.add_argument('--LEARNING_RATE', dest='LEARNING_RATE', default=1e-3, type=float, help='learning rate')
parser.add_argument('--BATCH_SIZE', dest='BATCH_SIZE', default=2, type=int, help='batch size')
parser.add_argument('--PRETRAIN_EPOCH', dest='PRETRAIN_EPOCH', default=4, type=int, help='no of epochs for pre-train phase')
parser.add_argument('--Td_EPOCH', dest='Td_EPOCH', default=1, type=int, help='no of epochs for discriminator-train phase')
parser.add_argument('--Tot_EPOCH', dest='Tot_EPOCH', default=16, type=int, help='total no of epochs')
parser.add_argument('--alpha', dest='alpha', default=1.0, type=float, help='alpha')
parser.add_argument('--res', dest='res', default=128, type=int, help='Choose Resolution for Training')
parser.add_argument('--use_pretrain', dest='use_pretrain', default=False, type=str2bool, help='set True if using any pre-trained weights before training else set False')
parser.add_argument('--model', dest='model', default=2, type=int, help='choose desired model for training : 1-> BASELINE 2-> BILINEAR RESIZE 3-> PIXEL SHUFFLE')

#Extra folders setting
parser.add_argument('--checkpoints_path', dest='checkpoints_path', default='./backup', help='saved model checkpoint path')
parser.add_argument('--restoration_path', dest='restoration_path', default='./backup/latest', help='saved model restoration path')
parser.add_argument('--pretrain_path', dest='pretrain_path', default='./restore/latest', help='pre-trained model restoration path')
parser.add_argument('--data_path', dest='data_path', default='./npy', help='path to npy data file')
parser.add_argument('--original', dest='original', default='./original/', help='path to save original images')
parser.add_argument('--output', dest='output', default='./output/', help='path to save output images')
parser.add_argument('--perturbed', dest='perturbed', default='./perturbed/', help='path to save perturbed images')
parser.add_argument('--log', dest='log', default='./lossvalues.log', help='path to log training history')


args = parser.parse_args()

