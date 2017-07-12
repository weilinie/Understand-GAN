import argparse

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

def str2bool(v):
    return v.lower() in ('true', '1')

# Network
net_parser = add_argument_group('Network')
net_parser.add_argument('--g_net', type=str, default='MLP', choices=['ResNet', 'DCGAN', 'MLP'])
net_parser.add_argument('--d_net', type=str, default='MLP', choices=['ResNet', 'DCGAN', 'MLP'])
net_parser.add_argument('--conv_hidden_num', type=int, default=64, choices=[16, 64, 128, 256, 512])
net_parser.add_argument('--n_hidden_layers', type=int, default=-1, help='for MLP net only')
net_parser.add_argument('--batch_size', type=int, default=64)
net_parser.add_argument('--normalize_d', type=str, default='No', choices=['LN', 'BN', 'No'],
                        help='layer, batch or no normalization for D')
net_parser.add_argument('--normalize_g', type=str, default='No', choices=['LN', 'BN', 'No'],
                        help='layer, batch or no normalization for G')
net_parser.add_argument('--z_dim', type=int, default=128, choices=[64, 128])


# Data
data_parser = add_argument_group('Data')
data_parser.add_argument('--dataset_img', type=str, default='RenderBallTri')
data_parser.add_argument('--img_dim_celebA', type=int, default=64, help='image shape: [img_dim, img_dim, 3]')
data_parser.add_argument('--img_dim_mnist', type=int, default=28, help='image shape: [img_dim, img_dim, 1]')
data_parser.add_argument('--img_dim_gmm', type=int, default=2, help='image shape: [1, img_dim, 1]')
data_parser.add_argument('--img_dim_renderData', type=int, default=64, help='image shape: [img_dim, img_dim, 3]')
data_parser.add_argument('--data_dir', type=str, default='data')
data_parser.add_argument('--split', type=str, default='train', help='for CelebA only')
data_parser.add_argument('--num_cluster', type=int, default=4, help='for gmm dataset only')

# Training
train_parser = add_argument_group('Training')
train_parser.add_argument('--is_train', type=str2bool, default=True)
train_parser.add_argument('--loss_type', type=str, default='WGAN-GP')
train_parser.add_argument('--optimizer', type=str, default='adam')
train_parser.add_argument('--max_step', type=int, default=100000, help='maximum iterations')
train_parser.add_argument('--d_lr', type=float, default=1e-4)
train_parser.add_argument('--g_lr', type=float, default=1e-4)
train_parser.add_argument('--beta1', type=float, default=0.5, help='for Adam use only')
train_parser.add_argument('--beta2', type=float, default=0.9, help='for Adam use only')
train_parser.add_argument('--gpus', type=str, default='5')
train_parser.add_argument('--lmd', type=int, default=10, help='gradient penalty lambda hyperparameter')
train_parser.add_argument('--critic_iters', type=int, default=10, help='for WGAN use only')

# Summary and logs
summary_parser = add_argument_group('Summary')
summary_parser.add_argument('--load_path', type=str, default='wgan_gp_mlp1_none',
                            help='path or suffix [e.g. gan_dcgan_bn]')
summary_parser.add_argument('--log_step', type=int, default=20)
summary_parser.add_argument('--save_step', type=int, default=100)
summary_parser.add_argument('--log_dir', type=str, default='logs')
summary_parser.add_argument('--test_dir', type=str, default='tests',
                      help='directory with images which will be used in test sample generation')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed