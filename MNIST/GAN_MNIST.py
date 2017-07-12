__author__ = 'Weili Nie'

from config import get_config
from img_helpers import load_mnist
from model_img import *
from utils import *


# from inception_score import get_inception_score


class GAN_model(object):
    def __init__(self, config):
        self.d_net = config.d_net
        self.g_net = config.g_net
        self.dataset = config.dataset_img
        self.data_path = config.data_path
        self.n_hidden_layers = config.n_hidden_layers

        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size
        self.loss_type = config.loss_type

        self.z_dim = config.z_dim
        self.conv_hidden_num = config.conv_hidden_num
        self.img_dim = config.img_dim_mnist
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.normalize_d = config.normalize_d
        self.normalize_g = config.normalize_g

        self.model_dir = config.model_dir
        self.log_step = config.log_step
        self.save_step = config.save_step
        self.max_step = config.max_step

        self.is_train = config.is_train

        if not self.is_train:
            self.test_model_dir = config.test_model_dir

        self.lmd = config.lmd

        if self.loss_type in ['WGAN', 'WGAN-GP']:
            self.critic_iters = config.critic_iters
        else:
            self.critic_iters = 1

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.build_model()

        self.summary_writer = tf.summary.FileWriter(self.model_dir)
        self.sv = tf.train.Supervisor(
            logdir=self.model_dir,
            summary_op=None,
            summary_writer=self.summary_writer,
            global_step=self.global_step,
            save_model_secs=300)

    def build_model(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.img_dim**2])
        x_img = tf.reshape(self.x, [-1, self.img_dim, self.img_dim, 1])
        img_chs = x_img.get_shape().as_list()[-1]

        x = x_img  # Normalization

        self.z = tf.random_normal(shape=[self.batch_size, self.z_dim])

        fake_data, g_vars = generator(
            self.g_net, self.z, self.conv_hidden_num,
            self.img_dim, img_chs, self.normalize_g, reuse=False,
            n_hidden_layers=self.n_hidden_layers
        )

        self.fake_data = tf.clip_by_value(fake_data * 255, 0, 255) # Denormalization

        d_out_real, d_vars = discriminator(
            self.d_net, x, self.conv_hidden_num,
            self.normalize_d, reuse=False
        )
        d_out_fake, _ = discriminator(
            self.d_net, fake_data, self.conv_hidden_num,
            self.normalize_d
        )

        self.d_loss, self.g_loss, metric = self.cal_losses(
            x, fake_data, d_out_real, d_out_fake, self.loss_type
        )

        grad_real = tf.reshape(tf.gradients(d_out_real, [x])[0], [self.batch_size, -1])
        self.slope_real = tf.reduce_mean(tf.norm(grad_real, axis=1))

        if self.optimizer == 'adam':
            optim_op = tf.train.AdamOptimizer
        elif self.optimizer == 'rmsprop':
            optim_op = tf.train.RMSPropOptimizer
        else:
            raise Exception("[!] Caution! Other optimizers do not apply right now!")

        self.d_optim = optim_op(self.d_lr, self.beta1, self.beta2).minimize(
            self.d_loss, var_list=d_vars
        )
        self.g_optim = optim_op(self.g_lr, self.beta1, self.beta2).minimize(
            self.g_loss, global_step=self.global_step, var_list=g_vars
        )

        self.summary_op = tf.summary.merge([
            tf.summary.scalar("loss/d_loss", self.d_loss),
            tf.summary.scalar("loss/g_loss", self.g_loss),
            tf.summary.scalar("grad/norm_gradient", self.slope_real),
            tf.summary.scalar("loss/metric", metric)
        ])

        '''
        # For computing inception score
        z_incept_score = tf.random_normal(shape=[100, self.z_dim])
        samples_100, _ = generator(
            self.g_net, z_incept_score, self.conv_hidden_num,
            self.img_dim, img_chs, self.normalize_g
        )
        self.samples_100 = tf.clip_by_value((samples_100 + 1) * 127.5, 0, 255)  # Denormalization
        '''

        # For testing
        self.z_test = tf.placeholder(dtype=tf.float32, shape=[None, self.z_dim])
        samples_test, _ = generator(
            self.g_net, self.z_test, self.conv_hidden_num,
            self.img_dim, img_chs, self.normalize_g
        )
        self.sample_test_normz = samples_test
        self.samples_test = tf.clip_by_value((samples_test + 1) * 127.5, 0, 255)  # Denormalization

    def train(self):
        print('start training...\n [{}] using d_net [{}] and g_net [{}] with loss type [{}]\n'
              'normalize_d: {}, normalize_g: {}'.format(
            self.dataset, self.d_net, self.g_net, self.loss_type, self.normalize_d, self.normalize_g
        ))
        z_fixed = np.random.normal(size=[self.batch_size, self.z_dim])
        mnist_data = load_mnist(self.data_path)

        with self.sv.managed_session() as sess:
            for _ in range(self.max_step):
                if self.sv.should_stop():
                    break

                step = sess.run(self.global_step)

                # Train generator
                sess.run(self.g_optim)

                # Train critic
                for _ in range(self.critic_iters):
                    batch_x, _ = mnist_data.train.next_batch(self.batch_size)
                    sess.run(self.d_optim, {self.x: batch_x})

                if step % self.log_step == 0:
                    batch_x, _ = mnist_data.train.next_batch(self.batch_size)
                    g_loss, d_loss, slope, summary_str = sess.run(
                        [self.g_loss, self.d_loss, self.slope_real, self.summary_op],
                        {self.x: batch_x}
                    )
                    self.summary_writer.add_summary(summary_str, step)
                    self.summary_writer.flush()

                    """
                    incept_score = self.get_inception_score(sess, step, self.save_step, self.model_dir)
                    print("[{}/{}] Loss_D: {:.6f} Loss_G: {:.6f} Slope: {:.6f} Incept_score: {:.6f}".
                          format(step, self.max_step, d_loss, g_loss, slope, incept_score))
                    """

                    print("[{}/{}] Loss_D: {:.6f} Loss_G: {:.6f} Slope: {:.6f}".
                          format(step, self.max_step, d_loss, g_loss, slope))

                if step % self.save_step == 0:
                    self.generate_samples(sess, z_fixed, self.model_dir, idx=step)

    def generate_samples(self, sess, z_fixed, model_dir, idx):
        x = sess.run(self.fake_data, {self.z: z_fixed})
        sample_dir = os.path.join(model_dir, 'samples')

        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)

        path = os.path.join(sample_dir, 'sample_G_{:06d}.png'.format(idx))
        save_image(x, path)
        print("[*] Samples saved: {}".format(path))
        # save_image(sess.run(self.x), os.path.join(sample_dir, 'sample_x_{}.png'.format(idx)))

    def cal_grad_penalty(self, real_data, fake_data):
        # WGAN lipschitz-penalty
        epsilon = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)

        data_diff = fake_data - real_data
        interp_data = real_data + epsilon * data_diff
        disc_interp, _ = discriminator(
            self.d_net, interp_data, self.conv_hidden_num,
            self.normalize_d
        )
        grad_interp = tf.gradients(disc_interp, [interp_data])[0]
        print('The shape of grad_interp: {}'.format(grad_interp.get_shape().as_list()))
        grad_interp_flat = tf.reshape(grad_interp, [self.batch_size, -1])
        slope = tf.norm(grad_interp_flat, axis=1)
        print('The shape of slope: {}'.format(slope.get_shape().as_list()))

        grad_penalty = tf.reduce_mean((slope - 1.) ** 2)
        return grad_penalty

    def cal_one_side_grad_penalty(self, real_data, fake_data):
        # WGAN lipschitz-penalty
        epsilon = tf.random_uniform(shape=[self.batch_size, 1, 1], minval=0., maxval=1.)

        data_diff = fake_data - real_data
        interp_data = real_data + epsilon * data_diff
        disc_interp, _ = discriminator(
            self.d_net, interp_data, self.conv_hidden_num,
            self.normalize_d
        )
        grad_interp = tf.gradients(disc_interp, [interp_data])[0]
        print('The shape of grad_interp: {}'.format(grad_interp.get_shape().as_list()))
        grad_interp_flat = tf.reshape(grad_interp, [self.batch_size, -1])
        slope = tf.norm(grad_interp_flat, axis=1)
        print('The shape of slope: {}'.format(slope.get_shape().as_list()))

        grad_penalty = tf.reduce_mean(tf.nn.relu(slope - 1.) ** 2)
        return grad_penalty

    def cal_real_nearby_grad_penalty(self, real_data):
        # WGAN lipschitz-penalty
        epsilon = tf.random_uniform(shape=[self.batch_size, 1, 1], minval=0., maxval=1.)

        data_diff = get_perturbed_batch(real_data) - real_data
        interp_data = real_data + epsilon * data_diff
        disc_real_nearby, _ = discriminator(
            self.d_net, interp_data, self.conv_hidden_num,
            self.normalize_d
        )
        grad_real_nearby = tf.gradients(disc_real_nearby, [interp_data])[0]
        print('The shape of grad_real_nearby: {}'.format(grad_real_nearby.get_shape().as_list()))
        grad_real_nearby_flat = tf.reshape(grad_real_nearby, [self.batch_size, -1])
        slope = tf.norm(grad_real_nearby_flat, axis=1)
        print('The shape of slope: {}'.format(slope.get_shape().as_list()))

        grad_penalty = tf.reduce_mean((slope - 1.) ** 2)
        return grad_penalty

    def cal_losses(self, real_data, fake_data, d_out_real, d_out_fake, loss_type):
        f_div = ['KL', 'RKL', 'JS', 'Hellinger', 'TV', 'Pearson', 'alpha']
        f_div_gp = [name+'-GP' for name in f_div]
        f_div_osgp = [name + '-OSGP' for name in f_div]
        f_div_rngp = [name + '-RNGP' for name in f_div]
        f_div_all = f_div + f_div_gp + f_div_osgp + f_div_rngp

        if loss_type in ['WGAN-GP', 'WGAN-OSGP', 'WGAN-RNGP']:
            d_loss = tf.reduce_mean(d_out_fake) - tf.reduce_mean(d_out_real)
            g_loss = -tf.reduce_mean(d_out_fake)
            metric = -d_loss

            if loss_type == 'WGAN-GP':
                grad_penalty = self.cal_grad_penalty(real_data, fake_data)
                d_loss += self.lmd * grad_penalty

            elif loss_type == 'WGAN-OSGP':
                grad_penalty = self.cal_one_side_grad_penalty(real_data, fake_data)
                d_loss += self.lmd * grad_penalty

            elif loss_type == 'WGAN-RNGP':
                grad_penalty = self.cal_real_nearby_grad_penalty(real_data)
                d_loss += self.lmd * grad_penalty

            return d_loss, g_loss, metric

        elif loss_type in ['GAN', 'GAN-GP', 'GAN-OSGP', 'GAN-RNGP']:
            d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_out_real, labels=tf.ones_like(d_out_real)
            ))
            d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_out_fake, labels=tf.zeros_like(d_out_fake)
            ))
            d_loss = d_loss_real + d_loss_fake
            # use -logD trick
            g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_out_fake, labels=tf.ones_like(d_out_fake)
            ))
            metric = -d_loss/2 + np.log(2)

            if loss_type == 'GAN-GP':
                grad_penalty = self.cal_grad_penalty(real_data, fake_data)
                d_loss += self.lmd * grad_penalty

            elif loss_type == 'GAN-OSGP':
                grad_penalty = self.cal_one_side_grad_penalty(real_data, fake_data)
                d_loss += self.lmd * grad_penalty

            elif loss_type == 'GAN-RNGP':
                grad_penalty = self.cal_real_nearby_grad_penalty(real_data)
                d_loss += self.lmd * grad_penalty

            return d_loss, g_loss, metric

        elif loss_type in f_div_all:
            loss_name = loss_type.split('-')[0]
            d_loss_real = -tf.reduce_mean(g_f(d_out_real, loss_name))
            d_loss_fake = tf.reduce_mean(f_congugate(g_f(d_out_fake, loss_name)))
            d_loss = d_loss_real + d_loss_fake
            g_loss = -d_loss_fake
            metric = -d_loss

            if loss_type in f_div_gp:
                grad_penalty = self.cal_grad_penalty(real_data, fake_data)
                d_loss += self.lmd * grad_penalty

            elif loss_type in f_div_osgp:
                grad_penalty = self.cal_one_side_grad_penalty(real_data, fake_data)
                d_loss += self.lmd * grad_penalty

            elif loss_type in f_div_rngp:
                grad_penalty = self.cal_real_nearby_grad_penalty(real_data)
                d_loss += self.lmd * grad_penalty

            return d_loss, g_loss, metric

        return None

    '''
    # For calculating inception score
    def get_inception_score(self, sess, idx, save_step, model_dir):
        all_samples = []
        for i in range(10):
            all_samples.append(sess.run(self.samples_100))
        all_samples = np.concatenate(all_samples, axis=0)
        all_samples = ((all_samples + 1.) * 127.5).astype('int32')
        all_samples = all_samples.reshape((-1, 64, 64, 3))
        incept_score = get_inception_score(list(all_samples))

        plot_incept_score(idx, incept_score[0], save_step, model_dir)

        return incept_score[0]
    '''

    def interpolate_G(self, sess, save_path, test_size=10):

        z1 = np.random.normal(size=[test_size, self.z_dim])
        z2 = np.random.normal(size=[test_size, self.z_dim])
        generated = []
        for idx, ratio in enumerate(np.linspace(0, 1, 10)):
            z_slerp = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(z1, z2)])
            samples_slerp = sess.run(self.samples_test, {self.z_test: z_slerp})
            generated.append(samples_slerp)

        generated = np.stack(generated).transpose([1, 0, 2, 3, 4])
        for idx, img in enumerate(generated):
            save_image(img, os.path.join(save_path, 'test_interp_G_{}.png'.format(idx)), nrow=10)

        all_img_num = np.prod(generated.shape[:2])
        batch_generated = np.reshape(generated, [all_img_num] + list(generated.shape[2:]))
        save_image(batch_generated, os.path.join(save_path, 'test_interp_G.png'), nrow=10)

    def one_hot_input_viz(self, sess, save_path):
        z = np.eye(self.z_dim, self.z_dim)
        generator_bias_normz = sess.run(self.sample_test_normz, {self.z_test: np.zeros_like(z)})
        generator_bias_viz = np.clip((generator_bias_normz + 1) * 127.5, 0, 255)
        print('generator_bias_normz: {}'.format(generator_bias_normz[1, 1, 0:15, :]))
        save_image(generator_bias_viz, os.path.join(save_path, 'test_bias.png'), nrow=16)

        for i in range(1,11):
            z_var = z*i

            samples_OH = sess.run(self.samples_test, {self.z_test: z_var})
            samples_OH_normz = sess.run(self.sample_test_normz, {self.z_test: z_var})
            samples_columns_normz = samples_OH_normz - generator_bias_normz
            samples_columns_viz = np.clip((samples_columns_normz + 1) * 127.5, 0, 255)
            print('samples_columns_normz: {}'.format(samples_columns_normz[1, 1, 0:15, :]))
            save_image(samples_OH, os.path.join(save_path, 'test_one_hot_input_add_bias_{}.png'.format(i)), nrow=16)
            save_image(samples_columns_viz, os.path.join(save_path, 'test_one_hot_input_{}.png'.format(i)), nrow=16)


    def test(self):
        print('start testing...')
        with self.sv.managed_session() as sess:
            self.interpolate_G(sess, save_path=self.test_model_dir)

            self.one_hot_input_viz(sess, save_path=self.test_model_dir)


if __name__ == '__main__':
    config, unparsed = get_config()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    prepare_dirs(config, config.dataset_img)

    gan_gp_img = GAN_model(config)
    if config.is_train:
        gan_gp_img.train()
    else:
        if not config.load_path:
            raise Exception("[!] You should specify `load_path` to load a pretrained model")
        gan_gp_img.test()
