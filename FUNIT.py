import tensorflow as tf
from ops import *
from Dataset import save_images, replace_eyes
import os
import numpy as np

class FSUGAN(object):

    # build model
    def __init__(self, data_ob, config):
        # train hyper
        self.g_learning_rate = config.g_learning_rate
        self.d_learning_rate = config.d_learning_rate

        self.lam_recon = config.lam_recon
        self.lam_fp = config.lam_fp
        self.lan_gp = config.lam_gp
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        self.max_iters = config.max_iters
        self.log_vars = []

        # input
        self.data_ob = data_ob
        self.image_size = config.image_size

        self.channel = data_ob.channel
        self.batch_size = config.batch_size

        # output directory
        self.write_model_path = config.write_model_path
        self.validate_evaluation_path = config.validate_evaluation_path
        self.validate_sample_path = config.validate_sample_path
        self.test_sample_path = config.test_sample_path
        self.test_evaluation_path = config.test_evaluation_path
        self.sample_path = config.sample_path
        self.log_dir = config.log_dir

        self.lr_decay = tf.placeholder(tf.float32, None, name='lr_decay')

        # FSU
        self.image_size = config.image_size
        self.K = 2
        self.num_source_class = config.num_source_class
        self.content_ch = config.content_ch
        self.style_ch = config.style_ch

        # placeholder defination
        self.x = tf.placeholder(tf.float32,[self.batch_size,self.image_size,self.image_size,3])
        self.y_1 = tf.placeholder(tf.float32,[self.batch_size,self.image_size,self.image_size,3])
        self.y_2 = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, 3])
        self.cls_x = tf.placeholder(tf.int32, [self.batch_size, 1])

        #self.Y = tf.placeholder(tf.float32, [self.K, self.batch_size, self.image_size, self.image_size,3])
        self.Y = [self.y_1,self.y_2]

    def build_model(self):

        self.content_code = self.content_encoder(self.x, reuse=False)

        self.encode_y1 = self.class_encoder_k(self.y_1, reuse=False)
        self.encode_y2 = self.class_encoder_k(self.y_2, reuse=True)

        self.class_code = tf.add(self.encode_y1, self.encode_y2) / 2

        self.x_bar = self.decoder(content_code=self.content_code, class_code=self.class_code, reuse=False)

        self.encode_x = self.class_encoder_k(self.x, reuse=True)

        self.x_recon = self.decoder(content_code=self.content_code, class_code=self.encode_x, reuse=True)

        self.content_image_recon = tf.reduce_mean( tf.square(self.x - self.x_recon))

        self.x_bar_feature, _ = self.discriminator(self.x_bar,reuse=False)
        self.y_feature_1, _ = self.discriminator(self.y_1,reuse=True)
        self.y_feature_2, _ = self.discriminator(self.y_2, reuse= True)
        self.y_feature = tf.add(self.y_feature_1, self.y_feature_2) / 2

        self.feature_matching = tf.reduce_mean(tf.abs(self.y_feature - self.x_bar_feature))

        _, self.D_real = self.discriminator(self.x, reuse= True)

        #self.D_real = tf.reduce_mean(self.D_real, axis=[1,2,3])
        _, self.D_fake = self.discriminator(self.x_bar,reuse=True)

        self.grad_penalty = self.gradient_penalty(x=self.x, x_tilde=self.x_bar,center=0.0)
        self.cls_x_reshape = tf.reshape(tf.one_hot(self.cls_x, depth=self.num_source_class), shape=[-1, 1, 1, self.num_source_class])

        self.D_gan_loss = self.loss_hinge_dis(self.D_real, self.D_fake, self.cls_x_reshape) + 10 * self.grad_penalty
        self.G_gan_loss = self.loss_hinge_gen(self.D_fake, self.cls_x_reshape)

        self.G_loss = self.G_gan_loss + self.lam_recon * self.content_image_recon + self.lam_fp * self.feature_matching

        self.log_vars.append(('D_loss', self.D_gan_loss))
        self.log_vars.append(('G_loss', self.G_loss))

        self.vars = tf.trainable_variables()

        self.encoder_vars = [var for var in self.vars if "encoder" in var.name]
        self.decoder_vars = [var for var in self.vars if 'decoder' in var.name]
        self.g_vars = self.encoder_vars + self.decoder_vars
        self.d_vars = [var for var in self.vars if 'discriminator' in var.name]

        print "encoder", len(self.encoder_vars),self.encoder_vars
        print "decoder", len(self.decoder_vars),self.decoder_vars
        print "generator", len(self.g_vars), self.g_vars
        print "discriminator_vars", len(self.d_vars),self.d_vars

        self.saver = tf.train.Saver()
        for k, v in self.log_vars:
            tf.summary.scalar(k, v)

    def train(self):

        opti_G = tf.train.RMSPropOptimizer(self.g_learning_rate * self.lr_decay).minimize(loss=self.G_loss,
                                                                                          var_list=self.encoder_vars + self.decoder_vars)
        opti_D = tf.train.RMSPropOptimizer(self.d_learning_rate * self.lr_decay).minimize(loss=self.D_gan_loss,
                                                                                          var_list=self.d_vars)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            ckpt = tf.train.get_checkpoint_state(self.write_model_path)
            if ckpt and ckpt.model_checkpoint_path:
                start_step = int(ckpt.model_checkpoint_path.split('model_', 2)[1].split('.', 2)[0])
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                print(ckpt.model_checkpoint_path)
                print('Load Successfully!')
            else:
                start_step = 0

            step = start_step
            lr_decay = 1

            print("Start read dataset")
            while step <= self.max_iters:

                if step > 20000 and step % 2000 == 0:
                    lr_decay = (self.max_iters - step) / float(self.max_iters - 20000)

                source_image_x_data, target_image_y1_data, target_image_y2_data, cls_x = self.data_ob.getNextBatch()

                source_image_x = self.data_ob.getShapeForData(source_image_x_data)
                target_image_y1 = self.data_ob.getShapeForData(target_image_y1_data)
                target_image_y2 = self.data_ob.getShapeForData(target_image_y2_data)

                cls_x = np.reshape(cls_x, newshape=[-1, 1])

                f_d = {
                    self.x :source_image_x,
                    self.y_1:target_image_y1,
                    self.y_2 : target_image_y2,
                    self.cls_x: cls_x,
                    self.lr_decay: lr_decay
                }

                sess.run(opti_D, feed_dict=f_d)
                sess.run(opti_G, feed_dict=f_d)

                summary_str = sess.run(summary_op, feed_dict=f_d)
                summary_writer.add_summary(summary_str, step)

                if step % 500 == 0:

                    output_loss = sess.run([ self.D_gan_loss, self.G_loss, self.G_gan_loss,
                                             self.content_image_recon, self.feature_matching], feed_dict=f_d)
                    print("step %d, D_loss=%.4f,"
                          "G_loss=%.4f, G_gan_loss=%.4f, content_recon=%.4f, feautre_loss=%.4f"
                          ", lr_decay=%.4f" % (step, output_loss[0], output_loss[1], output_loss[2], output_loss[3],
                                                          output_loss[4] ,lr_decay))

                if np.mod(step, 1000) == 0:

                    f_d = {
                        self.x: source_image_x,
                        self.y_1: target_image_y1,
                        self.y_2: target_image_y2,
                    }

                    train_output_img = sess.run([
                        self.x,
                        self.y_1,
                        self.y_2,
                        self.x_bar,
                        self.x_recon
                    ],feed_dict=f_d)

                    output_img = np.concatenate([train_output_img[0],train_output_img[1],train_output_img[2],train_output_img[3],train_output_img[4]],axis=0)

                    save_images(output_img, [output_img.shape[0]/self.batch_size, self.batch_size],
                                '{}/{:02d}_output_img.jpg'.format(self.sample_path, step))

                if np.mod(step, 20000) == 0 and step != 0:
                    self.saver.save(sess, os.path.join(self.write_model_path, 'model_{:06d}.ckpt'.format(step)))
                step += 1

            save_path = self.saver.save(sess, os.path.join(self.write_model_path, 'model_{:06d}.ckpt'.format(step)))
            summary_writer.close()

            print "Model saved in file: %s" % save_path
    def cosine(self, f1, f2):
        f1_norm = tf.nn.l2_normalize(f1, dim=0)
        f2_norm = tf.nn.l2_normalize(f2, dim=0)

        return tf.losses.cosine_distance(f1_norm, f2_norm, dim=0)

    #softplus
    def loss_gen(self, d_fake_logits):
        return tf.reduce_mean(tf.nn.softplus(-d_fake_logits))

    def loss_dis(self, d_real_logits, d_fake_logits):
        l1 = tf.reduce_mean(tf.nn.softplus(-d_real_logits))
        l2 = tf.reduce_mean(tf.nn.softplus(d_fake_logits))
        return l1 + l2

    #hinge loss
    #Hinge Loss + sigmoid
    def loss_hinge_dis(self, d_real_logits, d_fake_logits, cls_x):

        print cls_x.shape
        print (tf.nn.relu(1.0 - d_real_logits)).shape
        print (tf.nn.relu(1.0 + d_fake_logits)).shape

        loss = tf.reduce_mean(tf.reduce_sum(cls_x * (tf.nn.relu(1.0 - d_real_logits)), axis=3), axis=[0,1,2])
        loss += tf.reduce_mean(tf.reduce_sum(cls_x * (tf.nn.relu(1.0 + d_fake_logits)), axis=3), axis=[0,1,2])

        return loss

    def loss_hinge_gen(self, d_fake_logits, cls_x):
        loss = - tf.reduce_mean(tf.reduce_sum(cls_x * d_fake_logits, axis=3), axis=[0,1,2])
        return loss

    #wgan loss
    def d_wgan_loss(self, d_real_logits, d_fake_logits):
        return tf.reduce_mean(d_fake_logits) - tf.reduce_mean(d_real_logits)

    def g_wgan_loss(self, g_fake_logits):
        return - tf.reduce_mean(g_fake_logits)

    def gradient_penalty(self, x_tilde, x, center=1.0):
        self.differences = x_tilde - x
        self.alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        interpolates = x + self.alpha * self.differences
        discri_logits = self.discriminator(interpolates, reuse=True)
        gradients = tf.gradients(discri_logits, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))

        return tf.reduce_mean((slopes - center)**2)

    #lsgan
    def d_lsgan_loss(self, d_real_logits, d_fake_logits):
        return tf.reduce_mean((d_real_logits - 0.9)*2) + tf.reduce_mean((d_fake_logits)*2)

    def g_lsgan_loss(self, d_fake_logits):
        return tf.reduce_mean((d_fake_logits - 0.9)*2)

    def transform_image(self,image):
        return (image+1) * 127.5

    def content_encoder(self, x, reuse=False):

        channel = self.content_ch
        with tf.variable_scope("content_encoder") as scope:

            if reuse:
                scope.reuse_variables()

            x = conv2d(x, output_dim=64, kernel=7, stride=1, padding='SAME', scope='conv-1')
            x = instance_norm(x,scope='IN-1')
            x = tf.nn.relu(x)

            for i in range(3):
                x = conv2d(x, output_dim=pow(2,i)*channel, kernel=4, stride=2, padding='SAME', scope='conv_{}'.format(i+1))
                x = instance_norm(x,scope='ins_{}'.format(i+1))
                x = tf.nn.relu(x)

            for i in range(2):
                x = Resblock(x, channels=512, is_start=False, is_norm=True, is_acti=True, scope='residual_{}'.format(i))

        return x

    def class_encoder_k(self, y, reuse=False):

        channel = self.style_ch
        with tf.variable_scope("class_encoder_k") as scope:
            if reuse:
                scope.reuse_variables()

            y = conv2d(y,output_dim=64, kernel=7, stride=1, padding='SAME',scope='conv-1')

            for i in range(4):
                y = tf.nn.relu(y)
                y = conv2d(y, output_dim=channel*pow(2,i), kernel=4, stride=2, padding='SAME',scope='conv_{}'.format(i+1))

            y = tf.reduce_mean(y, axis=[1,2])

        return y

    def Resblocks_AdaIn(self,x, stage, beta, gamma, reuse=False):
        conv_name_base = 'adain_resblk_' + str(stage)
        with tf.variable_scope(conv_name_base) as scope:
            if reuse:
                scope.reuse_variables()
            x = Resblock_AdaIn(x, beta, gamma, channels=512, scope='R_AdaIn1')
            x = Resblock_AdaIn(x, beta, gamma, channels=512, scope='R_AdaIn2')

        return x

    def decoder(self, content_code, class_code, reuse =True):

        channel = 256
        with tf.variable_scope("decoder") as scope:
            if reuse:
                scope.reuse_variables()

            for i in range(4):
                if i == 3:
                    class_code = fully_connect(input_=class_code, output_size=512*4, scope='fc_{}'.format(i+1))
                else:
                    class_code = tf.nn.relu(fully_connect(input_=class_code, output_size=256, scope='fc_{}'.format(i+1)))

            mean1 = class_code[:,0:512]
            stand_dev1 = class_code[:,512:512*2]
            mean2 = class_code[:,512*2:512*3]
            stand_dev2 = class_code[:,512*3:512*4]

            de = Resblock_AdaIn(content_code, beta=mean1, gamma=stand_dev1, channels=512, scope='res_1')
            de = Resblock_AdaIn(de, beta=mean2,  gamma=stand_dev2, channels=512, scope='res_2')

            for i in range(3):
                de = upscale(de, scale=2)
                de = conv2d(de, output_dim=channel/pow(2,i), kernel=4, stride=1, padding='SAME', scope='conv_{}'.format(i+1))
                de = instance_norm(de,scope='ins_{}'.format(i+1))
                de = tf.nn.relu(de)

            y = conv2d(de, output_dim=3, kernel=7, stride=1, padding='SAME', scope='conv_final')

            return tf.nn.tanh(y)

    def discriminator(self,x, reuse= True):

        channel = 128
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            x = conv2d(input_=x, output_dim=64, kernel=7, stride=1, scope='conv-64')

            for i in range(5):
                x = Resblock(x, channels=channel * pow(2, i), is_acti=True, is_start=True, is_norm=False,
                             scope='r1_{}'.format(i + 1))
                if i == 4:
                    x = Resblock(x, channels=channel*pow(2,i), is_acti=False, is_start=True, is_norm=False, scope='r2_{}'.format(i+1))
                else:
                    x = Resblock(x, channels=channel*pow(2,i), is_acti=True, is_start=True, is_norm=False, scope='r2_{}'.format(i+1))
                    x = avgpool2d(x, k=2)

            x_predict = conv2d(x, output_dim=self.num_source_class, kernel=1, stride=1, padding='SAME')

        return x, x_predict







