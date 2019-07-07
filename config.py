import os

class Config:

    operation_name = "t"

    is_training = True # no use
    gpu_id = 7
    image_size = 256
    hwc = [image_size, image_size, 3]
    max_iters = 500000
    g_learning_rate = 0.0001
    d_learning_rate = 0.0001

    num_source_class = 170

    # hyper
    batch_size = 2
    lam_recon = 0.1
    lam_fp = 1
    lam_gp = 10
    use_sp = True
    K = 2
    content_ch = 128
    style_ch = 128


    beta1 = 0.5
    beta2 = 0.999


    @property
    def base_path(self):
        return os.path.abspath(os.curdir)

    @property
    def data_dir(self):
        data_dir = '/mnt/sata/jichao/dataset/CUB_200_2011/croped_images'
        if not os.path.exists(data_dir):
            raise ValueError('Please specify a data dir.')
        return data_dir

    @property
    def exp_dir(self):
        exp_dir = os.path.join(self.base_path, 'train_log' + self.operation_name)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        return exp_dir

    @property
    def write_model_path(self):
        model_path = os.path.join(self.exp_dir, 'write_model')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        return model_path

    @property
    def log_dir(self):
        log_dir = os.path.join(self.exp_dir, 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    @property
    def sample_path(self):
        sample_path = os.path.join(self.exp_dir, 'sample_img')
        if not os.path.exists(sample_path):
            os.makedirs(sample_path)
        return sample_path

    @property
    def validate_sample_path(self):
        sample_path = os.path.join(self.exp_dir, 'validate_output_img')
        if not os.path.exists(sample_path):
            os.makedirs(sample_path)
        return sample_path
    @property
    def validate_evaluation_path(self):
        evaluation_path = os.path.join(self.exp_dir,'validate_evaluation_result')
        if not os.path.exists(evaluation_path):
            os.makedirs(evaluation_path)
        return evaluation_path
    @property
    def test_sample_path(self):
        sample_path = os.path.join(self.exp_dir,'test_output_img')
        if not os.path.exists(sample_path):
            os.makedirs(sample_path)
        return sample_path
    @property
    def test_evaluation_path(self):
        sample_path = os.path.join(self.exp_dir,'test_evaluation_result')
        if not os.path.exists(sample_path):
            os.makedirs(sample_path)
        return sample_path


