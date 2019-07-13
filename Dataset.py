import errno
import numpy as np
import scipy
import random
import cv2
import os
import scipy.misc as misc
import imageio
from config import Config # only for code test phase

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def get_image(image_path , image_size, is_crop= False, resize_w= 64, is_grayscale= False, is_test=False):
    return transform(imread(image_path , is_grayscale), image_size, is_crop , resize_w, is_test=is_test)

def transform(image, npx= 64 , is_crop=False, resize_w=64, is_test=False):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image , npx , resize_w=resize_w, is_test=is_test)
    else:
        cropped_image = image
        cropped_image = scipy.misc.imresize(cropped_image ,
                            [resize_w , resize_w])
    return np.array(cropped_image) / 127.5 - 1

def center_crop(x, crop_h , crop_w=None, resize_w=64, is_test=False):

    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))

    if not is_test:
        rate = np.random.uniform(0, 1, size=1)
        if rate < 0.5:
            x = np.fliplr(x)
    # return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
    #                            [resize_w, resize_w])
    return scipy.misc.imresize(x[20:218 - 20, 0: 178], [resize_w, resize_w])

def save_images(images, size, image_path, is_ouput=False):
    return imsave(inverse_transform(images, is_ouput), size, image_path)

def imread(path, is_grayscale=False):

    if (is_grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def merge(images, size):

    if size[0] + size[1] == 2:
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1], 3))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w: i * w + w, :] = image

    else:
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1], 3))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w: i * w + w, :] = image

    return img

def inverse_transform(image, is_ouput=False):

    if is_ouput == True:
        print image[0]
    result = ((image + 1) * 127.5).astype(np.uint8)
    if is_ouput == True:
        print result
    return result

log_interval = 1000

class CUB_bird(object):

    def __init__(self, config):

        self.dataname = "CUB_200_2011"
        self.data_dir = config.data_dir
        self.height, self.width, self.channel = config.hwc
        self.batch_size = config.batch_size
        self.image_size = config.image_size

        self.shape = [self.image_size, self.image_size, self.channel]
        self.images_dict = self.read_image_dict()

    def read_image_dict(self):

        #fh = open(self.data_dir + "/eye_position_2.txt")
        fh = open('images.txt')
        images_dict = dict()

        for f in fh.readlines():
            f = f.strip('\n').split(' ')[-1]
            info = f.split('/')
            image_folder = info[0]
            image_folder_num = info[0].split('.')[0]
            imagename = info[1]
            if not images_dict.has_key(image_folder_num):
                images_dict[image_folder_num] = []
            if os.path.exists(os.path.join(self.data_dir, image_folder+'/'+imagename)):
                images_dict[image_folder_num] .append(image_folder+'/'+imagename)

        fh.close()
        #print(images_dict)
        #print(len(images_dict))# 200
        return images_dict

    def getNextBatch(self):

        source_image_x = []
        target_image_y1 = []
        target_image_y2 = []
        x_cls = []
        y_cls = []

        for i in range(self.batch_size):

            # using 1-170 for train , 170-200 for test
            id_domain = range(1, 171)
            id_x, id_y = random.sample(id_domain, 2)
            x_cls.append(id_x)
            y_cls.append(id_y)
            format_id_x = '%03d' % id_x  # source class
            format_id_y = '%03d' % id_y  # target class
            #print self.images_dict[format_id_x]
            source_image_name = random.sample(self.images_dict[format_id_x], 1)[0]
            the_path = os.path.join(self.data_dir, source_image_name)
            source_image_x.append(the_path)
            target_image_name_1, target_image_name_2 = random.sample(self.images_dict[format_id_y], 2)
            the_path_1 = os.path.join(self.data_dir,target_image_name_1)
            the_path_2 = os.path.join(self.data_dir, target_image_name_2)
            target_image_y1.append(the_path_1)
            target_image_y2.append(the_path_2)

        return np.asarray(source_image_x), np.asarray(target_image_y1), \
                    np.asarray(target_image_y2), np.asarray(x_cls), np.asarray(y_cls)

    def getShapeForData(self, filenames, is_test=False):
        array = [get_image(batch_file, 108, is_crop=False, resize_w=self.image_size,
                           is_grayscale=False, is_test=is_test) for batch_file in filenames]
        sample_images = np.array(array)

        return sample_images


def save_as_gif(images_list, out_path, gif_file_name='all', save_image=False):
    if os.path.exists(out_path) == False:
        os.mkdir(out_path)

    # save as .png
    if save_image == True:
        for n in range(len(images_list)):
            file_name = '{}.png'.format(n)
            save_path_and_name = os.path.join(out_path, file_name)
            misc.imsave(save_path_and_name, images_list[n])
    # save as .gif
    out_path_and_name = os.path.join(out_path, '{}.gif'.format(gif_file_name))
    imageio.mimsave(out_path_and_name, images_list, 'GIF', duration=0.1)


