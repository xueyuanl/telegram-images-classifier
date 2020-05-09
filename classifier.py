import logging
import os
from random import randint

import yaml
from PIL import UnidentifiedImageError

from label import label_image

logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)
logging.getLogger('classifier').setLevel(level=logging.WARNING)

with open('conf.yaml') as f:
    path = yaml.safe_load(f)['path']


def create_folder(ff):
    if not os.path.exists(ff):
        os.makedirs(ff)


def class_photo(s_file, t_file):
    logging.info('move {} to {}'.format(s_file, t_file))
    os.system('mv {} {}'.format(s_file, t_file))


def pick_up_photo(folder):
    files = os.listdir(folder)
    file_number = len(files)
    if file_number < 1:
        return None
    logging.info('get the total number of files is {}.'.format(file_number))
    index = randint(0, file_number - 1)
    return files[index]


def format_filename(file):
    pass


def append_rate_to_file_name(rate, file):
    r = str(rate)[2:5]
    tmp = file.split('.')
    new_file = tmp[0] + '_' + r + '.' + tmp[1]
    return new_file


def main():
    while True:
        # this file name is supposed not to include space or other illegal characters
        # otherwise, implement format_filename func to filter them.
        file = pick_up_photo(path)
        logging.info('picked up the photo file name is {}'.format(file))
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            continue
        logging.info('file_path is {}'.format(file_path))
        try:
            rate, label = label_image(input_mean=0, input_std=255, model_file='trained_model/new_mobile_model.tflite',
                                      label_file='trained_model/class_labels.txt', image=file_path)
            logging.info('get class is: {}, rate: {}'.format(label, rate))
            class_path = os.path.join('classified_data', label)
            new_file_name = append_rate_to_file_name(rate, file)
            class_photo(file_path, os.path.join(class_path, new_file_name))
        except UnidentifiedImageError as e:
            logging.info(e)
            class_path = os.path.join('classified_data', 'error_image')
            create_folder(class_path)
            logging.info('move error image {} to {}'.format(file_path, class_path))
            class_photo(file_path, class_path)
        except ValueError as e:
            logging.info(e)
            class_path = os.path.join('classified_data', 'value_error_image')
            create_folder(class_path)
            logging.info('move value error image {} to {}'.format(file_path, class_path))
            class_photo(file_path, class_path)


if __name__ == '__main__':
    main()
