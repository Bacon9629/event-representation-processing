import os
import sys

import numpy as np
from numpy import ndarray
import cv2
from tqdm import tqdm

from .BaseEventImageConverter import BaseEventImageConverter

class EventFrameConverter(BaseEventImageConverter):

    def __init__(self, width=320, height=240, interval=0.5):
        super().__init__(width=width, height=height, interval=interval)

    def _make_color_histo(self, events: ndarray):
        """
        simple display function that shows negative events as blue dots and positive as red one
        on a white background
        args :
            - events structured numpy array: timestamp, x, y, polarity.
            - img (numpy array, height x width x 3) optional array to paint event on.
            - width int.
            - height int.
        return:
            - img numpy array, height x width x 3).
        """

        img = 255 * np.ones((self.height, self.width, 3), dtype=np.uint8)
        if events.size:
            assert events['x'].max() < self.width, "out of bound events: x = {}, w = {}".format(events['x'].max(),
                                                                                                self.width)
            assert events['y'].max() < self.height, "out of bound events: y = {}, h = {}".format(events['y'].max(),
                                                                                                 self.height)

            ON_index = np.where(events['polarity'] == 1)
            img[events['y'][ON_index], events['x'][ON_index], :] = [30, 30, 220] * events['polarity'][ON_index][:,
                                                                                   None]  # red
            OFF_index = np.where(events['polarity'] == 0)
            img[events['y'][OFF_index], events['x'][OFF_index], :] = [200, 30, 30] * (events['polarity'][
                                                                                          OFF_index] + 1)[:,
                                                                                     None]  # blue
        return img

    def events_to_event_images(self, input_filepath: str, output_file_dir: str):

        if input_filepath.endswith('.aedat4'):
            events = self.aedat_reader(input_filepath)
        elif input_filepath.endswith('.npy') or input_filepath.endswith('.npz'):
            events = self.npy_reader(input_filepath)
        else:
            raise NotImplementedError("File type not supported.")

        aps_frames_NUM = self._get_frames_num_from_npy(events)
        interval = self.interval

        os.makedirs(output_file_dir, exist_ok=True)
        if os.path.exists(input_filepath):
            start_timestamp = events[0][0]

            # saving event images.
            for i in range(int(aps_frames_NUM)):
                start_index = np.searchsorted(events['timestamp'], int(start_timestamp) + i * interval * 1e6)
                end_index = np.searchsorted(events['timestamp'], int(start_timestamp) + (i + 1) * interval * 1e6)

                rec_events = events[start_index:end_index]

                event_image = self._make_color_histo(rec_events)
                save_path = output_file_dir + '/{:08d}.png'.format(i)

                cv2.imwrite(save_path, event_image)


import glob
import re


def process_my_data(interval):
    return
    converter = EventFrameConverter(interval=interval)
    data_root = "/media/2TB_1/dataset/DailyDVS-200" if sys.platform == 'linux' else r"E:\dataset\DailyDvs-200"
    out_root = "/media/2TB_1/Bacon/dataset/DailyDvs-200" if sys.platform == 'linux' else r"E:\dataset\DailyDvs-200"
    file_list = glob.glob(fr"{data_root}/DailyDvs-200/*/*.aedat4")

    train_txt = rf"{data_root}/description/train.txt"
    val_txt = rf"{data_root}/description/val.txt"
    test_txt = rf"{data_root}/description/test.txt"
    with open(train_txt) as f:
        raw_train_file_list = f.readlines()
    with open(val_txt) as f:
        raw_val_file_list = f.readlines()
    with open(test_txt) as f:
        raw_test_file_list = f.readlines()

    train_file_dict = {re.search(r'/(.*?\.aedat4)', item).group(1): int(item.split(' ')[1]) for item in
                       raw_train_file_list}  # {'C47P48M1S2_20231203_19_44_04': label_id}
    val_file_dict = {re.search(r'/(.*?\.aedat4)', item).group(1): int(item.split(' ')[1]) for item in raw_val_file_list}
    test_file_dict = {re.search(r'/(.*?\.aedat4)', item).group(1): int(item.split(' ')[1]) for item in
                      raw_test_file_list}

    output_file_list = []
    for item in file_list:
        # 取得item的檔案名稱
        file_name = item.split(os.sep)[-1]
        if file_name in train_file_dict:
            is_which = "train"
            label = train_file_dict[file_name]
        elif file_name in val_file_dict:
            is_which = "val"
            label = val_file_dict[file_name]
        elif file_name in test_file_dict:
            is_which = "test"
            label = test_file_dict[file_name]
        else:
            raise ValueError
        output_file_list.append(
            rf"{out_root}/event_count_interval_{interval}/{is_which}/{label}/{file_name.replace('.aedat4', '')}")

    print(output_file_list[0])

    for in_path, out_path in tqdm(zip(file_list, output_file_list), total=len(file_list)):
        converter.events_to_event_images(input_filepath=in_path, output_file_dir=out_path)


# def one_data():
#     converter = EventImageConvert(width=346, height=260, interval=0.5)
#     in_path = r"E:\dataset\SeAct\m6-yexin\0006-2023_10_29_15_59_11.aedat4"
#     out_path = rf"E:\dataset\SeAct_m\m6-yexin\intervel_{converter.interval}"
#     converter._events_to_event_images(input_filename=in_path, output_file=out_path)


if __name__ == '__main__':
    # one_data()
    # process_my_data(interval=0.125)
    # process_my_data(interval=0.25)
    converter = EventFrameConverter(interval=0.5)
    in_path = r"E:\dataset\DailyDvs-200\test\011\C11P4M0S3_20231116_10_59_47.npz.npy"
    output_path = r"E:\dataset\DailyDvs-200\test\011\C11P4M0S3_20231116_10_59_47.npz.npy"

