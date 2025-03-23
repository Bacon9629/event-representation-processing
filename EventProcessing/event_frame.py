import os
import sys

import numpy as np
from numpy import ndarray
import cv2
from tqdm import tqdm

from .BaseEventImageConverter import BaseEventImageConverter

class EventFrameConverter(BaseEventImageConverter):

    def __init__(self, width=320, height=240, interval=0.5):
        """

        :param width: event camera width
        :param height: event camera height
        :param interval: merge events whose time difference is less than this value
        """
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
        if not os.path.exists(input_filepath):
            raise FileNotFoundError("File not found: {}".format(input_filepath))

        if input_filepath.endswith('.aedat4') or input_filepath.endswith('.aedat'):
            events = self.aedat_reader(input_filepath)
        elif input_filepath.endswith('.npy') or input_filepath.endswith('.npz'):
            events = self.npy_reader(input_filepath)
        else:
            raise NotImplementedError("File type not supported.")

        aps_frames_NUM = self._get_frames_num_from_npy(events)
        interval = self.interval

        os.makedirs(output_file_dir, exist_ok=True)
        start_timestamp = events[0][0]

        # saving event images.
        for i in range(int(aps_frames_NUM)):
            start_index = np.searchsorted(events['timestamp'], int(start_timestamp) + i * interval * 1e6)
            end_index = np.searchsorted(events['timestamp'], int(start_timestamp) + (i + 1) * interval * 1e6)

            rec_events = events[start_index:end_index]

            event_image = self._make_color_histo(rec_events)
            save_path = output_file_dir + '/{:08d}.png'.format(i)

            cv2.imwrite(save_path, event_image)


if __name__ == '__main__':
    converter = EventFrameConverter(interval=0.5)
    in_path = r"E:\dataset\DailyDvs-200\test\011\C11P4M0S3_20231116_10_59_47.npz.npy"
    output_path = r"E:\dataset\DailyDvs-200\test\011\C11P4M0S3_20231116_10_59_47.npz.npy"
    converter.events_to_event_images(input_filepath=in_path, output_file_dir=output_path)

