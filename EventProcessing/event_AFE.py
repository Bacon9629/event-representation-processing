import os
import sys
from typing import Literal

import numpy as np
from numpy import ndarray
import cv2
from tqdm import tqdm

from .BaseEventImageConverter import BaseEventImageConverter


class EventAFEConverter(BaseEventImageConverter):

    def __init__(self, width: int = 320, height: int = 240, interval: float = 0.5,
                 representation: Literal['rgb', 'gray_scale'] = 'rgb',
                 sample_event_threshold: int = 40,
                 sample_event_num_min: int = 100000
                 ):
        """

        :param width:
        :param height:
        :param interval:
        :param representation:
        :param sample_event_threshold: Hyperparameter, the author indicates that 40 is the best parameter of the data set 'SeAct', the author indicates that the parameter values of each data set are different.
        :param sample_event_num_min: Hyperparameter, the author indicates that 100,000 is the best parameter of the data set 'SeAct', the author indicates that the parameter values of each data set are different.
        """
        super().__init__(width=width, height=height, interval=interval)
        self.representation = representation

        # AFE Hyperparameter define
        self.sample_event_threshold = sample_event_threshold
        self.sample_event_num_min = sample_event_num_min
        # AFE Hyperparameter define END

    def generate_abs_image(self, events_clip):
        """
        generate event image without normalization, resize
        """
        x, y, t, p = events_clip.T
        x = x.astype(np.int32)
        y = y.astype(np.int32)

        w_event = 261
        h_event = 346
        img_pos = np.zeros((h_event * w_event,), dtype="float32")
        img_neg = np.zeros((h_event * w_event,), dtype="float32")
        np.add.at(img_pos, x[p == 1] + w_event * y[p == 1], 1)
        np.add.at(img_neg, x[p == 0] + w_event * y[p == 0], 1)

        img_pos = img_pos.reshape((h_event, w_event, 1))
        img_neg = img_neg.reshape((h_event, w_event, 1))

        # denoising using morphological transformation
        kernel_dilate = np.ones((2, 2), np.uint8)
        kernel_erode = np.ones((2, 2), np.uint8)
        img_pos = cv2.erode(img_pos, kernel_erode, iterations=1)
        img_neg = cv2.erode(img_neg, kernel_erode, iterations=1)
        img_pos = cv2.dilate(img_pos, kernel_dilate, iterations=1)
        img_neg = cv2.dilate(img_neg, kernel_dilate, iterations=1)

        img_pos = img_pos.reshape((h_event, w_event, 1))
        img_neg = img_neg.reshape((h_event, w_event, 1))

        gray_scale = (img_pos + img_neg) * [1, 1, 1]

        return gray_scale

    def if_sufficiently_sampled(self, events_stream):
        N, _ = events_stream.shape
        # print(N)
        half_N = int(np.floor(N / 2))
        # print(half_N)
        half_frame1 = self.generate_abs_image(events_stream[:half_N, :])
        # print(half_frame1.sum())
        half_frame2 = self.generate_abs_image(events_stream[half_N:, :])
        diff_image = np.abs(half_frame1 - half_frame2)
        idx = 200 * (diff_image.sum() / N)
        # print(idx)

        if idx <= self.sample_event_threshold:
            return True
        else:
            return False

    def generate_event_image(self, events_clip, shape, representation):
        """
        events_clip: N x 4, where cols are x, y, t, polarity, and polarity is in {0,1}.
        x and y correspond to image coordinates u and v.
        """
        H, W = shape
        x, y, t, p = events_clip.T
        x = x.astype(np.int32)
        y = y.astype(np.int32)

        w_event = x.max() + 1
        h_event = y.max() + 1
        img_pos = np.zeros((h_event * w_event,), dtype="float32")
        img_neg = np.zeros((h_event * w_event,), dtype="float32")
        np.add.at(img_pos, x[p == 1] + w_event * y[p == 1], 1)
        np.add.at(img_neg, x[p == -1] + w_event * y[p == -1], 1)

        img_pos = img_pos.reshape((h_event, w_event, 1))
        img_neg = img_neg.reshape((h_event, w_event, 1))

        # denoising using morphological transformation
        kernel_dilate = np.ones((2, 2), np.uint8)
        kernel_erode = np.ones((2, 2), np.uint8)
        img_pos = cv2.erode(img_pos, kernel_erode, iterations=1)
        img_neg = cv2.erode(img_neg, kernel_erode, iterations=1)
        img_pos = cv2.dilate(img_pos, kernel_dilate, iterations=1)
        img_neg = cv2.dilate(img_neg, kernel_dilate, iterations=1)

        img_pos = img_pos.reshape((h_event, w_event, 1))
        img_neg = img_neg.reshape((h_event, w_event, 1))

        if representation.lower().replace("_", "") == 'rgb':
            event_frame = 255 * (1 - (img_pos.reshape((h_event, w_event, 1)) * [0, 255, 255] + img_neg.reshape(
                (h_event, w_event, 1)) * [255, 255, 0]) / 255)
        elif representation.lower().replace("_", "") == 'grayscale':
            event_frame = (np.clip(img_pos, 0, 1) + np.clip(img_neg, 0, 1)) * [127.5, 127.5, 127.5]
        else:
            raise NotImplementedError("Representation not supported.")

        event_frame = np.clip(event_frame, 0, 255)
        # scale
        scale = H * 1.0 / h_event
        scale2 = W * 1.0 / w_event
        event_frame = cv2.resize(event_frame, dsize=None, fx=scale2, fy=scale)

        return event_frame

    def adaptive_event_sampling(self, events_stream):

        N, _ = events_stream.shape
        divide_N = int(np.floor(N / 2))

        if self.if_sufficiently_sampled(events_stream):  # return True for sufficiently sampled, no need for proceed.
            current_frame = self.generate_event_image(events_stream, (self.height, self.width), self.representation)
            # print('N:'+ str(N))
            return [current_frame]

        if divide_N <= self.sample_event_num_min:  # return the event frame if the event number is smaller than the default minimum number.
            half_frame1 = self.generate_event_image(events_stream[:divide_N, :], (self.height, self.width),
                                                    self.representation)
            half_frame2 = self.generate_event_image(events_stream[divide_N:, :], (self.height, self.width),
                                                    self.representation)
            return [half_frame1, half_frame2]

        # For unsufficiently sampled and divide_N > self.sample_event_num_min,
        # evenly divide the event stream and then sampled the two divided event streams recursively.
        frame_list1 = self.adaptive_event_sampling(events_stream[:divide_N, :])
        frame_list2 = self.adaptive_event_sampling(events_stream[divide_N:, :])
        return np.concatenate((frame_list1, frame_list2), axis=0)

    def events_to_event_images(self, input_filepath: str, output_file_dir: str):
        if not os.path.exists(input_filepath):
            raise FileNotFoundError("File not found: {}".format(input_filepath))

        if input_filepath.endswith('.aedat4'):
            events = self.aedat_reader(input_filepath)
        elif input_filepath.endswith('.npy') or input_filepath.endswith('.npz'):
            events = self.npy_reader(input_filepath)
        else:
            raise NotImplementedError("File type not supported.")

        ts = events['timestamp'].tolist()
        x = events['x'].tolist()
        y = events['y'].tolist()
        pol = events['polarity'].tolist()
        events_stream = np.array([x, y, ts, pol]).transpose()

        all_frame = self.adaptive_event_sampling(events_stream)
        all_frame = np.array(all_frame)  # T,H,W,3

        os.makedirs(output_file_dir, exist_ok=True)
        for index, frame in enumerate(all_frame):
            cv2.imwrite(os.path.join(output_file_dir, '{:08d}.png'.format(index)), frame)
            # # Show the accumulated image
            # cv2.imshow("Preview", frame)
            # cv2.waitKey(0)


if __name__ == '__main__':
    converter = EventAFEConverter(interval=0.5)
    in_path = r"E:\dataset\DailyDvs-200\test\011\C11P4M0S3_20231116_10_59_47.npz.npy"
    output_path = r"E:\dataset\DailyDvs-200\test\011\C11P4M0S3_20231116_10_59_47.npz.npy"
    converter.events_to_event_images(input_filepath=in_path, output_file_dir=output_path)
