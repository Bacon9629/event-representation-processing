import os
import sys
from copy import copy

import dv_processing as dv_p
import numpy as np
from numpy import ndarray
import cv2
from tqdm import tqdm

from .BaseEventImageConverter import BaseEventImageConverter

class EventSpeedInvariantTimeSurfaceConverter(BaseEventImageConverter):

    def __init__(self, width=320, height=240, interval=0.5):
        """

        :param width: event camera width
        :param height: event camera height
        :param interval: merge events whose time difference is less than this value
        """
        super().__init__(width=width, height=height, interval=interval)
        self.time_window = int(interval * 1_000_000)  # 微秒

    def events_to_event_images(self, input_filepath: str, output_file_dir: str):
        if not os.path.exists(input_filepath):
            raise FileNotFoundError("File not found: {}".format(input_filepath))

        if input_filepath.endswith('.aedat4') or input_filepath.endswith('.aedat'):
            events = self.aedat_reader(input_filepath)
        elif input_filepath.endswith('.npy') or input_filepath.endswith('.npz'):
            events = self.npy_reader(input_filepath)
        else:
            raise NotImplementedError("File type not supported.")

        accumulator = dv_p.SpeedInvariantTimeSurface((self.width, self.height))
        store = dv_p.EventStore()
        result_frames = []

        for event in events[0:]:
            store.push_back(event['timestamp'], event['x'], event['y'], event['polarity'])
            if store.getHighestTime() - store.getLowestTime() < self.time_window:
                continue

            accumulator.accept(store)
            result_frames.append(copy(accumulator.generateFrame().image))
            accumulator.reset()
            store = dv_p.EventStore()

            # # Show the accumulated image
            # cv2.imshow("Preview", result_frames[-1])
            # cv2.waitKey(0)

        if store.size() > 0:
            accumulator.accept(store)
            result_frames.append(accumulator.generateFrame().image)

        os.makedirs(output_file_dir, exist_ok=True)
        for index, image in enumerate(result_frames):
            cv2.imwrite(os.path.join(output_file_dir, '{:08d}.png'.format(index)), image)


if __name__ == '__main__':
    converter = EventSpeedInvariantTimeSurfaceConverter(interval=0.5)
    in_path = r"E:\dataset\DailyDvs-200\test\011\C11P4M0S3_20231116_10_59_47.npz.npy"
    output_path = r"E:\dataset\DailyDvs-200\test\011\C11P4M0S3_20231116_10_59_47.npz.npy"
    converter.events_to_event_images(input_filepath=in_path, output_file_dir=output_path)

