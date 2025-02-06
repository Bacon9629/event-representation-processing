# Read .aedat4 file from DAVIS346 at the temporal window (t1-t2), generating frames.

import os
import sys
import abc

import numpy as np
from numpy import ndarray
from dv import AedatFile
import cv2
import math
from tqdm import tqdm


class BaseEventImageConverter(abc.ABC):

    def __init__(self, width=320, height=240, interval=0.5):
        """

        :param width: event camera width
        :param height: event camera height
        :param interval: merge events whose time difference is less than this value, some representation no need
        """
        self.width = width
        self.height = height

        self.interval = interval

    @staticmethod
    def aedat_reader(input_filepath: str) -> ndarray:
        """
        Convert .aedat file to numpy array
        :param input_filepath: aedat file path
        :return: ndarray, contain keys: "timestamp", "x", "y", "polarity"
        """
        with AedatFile(input_filepath) as f:
            events = np.hstack([event for event in f['events'].numpy()])

        return events

    @staticmethod
    def npy_reader(input_filepath: str) -> ndarray:
        """
        Convert .npy file to numpy structured array.
        :param input_filepath: npy file path
        :return: ndarray with fields: "timestamp", "x", "y", "polarity"
        """
        events = np.load(input_filepath, allow_pickle=True)

        if isinstance(events, dict):
            if 'timestamp' not in events:
                if 't' in events:
                    events['timestamp'] = events.pop('t')
            if 'polarity' not in events:
                if 'p' in events:
                    events['polarity'] = events.pop('p')

            required_keys = ['timestamp', 'x', 'y', 'polarity']
            if not all(key in events for key in required_keys):
                raise ValueError(f"Missing required keys in events. Required keys: {required_keys}")

            structured_array = np.core.records.fromarrays(
                [events['timestamp'], events['x'], events['y'], events['polarity']],
                names='timestamp, x, y, polarity'
            )
            return structured_array

        elif isinstance(events, np.ndarray) and events.dtype.names:
            fields = events.dtype.names
            rename_dict = {}
            if 't' in fields and 'timestamp' not in fields:
                rename_dict['t'] = 'timestamp'
            if 'p' in fields and 'polarity' not in fields:
                rename_dict['p'] = 'polarity'

            if rename_dict:
                new_dtype = [(rename_dict.get(name, name), events.dtype[name]) for name in fields]
                events = events.astype(new_dtype)
            return events

        else:
            raise ValueError("Unsupported data format in npy file. Expected dict or structured array.")

    def _get_frames_num_from_npy(self, event_raw_data):
        """
            Get the frame count for each event data
        """

        timestamps = [t[0] for t in event_raw_data]
        return math.ceil((max(timestamps) - min(timestamps)) * 1e-6 / self.interval)

    @abc.abstractmethod
    def events_to_event_images(self, input_filepath: str, output_file_dir: str):
        """
        Mapping asynchronous events into event images
        ex:
        ```
            if input_filepath.endswith('.aedat4'):
                events = self.aedat_reader(input_filepath)
            elif input_filepath.endswith('.npy') or input_filepath.endswith('.npz'):
                events = self.npy_reader(input_filepath)
            else:
                raise NotImplementedError("File type not supported.")
        ```

        args :
            - input_filepath:.aedat file, saving dvs events.
            - output_file_dir: the output filename saving timestamps.
        """
        return NotImplementedError("Please implement this method")