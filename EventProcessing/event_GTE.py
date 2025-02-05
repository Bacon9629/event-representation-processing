import os
from typing import Tuple

import numpy as np
import torch

from .BaseEventImageConverter import BaseEventImageConverter


class EventGTEConverter(BaseEventImageConverter):

    def __init__(self, width: int = 320, height: int = 240, interval: float = 0.5,
                 patch_size: Tuple[int, int] = (4, 4),
                 group_num: int = 12,
                 ):
        """

        :param width:
        :param height:
        :param interval:
        :param patch_size: Hyperparameter, The author sets it in the repo provided by the original paper as (4, 4)
        :param group_num: Hyperparameter, The author sets it to 12 in the repo provided by the original paper
        """
        super().__init__(width=width, height=height, interval=interval)
        self.H = height
        self.W = width

        # GTE Hyperparameter define
        self.patch_size = patch_size
        self.group_num = group_num
        self.time_div = group_num // 2
        # GTE Hyperparameter define END

    @staticmethod
    def get_repr(l, index, bins=None, weights=None):
        """
        Function to return histograms.
        """
        hist = torch.bincount(l, minlength=index[0] * bins[0], weights=weights[0])
        hist = hist.reshape(tuple(bins))
        if len(weights) > 1:
            hist2 = torch.bincount(l, minlength=index[0] * bins[0], weights=weights[1])
            hist2 = hist2.reshape(tuple(bins))
        else:
            return hist
        if len(weights) > 2:
            hist3 = torch.bincount(l, minlength=index[0] * bins[0], weights=weights[2])
            hist3 = hist3.reshape(tuple(bins))
        else:
            return hist, hist2
        if len(weights) > 3:
            hist4 = torch.bincount(l, minlength=index[0] * bins[0], weights=weights[3])
            hist4 = hist4.reshape(tuple(bins))
        else:
            return hist, hist2, hist3
        return hist, hist2, hist3, hist4

    @staticmethod
    def index_mapping(sample, bins=None):
        """
        Multi-index mapping method from N-D to 1-D.
        """
        device = sample.device
        bins = torch.as_tensor(bins).to(device)
        y = torch.max(sample, torch.zeros([], device=device, dtype=torch.int32))
        y = torch.min(y, bins.reshape(-1, 1))
        index = torch.ones_like(bins)
        index[1:] = torch.cumprod(torch.flip(bins[1:], [0]), -1).int()
        index = torch.flip(index, [0])
        l = torch.sum((index.reshape(-1, 1)) * y, 0)
        return l, index

    def forward(self, x) -> torch.Tensor:  # Input: x → [N, 4] tensor, each row is (t, x, y, p).
        """
        Given a set of events, return event tokens.
        """
        x = x[x != torch.inf].reshape(-1, 4)  # remove padding
        PH, PW = int((self.H + 1) / self.patch_size[0]), int((self.W + 1) / self.patch_size[1])
        Token_num, Patch_size, b = int(PH * PW), int(self.patch_size[0] * self.patch_size[1]), 1e-4
        y = torch.zeros([self.time_div, 2, 2, Patch_size, Token_num], dtype=x[0].dtype, device=x[0].device)
        if len(x):
            w = x[:, 3] != 2
            wt = torch.div(x[:, 0] - x[0, 0], x[-1, 0] - x[0, 0] + 1e-4)

            Position = torch.div(x[:, 1], (self.W / PW + b), rounding_mode='floor') + \
                       torch.div(x[:, 2], (self.H / PH + b), rounding_mode='floor') * PW
            Token = torch.floor(x[:, 1] % (self.W / PW + 1e-4)) + \
                    torch.floor(x[:, 2] % (self.H / PH + b)) * int((self.W + 1) / PW)
            t_double = x[:, 0].double()
            DTime = torch.floor(self.time_div * torch.div(t_double - t_double[0], t_double[-1] - t_double[0] + 1))

            # Mapping from 4-D to 1-D.
            bins = torch.as_tensor((self.time_div, 2, Patch_size, Token_num)).to(x.device)
            x_nd = torch.cat([DTime.unsqueeze(1), x[:, 3].unsqueeze(1), Token.unsqueeze(1), Position.unsqueeze(1)],
                             dim=1).permute(1, 0).int()
            x_1d, index = self.index_mapping(x_nd, bins)

            # Get 1-D histogram which encodes the event tokens.
            # [self.time_div, polarity, channel, Patch_size, Token_num]]
            # [6, 2, 2, 16, 4800]
            y[:, :, 0, :, :], y[:, :, 1, :, :] = self.get_repr(x_1d, index, bins=bins, weights=[w, wt])
        else:
            raise ValueError("No events found.")

        # [1, channel, H // patch_size, W // patch_size]
        return y.reshape(1, -1, PH,
                         PW)  # Output: y → [1, group_num * '2' * (patch_size ** 2), H // patch_size, W // patch_size] tensor.


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
        events = np.array([ts, x, y, pol]).transpose()
        events = torch.tensor(events, dtype=torch.float32)

        result = self.forward(events)[0]  # [channel, H // patch_size, W // patch_size]

        result = result.detach().cpu().numpy()
        os.makedirs(output_file_dir, exist_ok=True)
        np.save(os.path.join(output_file_dir, "GTE_representation"), result)


if __name__ == '__main__':
    converter = EventGTEConverter(interval=0.5)
    in_path = r"E:\dataset\DailyDvs-200\test\011\C11P4M0S3_20231116_10_59_47.npz.npy"
    output_path = r"E:\dataset\DailyDvs-200\test\011\C11P4M0S3_20231116_10_59_47.npz.npy"
    converter.events_to_event_images(input_filepath=in_path, output_file_dir=output_path)
