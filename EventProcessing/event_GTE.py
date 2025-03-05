import os
from typing import Tuple, Literal

import cv2
import numpy as np
import torch

from .BaseEventImageConverter import BaseEventImageConverter


class EventGTEConverter(BaseEventImageConverter):

    def __init__(self, width: int = 320, height: int = 240,
                 output_npy_or_frame: Literal['npy', 'ori_frame', 'enhancement_frame'] = 'npy',
                 patch_size: Tuple[int, int] = (4, 4),
                 group_num: int = 12,
                 ):
        """

        :param width: event camera width
        :param height: event camera height
        :param output_npy_or_frame: Choose what data format to output
        :param patch_size: Hyperparameter, The author sets it in the repo provided by the original paper as (4, 4)
        :param group_num: Hyperparameter, The author sets it to 12 in the repo provided by the original paper
        """
        super().__init__(width=width, height=height, interval=0)
        self.output_npy_or_frame = output_npy_or_frame.lower().strip()
        self.H = height
        self.W = width

        # GTE Hyperparameter define
        self.patch_size = patch_size
        self.group_num = group_num
        self.time_div = group_num // 2
        self.max_point = 500000  # define by dailydvs-200 repo
        # GTE Hyperparameter define END

    @staticmethod
    def index_mapping(sample, bins=None):
        """
        Multi-index mapping method from N-D to 1-D.

        假設有n個event

        @para sample: shape(4, n), [時間分區, event polarity, patch內pixel index, patch index]
        @para bins: shape(4), [時間分區數量, 2(polarity), patch內pixel數量, patch數量]

        從這邊可發現sample與bins是對映的，且bins是sample每個row的最大值
        """
        device = sample.device
        bins = torch.as_tensor(bins).to(device)
        # sample一定比0大、比bins小
        y = torch.max(sample, torch.zeros([], device=device, dtype=torch.int32))
        y = torch.min(y, bins.reshape(-1, 1))
        index = torch.ones_like(bins)
        index[1:] = torch.cumprod(torch.flip(bins[1:], [0]), -1).int()
        """
        index[1:] = [
            patch數量, 
            patch數量 * patch內pixel數量,
            patch數量 * patch內pixel數量 * 2(polarity),
        ]
        """
        index = torch.flip(index, [0])
        """
        index也代表的4-D result matrix每個dim的長度
        index = [
            時間分區數量,
            patch數量 * patch內pixel數量 * 2(polarity),
            patch數量 * patch內pixel數量,
            patch數量, 
        ]

        """
        l = torch.sum((index.reshape(-1, 1)) * y, 0)
        """
        shape[4, n] = index.shape[4, 1] * y.shape[4, n]
        [
            時間分區 * 時間分區數量, 
            event polarity * patch數量 * patch內pixel數量 * 2(polarity), 
            patch內pixel index * patch數量 * patch內pixel數量, 
            patch index * patch數量
        ]
        l.shape[n] = sum(shape[4, n], dim=0)
        l計算了每個event在result matrix中的1-D位置，用這個1-D index製作的1-D result matrix，可以快速轉成4-D matrix
        """
        return l, index

    @staticmethod
    def get_repr(l, index, bins=None, weights=None):
        """
        Function to return histograms.
        """
        # 計算pixel space下哪個地方有event的mask，1=有
        hist = torch.bincount(l, minlength=index[0] * bins[0], weights=weights[0])  # weights大概都是1，不用管
        hist = hist.reshape(tuple(bins))
        if len(weights) > 1:
            hist2 = torch.bincount(l, minlength=index[0] * bins[0], weights=weights[1])  # 時間越後面的event weight就越高?
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

    def forward(self, x):  # Input: x → [N, 4] tensor, each row is (t, x, y, p).
        """
        Given a set of events, return event tokens.
        return: (batch, group_num*polarity*patch內pixel, H上的patch數量, W上的patch數量)
        """
        # print(self.patch_size)
        # print('******* input token start********')
        # [t w h p]
        x = x[x != torch.inf].reshape(-1, 4)  # remove padding
        # print('************')
        # print(x)
        # print('***********')
        PH, PW = int((self.H + 1) / self.patch_size[0]), int((self.W + 1) / self.patch_size[1])
        """
        PH, PW: 計算在H和W上，總共有幾個patch
        """
        # print(self.H, self.W)
        # print(PH, PW)

        Token_num, Patch_size, b = int(PH * PW), int(self.patch_size[0] * self.patch_size[1]), 1e-4
        """
        Token_num: 計算總共有幾個patch，有幾個patch就有幾個token
        Patch_size: 計算一個patch有幾個pixel
        b: 1e-4
        """
        # self.time_div = 6  [6,2,2,4*4, H/4 * W/4]
        y = torch.zeros([self.time_div, 2, 2, Patch_size, Token_num], dtype=x[0].dtype, device=x[0].device)
        if len(x):
            w = x[:, 3] != 2
            # (t - t0)/（t_end - t_0） relative_time
            wt = torch.div(x[:, 0] - x[0, 0], x[-1, 0] - x[0, 0] + 1e-4)  # normalize time, mapping time to 0-1
            # (0, W) -> (0, PW - 1)
            # 計算每一個event對應到result matrix中的哪一個位置，這裡算出來的位置是把result matrix轉成1-D後的位置，到後面就可以直接reshape成2-D
            Position = torch.div(x[:, 1], (self.W / PW + b), rounding_mode='floor') + \
                       torch.div(x[:, 2], (self.H / PH + b), rounding_mode='floor') * PW
            # 計算每一個event會被對應到patch內的哪個position，一樣是把result matrix轉成1-D後的位置，到後面就可以直接reshape成2-D
            # Warning!!!, 原始論文在(self.W / PW)加上1e-4的話會造成bug，刪掉他就沒事了
            Token = torch.floor(x[:, 1] % (self.W / PW)) + \
                    torch.floor(x[:, 2] % (self.H / PH)) * int((self.W + 1) / PW)
            # print(Token, Token.size)
            t_double = x[:, 0].double()
            # 計算每一個event應該要對應到哪一個時間分區，利用每一個event自己的時間scale到對應的時間分區
            # 時間分區: 把一段event video切成多個等份，這個等份就是時間分區
            DTime = torch.floor(self.time_div * torch.div(t_double - t_double[0], t_double[-1] - t_double[0] + 1))

            # Mapping from 4-D to 1-D.
            bins = torch.as_tensor((self.time_div, 2, Patch_size, Token_num)).to(x.device)
            """
            bins: [時間分區數量, 2(polarity), patch內pixel數量, patch數量]
            """
            x_nd = torch.cat([DTime.unsqueeze(1), x[:, 3].unsqueeze(1), Token.unsqueeze(1), Position.unsqueeze(1)],
                             dim=1).permute(1, 0).int()
            x_1d, index = self.index_mapping(x_nd, bins)
            """
            x_1d: 計算了每個event在result matrix中的1-D位置，用這個1-D index製作的1-D result matrix，可以快速轉成4-D matrix
            index: 代表的4-D result matrix每個dim的長度，方便將1-D index mapping回4-D
            index = [
                時間分區數量,
                patch數量 * patch內pixel數量 * 2(polarity),
                patch數量 * patch內pixel數量,
                patch數量, 
            ]
            """

            # Get 1-D histogram which encodes the event tokens.
            # print('**************')
            # print(f"bins:{bins}")
            # print(f"index:{index}")
            # print(f"x_1d:{x_1d}, {len(x_1d)}")
            # print('**************')
            y[:, :, 0, :, :], y[:, :, 1, :, :] = self.get_repr(x_1d, index, bins=bins, weights=[w, wt])
            """
            把1-D event in pixel space index mapping轉換成4-D pixel space data，計算在各個時間區段內、各個patch上、各個pixel上的index數量  
            y[:, :, 0, :, :] : event weight大概都是1，不用管 (代表每個event都一樣重要) (這是基於polarity是0 or 1 ， 若polarity有2的話就不一樣了)
            y[:, :, 1, :, :] : event weight會隨著時間越後面，event的weight就越高 (代表越後面的event數值佔比越高)
            y.shape: (時間分區數量, 2(polarity), 2(前段提到), patch內pixel數量, patch數量)
            """

            """
            y = [時間分區數量, 2(polarity), 2(前段提到), patch內pixel數量, patch數量(PH*PW)]
#return (Batch, |____________________合併成channel___________________|, PH, PW)
            return: (batch, group_num*polarity*patch內pixel, patch數量H, patch數量W)
            group_num: 時間區段*2 (假設group_num=12，code計算的方法是分成6個時間區段，最後在透過get_repr算出兩個channel，return時把兩個channel合併)
            """
        # print('******* input token end ********')

        # Output: y → [1, group_num * '2' * (patch_size ** 2), H // patch_size, W // patch_size] tensor.
        return y.reshape(1, -1, PH, PW)


    def events_to_event_images(self, input_filepath: str, output_file_dir: str = None):
        if not os.path.exists(input_filepath):
            raise FileNotFoundError("File not found: {}".format(input_filepath))

        if input_filepath.endswith('.aedat4'):
            events = self.aedat_reader(input_filepath)
        elif input_filepath.endswith('.npy') or input_filepath.endswith('.npz'):
            events = self.npy_reader(input_filepath)
        else:
            raise NotImplementedError("File type not supported.")

        if len(events['timestamp']) > self.max_point:
            # 把bias的中間設定在events的中間
            bias = int((len(events['timestamp']) - self.max_point) / 2)
            t = torch.tensor(events['timestamp'][bias:bias + self.max_point]).unsqueeze(1)
            x = torch.tensor(events['x'][bias:bias + self.max_point]).unsqueeze(1)
            y = torch.tensor(events['y'][bias:bias + self.max_point]).unsqueeze(1)
            p = torch.tensor(events['polarity'][bias:bias + self.max_point]).unsqueeze(1)
        else:
            t = torch.tensor(events['timestamp']).unsqueeze(1)
            x = torch.tensor(events['x']).unsqueeze(1)
            y = torch.tensor(events['y']).unsqueeze(1)
            p = torch.tensor(events['polarity']).unsqueeze(1)

        from time_cost_record import CostRecord
        for i in range(100):
            with CostRecord(self.__class__.__name__):
                t = t[:, :] - t[0, 0]
                events = torch.cat([t, x, y, p], dim=1) / 1.0

                result = self.forward(events)[0].detach().cpu().numpy()  # [channel, H // patch_size, W // patch_size]


            if self.output_npy_or_frame == 'npy':
                if output_file_dir is not None:
                    os.makedirs(output_file_dir, exist_ok=True)
                    np.save(os.path.join(output_file_dir, "GTE_representation"), result)
                return
            elif 'frame' not in self.output_npy_or_frame:
                raise NotImplementedError("Unsupported output format..")

            """
            Data pipeline:
            result shape: (time_div, 2(polarity), 2(hist), patch內pixel, patch)
            result shape: (time_div, 2(polarity), 2(hist), patch內pixel y, patch內pixel x, patch y, patch x)
            result shape: (time_div, 2(polarity), 2(hist), patch y, patch內pixel y, patch x, patch內pixel x)
            result shape: (time_div, 2(polarity), 2(hist), image y, image x)
            """
            PH, PW = int((self.H + 1) / self.patch_size[0]), int((self.W + 1) / self.patch_size[1])
            result = result.reshape(self.time_div, 2, 2, self.patch_size[0], self.patch_size[1], PH, PW)
            result = result.transpose(0, 1, 2, 5, 3, 6, 4)
            result = result.reshape(self.time_div, 2, 2, self.H, self.W)

            result_polarity_positive_hist0 = result[:, 0, 0, :, :]  # shape = (time_div, y, x)
            result_polarity_positive_hist1 = result[:, 0, 1, :, :]  # shape = (time_div, y, x)
            result_polarity_negative_hist0 = result[:, 1, 0, :, :]  # shape = (time_div, y, x)
            result_polarity_negative_hist1 = result[:, 1, 1, :, :]  # shape = (time_div, y, x)

            result_hist1_fusion = (result_polarity_positive_hist1 + result_polarity_negative_hist1) / 2

            result_polarity_positive_hist0 = np.expand_dims(result_polarity_positive_hist0, axis=-1)
            result_polarity_negative_hist0 = np.expand_dims(result_polarity_negative_hist0, axis=-1)
            result_hist1_fusion = np.expand_dims(result_hist1_fusion, axis=-1)

            result = np.concatenate(
                (result_polarity_positive_hist0, result_polarity_negative_hist0, result_hist1_fusion),
                axis=-1)

            frame_list = []
            for index, frame in enumerate(result):
                frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)  # 確保數值範圍在 0-255
                frame = frame.astype(np.uint8)

                if self.output_npy_or_frame == 'ori_frame':
                    frame_list.append(frame)
                    continue

                if self.output_npy_or_frame != 'enhancement_frame':
                    raise NotImplementedError("Unsupported output format..")

                frame[:, :, 0] = cv2.equalizeHist(frame[:, :, 0])
                frame[:, :, 1] = cv2.equalizeHist(frame[:, :, 1])
                frame[:, :, 2] = cv2.equalizeHist(frame[:, :, 2])
                frame_list.append(frame)

            if output_file_dir is not None:
                os.makedirs(output_file_dir, exist_ok=True)
                for index, frame in enumerate(frame_list):
                    cv2.imwrite(os.path.join(output_file_dir, "{:08d}.png".format(index)), frame)
                    # cv2.imshow("frame", frame)
                    # cv2.waitKey(0)


if __name__ == '__main__':
    converter = EventGTEConverter(interval=0.5)
    in_path = r"E:\dataset\DailyDvs-200\test\011\C11P4M0S3_20231116_10_59_47.npz.npy"
    output_path = r"E:\dataset\DailyDvs-200\test\011\C11P4M0S3_20231116_10_59_47.npz.npy"
    converter.events_to_event_images(input_filepath=in_path, output_file_dir=output_path)
