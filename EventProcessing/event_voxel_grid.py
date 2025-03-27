import os
from typing import Literal

import cv2
import numpy as np
import torch

from .BaseEventImageConverter import BaseEventImageConverter


class EventVoxelGridConverter(BaseEventImageConverter):

    def __init__(self, width: int = 320, height: int = 240,
                 output_npy_or_frame: Literal['npy', 'ori_frame', 'enhancement_frame'] = 'npy',
                 voxel_bin_num: int = 9,
                 ):
        """

        :param width: event camera width
        :param height: event camera height
        :param output_npy_or_frame: Choose what data format to output
        :param voxel_bin_num: Hyperparameter, The author sets it to 9 in the repo provided by the original paper
        """
        super().__init__(width=width, height=height, interval=voxel_bin_num)
        self.output_npy_or_frame = output_npy_or_frame.lower().strip()
        self.H = height
        self.W = width

        # GTE Hyperparameter define
        self.voxel_bin_num = voxel_bin_num
        # GTE Hyperparameter define END

    @staticmethod
    def interpolate_to_image(pxs, pys, dxs, dys, weights, img):
        """
        Accumulate x and y coords to an image using bilinear interpolation
        @param pxs Numpy array of integer typecast x coords of events
        @param pys Numpy array of integer typecast y coords of events
        @param dxs Numpy array of residual difference between x coord and int(x coord)
        @param dys Numpy array of residual difference between y coord and int(y coord)
        @returns Image
        """
        img.index_put_((pys, pxs), weights * (1.0 - dxs) * (1.0 - dys), accumulate=True)
        img.index_put_((pys, pxs + 1), weights * dxs * (1.0 - dys), accumulate=True)
        img.index_put_((pys + 1, pxs), weights * (1.0 - dxs) * dys, accumulate=True)
        img.index_put_((pys + 1, pxs + 1), weights * dxs * dys, accumulate=True)
        return img

    def events_to_image_torch(self, xs, ys, ps,
                              device=None, sensor_size=(180, 240), clip_out_of_range=True,
                              interpolation=None, padding=True, default=0):
        """
        Method to turn event tensor to image. Allows for bilinear interpolation.
        @param xs Tensor of x coords of events
        @param ys Tensor of y coords of events
        @param ps Tensor of event polarities/weights
        @param device The device on which the image is. If none, set to events device
        @param sensor_size The size of the image sensor/output image
        @param clip_out_of_range If the events go beyond the desired image size,
           clip the events to fit into the image
        @param interpolation Which interpolation to use. Options=None,'bilinear'
        @param padding If bilinear interpolation, allow padding the image by 1 to allow events to fit:
        @returns Event image from the events
        """
        if device is None:
            device = xs.device
        if interpolation == 'bilinear' and padding:
            img_size = (sensor_size[0] + 1, sensor_size[1] + 1)
        else:
            img_size = list(sensor_size)

        mask = torch.ones(xs.size(), device=device)
        if clip_out_of_range:
            zero_v = torch.tensor([0.], device=device)
            ones_v = torch.tensor([1.], device=device)
            clipx = img_size[1] if interpolation is None and padding == False else img_size[1] - 1
            clipy = img_size[0] if interpolation is None and padding == False else img_size[0] - 1
            mask = torch.where(xs >= clipx, zero_v, ones_v) * torch.where(ys >= clipy, zero_v, ones_v)

        img = (torch.ones(img_size) * default).to(device)
        if interpolation == 'bilinear' and xs.dtype is not torch.long and xs.dtype is not torch.long:
            pxs = (xs.floor()).float()
            pys = (ys.floor()).float()
            dxs = (xs - pxs).float()
            dys = (ys - pys).float()
            pxs = (pxs * mask).long()
            pys = (pys * mask).long()
            masked_ps = ps.squeeze() * mask
            self.interpolate_to_image(pxs, pys, dxs, dys, masked_ps, img)
        else:
            if xs.dtype is not torch.long:
                xs = xs.long().to(device)
            if ys.dtype is not torch.long:
                ys = ys.long().to(device)
            try:
                mask = mask.long().to(device)
                xs, ys = xs * mask, ys * mask
                img.index_put_((ys, xs), ps, accumulate=True)
            except Exception as e:
                print("Unable to put tensor {} positions ({}, {}) into {}. Range = {},{}".format(
                    ps.shape, ys.shape, xs.shape, img.shape, torch.max(ys), torch.max(xs)))
                raise e
        return img

    def events_to_image(self, xs, ys, ps, sensor_size=(180, 240), interpolation=None, padding=False, meanval=False,
                        default=0):
        """
        Place events into an image using numpy
        @param xs x coords of events
        @param ys y coords of events
        @param ps Event polarities/weights
        @param sensor_size The size of the event camera sensor
        @param interpolation Whether to add the events to the pixels by interpolation (values: None, 'bilinear')
        @param padding If true, pad the output image to include events otherwise warped off sensor
        @param meanval If true, divide the sum of the values by the number of events at that location
        @returns Event image from the input events
        """
        img_size = (sensor_size[0] + 1, sensor_size[1] + 1)
        if interpolation == 'bilinear' and xs.dtype is not torch.long and xs.dtype is not torch.long:
            xt, yt, pt = torch.from_numpy(xs), torch.from_numpy(ys), torch.from_numpy(ps)
            xt, yt, pt = xt.float(), yt.float(), pt.float()
            img = self.events_to_image_torch(xt, yt, pt, clip_out_of_range=True, interpolation='bilinear',
                                             padding=padding)
            img[img == 0] = default
            img = img.numpy()
            if meanval:
                event_count_image = self.events_to_image_torch(xt, yt, torch.ones_like(xt),
                                                               clip_out_of_range=True, padding=padding)
                event_count_image = event_count_image.numpy()
        else:
            coords = np.stack((ys, xs))
            try:
                abs_coords = np.ravel_multi_index(coords, img_size)
            except ValueError:
                print("Issue with input arrays! minx={}, maxx={}, miny={}, maxy={}, coords.shape={}, \
                        sum(coords)={}, sensor_size={}".format(np.min(xs), np.max(xs), np.min(ys), np.max(ys),
                                                               coords.shape, np.sum(coords), img_size))
                raise ValueError
            img = np.bincount(abs_coords, weights=ps, minlength=img_size[0] * img_size[1])
            img = img.reshape(img_size)
            if meanval:
                event_count_image = np.bincount(abs_coords, weights=np.ones_like(xs),
                                                minlength=img_size[0] * img_size[1])
                event_count_image = event_count_image.reshape(img_size)
        if meanval:
            img = np.divide(img, event_count_image, out=np.ones_like(img) * default, where=event_count_image != 0)
        return img[0:sensor_size[0], 0:sensor_size[1]]

    def events_to_voxel(self, xs, ys, ts, ps, B, sensor_size=(180, 240), temporal_bilinear=True):
        """
        Turn set of events to a voxel grid tensor, using temporal bilinear interpolation
        @param xs List of event x coordinates (torch tensor)
        @param ys List of event y coordinates (torch tensor)
        @param ts List of event timestamps (torch tensor)
        @param ps List of event polarities (torch tensor)
        @param B Number of bins in output voxel grids (int)
        @param sensor_size The size of the event sensor/output voxels
        @param temporal_bilinear Whether the events should be naively
            accumulated to the voxels (faster), or properly
            temporally distributed
        @returns Voxel of the events between t0 and t1
        """
        assert (len(xs) == len(ys) and len(ys) == len(ts) and len(ts) == len(ps))
        num_events_per_bin = len(xs) // B
        bins = []
        dt = ts[-1] - ts[0]
        t_norm = (ts - ts[0]) / dt * (B - 1)
        zeros = (np.expand_dims(np.zeros(t_norm.shape[0]), axis=0).transpose()).squeeze()
        for bi in range(B):
            if temporal_bilinear:
                bilinear_weights = np.maximum(zeros, 1.0 - np.abs(t_norm - bi))
                weights = ps * bilinear_weights
                vb = self.events_to_image(xs.squeeze(), ys.squeeze(), weights.squeeze(),
                                          sensor_size=sensor_size, interpolation=None)
            else:
                raise NotImplementedError("In the code written by the author, variable 'weights' are not declared..., "
                                          "so I would like to prompt the user in this way first.")
                # beg = bi * num_events_per_bin
                # end = beg + num_events_per_bin
                # vb = self.events_to_image(xs[beg:end], ys[beg:end],
                #                      weights[beg:end], sensor_size=sensor_size)
            bins.append(vb)
        bins = np.stack(bins)
        return bins

    @staticmethod
    def standardize_and_convert_to_image(arr):
        mean = np.mean(arr)
        std = np.std(arr)
        standardized_arr = (arr - mean) / std if std != 0 else arr
        normalized_arr = (standardized_arr - standardized_arr.min()) / (standardized_arr.max() - standardized_arr.min()) * 255
        image_array = normalized_arr.astype(np.uint8)
        return image_array

    def events_to_event_images(self, input_filepath: str, output_file_dir: str = None):
        if not os.path.exists(input_filepath):
            raise FileNotFoundError("File not found: {}".format(input_filepath))

        if input_filepath.endswith('.aedat4'):
            events = self.aedat_reader(input_filepath)
        elif input_filepath.endswith('.npy') or input_filepath.endswith('.npz'):
            events = self.npy_reader(input_filepath)
        else:
            raise NotImplementedError("File type not supported.")

        ts = torch.tensor(events['timestamp'])
        x = torch.tensor(events['x'])
        y = torch.tensor(events['y'])
        pol = torch.tensor(events['polarity'])

        from time_cost_record import CostRecord
        with CostRecord(f"{self.__class__.__name__}_{self.interval}"):
            voxel_grid = self.events_to_voxel(xs=x, ys=y, ts=ts, ps=pol, B=self.voxel_bin_num, sensor_size=(self.H, self.W), temporal_bilinear=True)
            torch.cuda.synchronize()
            # print(voxel_grid.shape)  # (10, 240, 320)

        # save
        if self.output_npy_or_frame == 'npy':
            if output_file_dir is not None:
                os.makedirs(output_file_dir, exist_ok=True)
                np.save(os.path.join(output_file_dir, 'voxel_grid.npy'), voxel_grid)
            return
        elif "frame" not in self.output_npy_or_frame:
            raise NotImplementedError

        frame_list = []
        for index, frame in enumerate(voxel_grid):
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
            frame = frame.astype(np.uint8)

            if self.output_npy_or_frame == 'ori_frame':
                frame_list.append(frame)
            elif self.output_npy_or_frame == 'enhancement_frame':
                frame_list.append(cv2.equalizeHist(frame))
            else:
                raise NotImplementedError


        if output_file_dir is not None:
            os.makedirs(output_file_dir, exist_ok=True)
            for index, frame in enumerate(frame_list):
                # cv2.imwrite(os.path.join(output_file_dir, "{:08d}.png".format(index)), frame)
                pass

            # cv2.imshow("Preview", frame)
            # cv2.waitKey(0)


if __name__ == '__main__':
    converter = EventVoxelGridConverter(interval=0.5)
    in_path = r"E:\dataset\DailyDvs-200\test\011\C11P4M0S3_20231116_10_59_47.npz.npy"
    output_path = r"E:\dataset\DailyDvs-200\test\011\C11P4M0S3_20231116_10_59_47.npz.npy"
    converter.events_to_event_images(input_filepath=in_path, output_file_dir=output_path)
