import json
import os
import glob
import re
import sys
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from EventProcessing.BaseEventImageConverter import BaseEventImageConverter
from EventProcessing import EventFrameConverter
from EventProcessing import EventCountConverter
from EventProcessing import EventTimeSurfaceConverter
from EventProcessing import EventSpeedInvariantTimeSurfaceConverter
from EventProcessing import EventAFEConverter
from EventProcessing import EventVoxelGridConverter
from EventProcessing import EventGTEConverter


def load_file_descriptions(data_root):
    """
    Load train, val, and test file descriptions and return dictionaries.

    :return
        file_dicts: dict
        {
            'train': {file_name: label},
            'val': {file_name: label},
        }
        ex: 'm6-yexin/1020-2023_10_29_16_06_37' = {int} 43
            'm6-yexin/0015-2023_10_29_16_00_48' = {int} 15
            'm6-yexin/1003-2023_10_29_16_02_35' = {int} 26
    """
    description_paths = {
        'train': os.path.join(data_root, 'description', 'SeAct_train.txt'),
        'val': os.path.join(data_root, 'description', 'SeAct_val.txt'),
    }
    with open(os.path.join(data_root, 'description', 'SeAct_idx_to_label.json')) as f:
        id_label_mapping = json.load(f)

    file_dicts = {}
    for split, path in description_paths.items():
        with open(path) as f:
            raw_file_list = f.readlines()
        file_dicts[split] = {
            re.search(r'/.*?/(.*?\.aedat4)', item.replace(os.sep, "/")).group(1):
                id_label_mapping[str(int(re.findall(r"\d{4}", item)[0]))]
            for item in raw_file_list
        }
    return file_dicts


def __process_one_data(args):
    """
    Process a single file.
    :param args: Tuple containing (converter, input_filepath, output_file_dir).
    """
    converter, in_path, out_path = args
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if not os.path.exists(out_path):
            converter.events_to_event_images(input_filepath=in_path, output_file_dir=out_path)
    except Exception as e:
        print(f"Error processing {in_path}: {e}")


def process_dataset(converter: BaseEventImageConverter, file_extension, num_workers=None):
    """
    Generalized function to process .npy or .aedat4 files using parallel processing.
    """
    print(f"Using {converter.__class__.__name__} interval: {converter.interval}")

    data_root = "/media/2TB_1/dataset/SeAct" if sys.platform == 'linux' else r"E:/dataset/SeAct"
    out_root = "/media/2TB_1/Bacon/dataset/SeAct" if sys.platform == 'linux' else r"E:/dataset/SeAct"

    file_list = glob.glob(fr"{data_root}/SeAct_aedat/*/*.{file_extension}")
    file_dicts = load_file_descriptions(data_root)

    input_output_file_pairs = []

    for item in file_list:
        item = item.replace(os.sep, "/")
        # file_name = os.path.basename(item).split('.')[0]
        file_name = re.findall(r"SeAct_aedat/(.*?\.aedat4)", item)[0]
        found_in_split = False

        for split in ['train', 'val']:
            if file_name in file_dicts[split]:
                found_in_split = True
                label = file_dicts[split][file_name]
                input_output_file_pairs.append(
                    (
                        item,
                        rf"{out_root}/{converter.__class__.__name__}_interval_{converter.interval}/{split}/{label}/{file_name.replace('/', '-')}"
                    )
                )

        if not found_in_split:
            raise ValueError(f"{file_name} not found in description train/val/test")

    tasks = [(converter, in_path, out_path) for in_path, out_path in input_output_file_pairs]

    if num_workers is None:
        num_workers = 0

    if num_workers == 0:
        for task in tqdm(tasks):
            __process_one_data(task)
    else:
        with Pool(processes=num_workers) as pool:
            list(tqdm(pool.imap_unordered(__process_one_data, tasks), total=len(tasks)))


if __name__ == '__main__':
    # converter = EventFrameConverter(interval=0.5)
    # converter = EventCountConverter(interval=0.5)
    # converter = EventTimeSurfaceConverter(interval=0.5)
    # converter = EventSpeedInvariantTimeSurfaceConverter(interval=0.5)
    # converter = EventAFEConverter(interval=0.5, sample_event_threshold=40, sample_event_num_min=100000)
    # converter = EventVoxelGridConverter(interval=0.5, voxel_bin_num=9)
    # converter = EventGTEConverter(interval=0.5, patch_size=(4, 4), group_num=12)

    process_dataset(converter=EventFrameConverter(height=260, width=346, interval=0.75), file_extension='aedat4', num_workers=cpu_count())
    process_dataset(converter=EventFrameConverter(height=260, width=346, interval=0.5), file_extension='aedat4', num_workers=cpu_count())
    process_dataset(converter=EventFrameConverter(height=260, width=346, interval=0.25), file_extension='aedat4', num_workers=cpu_count())

    process_dataset(converter=EventCountConverter(height=260, width=346, interval=0.75), file_extension='aedat4', num_workers=cpu_count())
    process_dataset(converter=EventCountConverter(height=260, width=346, interval=0.5), file_extension='aedat4', num_workers=cpu_count())
    process_dataset(converter=EventCountConverter(height=260, width=346, interval=0.25), file_extension='aedat4', num_workers=cpu_count())

    process_dataset(converter=EventTimeSurfaceConverter(height=260, width=346, interval=0.75), file_extension='aedat4', num_workers=cpu_count())
    process_dataset(converter=EventTimeSurfaceConverter(height=260, width=346, interval=0.5), file_extension='aedat4', num_workers=cpu_count())
    process_dataset(converter=EventTimeSurfaceConverter(height=260, width=346, interval=0.25), file_extension='aedat4', num_workers=cpu_count())

    process_dataset(converter=EventSpeedInvariantTimeSurfaceConverter(height=260, width=346, interval=0.75), file_extension='aedat4',
                    num_workers=cpu_count())
    process_dataset(converter=EventSpeedInvariantTimeSurfaceConverter(height=260, width=346, interval=0.5), file_extension='aedat4',
                    num_workers=cpu_count())
    process_dataset(converter=EventSpeedInvariantTimeSurfaceConverter(height=260, width=346, interval=0.25), file_extension='aedat4',
                    num_workers=cpu_count())

    # process_dataset(converter=EventAFEConverter(height=260, width=346, sample_event_threshold=40, sample_event_num_min=100000),
    #                 file_extension='aedat4', num_workers=cpu_count())

    process_dataset(converter=EventVoxelGridConverter(height=260, width=346, output_npy_or_frame="enhancement_frame", voxel_bin_num=9),
                    file_extension='aedat4', num_workers=3)
    process_dataset(converter=EventVoxelGridConverter(height=260, width=346, output_npy_or_frame="enhancement_frame", voxel_bin_num=8),
                    file_extension='aedat4', num_workers=3)
    process_dataset(converter=EventVoxelGridConverter(height=260, width=346, output_npy_or_frame="enhancement_frame", voxel_bin_num=16),
                    file_extension='aedat4', num_workers=3)

    process_dataset(
        converter=EventGTEConverter(height=260, width=346, output_npy_or_frame="enhancement_frame", patch_size=(4, 4), group_num=12),
        file_extension='aedat4', num_workers=3)
    process_dataset(
        converter=EventGTEConverter(height=260, width=346, output_npy_or_frame="enhancement_frame", patch_size=(4, 4), group_num=16),
        file_extension='aedat4', num_workers=3)
    process_dataset(
        converter=EventGTEConverter(height=260, width=346, output_npy_or_frame="enhancement_frame", patch_size=(4, 4), group_num=32),
        file_extension='aedat4', num_workers=3)
