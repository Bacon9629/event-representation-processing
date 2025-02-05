import os
import glob
import re
import sys
from tqdm import tqdm
from EventProcessing.BaseEventImageConverter import BaseEventImageConverter
from EventProcessing.event_frame import EventFrameConverter


def load_file_descriptions(data_root):
    """
    Load train, val, and test file descriptions and return dictionaries.
    """
    description_paths = {
        'train': os.path.join(data_root, 'description', 'train.txt'),
        'val': os.path.join(data_root, 'description', 'val.txt'),
        'test': os.path.join(data_root, 'description', 'test.txt')
    }

    file_dicts = {}
    for split, path in description_paths.items():
        with open(path) as f:
            raw_file_list = f.readlines()
        file_dicts[split] = {
            re.search(r'/(.*?)\.aedat4', item).group(1): int(item.split(' ')[1])
            for item in raw_file_list
        }
    return file_dicts


def process_data(converter: BaseEventImageConverter, file_extension, interval):
    """
    Generalized function to process .npy or .aedat4 files.
    """
    data_root = "/media/2TB_1/dataset/DailyDVS-200" if sys.platform == 'linux' else r"E:/dataset/DailyDvs-200"
    out_root = "/media/2TB_1/Bacon/dataset/DailyDvs-200" if sys.platform == 'linux' else r"E:/dataset/DailyDvs-200"

    file_list = glob.glob(fr"{data_root}/DailyDvs-200/*/*.{file_extension}")
    file_dicts = load_file_descriptions(data_root)

    output_file_list = []

    for item in file_list:
        file_name = os.path.basename(item).split('.')[0]
        found_in_split = False

        for split in ['train', 'val', 'test']:
            if file_name in file_dicts[split]:
                found_in_split = True
                label = file_dicts[split][file_name]
                output_file_list.append(
                    rf"{out_root}/event_count_interval_{interval}/{split}/{label}/{file_name}"
                )

        if not found_in_split:
            raise ValueError(f"{file_name} not found in description train/val/test")

    for in_path, out_path in tqdm(zip(file_list, output_file_list), total=len(file_list)):
        if not os.path.exists(out_path):
            converter.events_to_event_images(input_filepath=in_path, output_file_dir=out_path)


if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.getcwd())

    converter = EventFrameConverter(interval=0.5)

    process_data(converter=converter, file_extension='aedat4', interval=0.5)
    process_data(converter=converter, file_extension='aedat4', interval=0.25)
    process_data(converter=converter, file_extension='aedat4', interval=0.125)

    # process_data(converter=converter, file_extension='npy', interval=0.5)

