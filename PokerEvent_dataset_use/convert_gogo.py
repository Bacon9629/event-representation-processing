import os
import glob
import re
import sys
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from EventProcessing.BaseEventImageConverter import BaseEventImageConverter
from EventProcessing import EventFrameConverter
from EventProcessing import EventCountConverter
from EventProcessing import EventTimeSurfaceConverter
from EventProcessing import EventSpeedInvariantTimeSurfaceConverter
from EventProcessing import EventAFEConverter
from EventProcessing import EventVoxelGridConverter
from EventProcessing import EventGTEConverter


def load_file_descriptions(description_root):
    """
    Load train.txt, val.txt, and test.txt file descriptions and return dictionaries.

    description file content:
        /workspace/dataset/PokerEvent/PokerEvent/action_Poker001/dvSave-2023_03_02_18_58_21.aedat4 0
        /workspace/dataset/PokerEvent/PokerEvent/action_Poker001/dvSave-2023_03_02_18_58_36.aedat4 0
        /workspace/dataset/PokerEvent/PokerEvent/action_Poker001/dvSave-2023_03_02_18_58_41.aedat4 0
        /workspace/dataset/PokerEvent/PokerEvent/action_Poker001/dvSave-2023_03_02_18_58_42.aedat4 0
        ...

    return:
    tuple: (train_dict, val_dict, test_dict)
    train_dict: dict, key is file path, value is label
    val_dict: dict, key is file path, value is label
    test_dict: dict, key is file path, value is label
    """
    description_paths = {
        'train': os.path.join(description_root, 'train.txt'),
        'val': os.path.join(description_root, 'val.txt'),
        'test': os.path.join(description_root, 'test.txt'),
    }
    train_dict = {}
    val_dict = {}
    test_dict = {}
    for split, path in description_paths.items():
        with open(path) as f:
            raw_file_list = f.readlines()
        for item in raw_file_list:
            file_path, label = item.strip().replace(os.sep, "/").split(' ')
            if split == 'train':
                train_dict[file_path] = label
            elif split == 'val':
                val_dict[file_path] = label
            elif split == 'test':
                test_dict[file_path] = label
    return train_dict, val_dict, test_dict


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


def process_dataset(converter: BaseEventImageConverter, dataset_root, num_workers=None):
    """
    Generalized function to process .npy or .aedat4 files using parallel processing.
    """
    print(f"Using {converter.__class__.__name__} interval: {converter.interval}")
    train_dict, val_dict, test_dict = load_file_descriptions(os.path.join(dataset_root, "description"))

    output_dir_base = os.path.join(dataset_root, f"{converter.__class__.__name__}_interval_{converter.interval}")
    desc_dir = os.path.join(dataset_root, f"description_{converter.__class__.__name__}_interval_{converter.interval}")
    os.makedirs(desc_dir, exist_ok=True)

    splits = {'train': train_dict, 'val': val_dict, 'test': test_dict}
    new_desc_data = {'train': [], 'val': [], 'test': []}
    process_tasks = []

    for split_name, split_dict in splits.items():
        for file_path, label in tqdm(split_dict.items(), desc=f"Collecting {split_name} paths"):
            input_filepath = os.path.join(dataset_root, file_path)
            assert os.path.exists(input_filepath), f"{input_filepath} does not exist"
            file_name = os.path.basename(input_filepath)
            output_dir = os.path.join(output_dir_base, split_name, label, file_name).replace(os.sep, "/")
            process_tasks.append((converter, input_filepath, output_dir))
            new_desc_data[split_name].append([output_dir, label])

    # 🔁 多進程處理
    num_workers = num_workers if num_workers is not None else cpu_count()
    print(f"Start multiprocessing with {num_workers} workers")
    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap_unordered(__process_one_data, process_tasks), total=len(process_tasks), desc="Processing files"))

    # 💾 儲存新的 description files
    for split_name in ['train', 'val', 'test']:
        desc_path = os.path.join(desc_dir, f"{split_name}.txt")
        with open(desc_path, "w") as f:
            for path, label in new_desc_data[split_name]:
                image_amount = len(os.listdir(path))
                f.write(f"{path} {image_amount} {label}\n")

    print(f"Finished processing {converter.__class__.__name__} dataset")


if __name__ == '__main__':
    process_dataset(
        converter=EventFrameConverter(interval=0.5, width=346, height=260),
        dataset_root="E:/dataset/PokerEvent",
        num_workers=12  # 指定 worker 數量，None 則使用 CPU 全部核心
    )
    # process_dataset(
    #     converter=EventFrameConverter(interval=0.5, width=346, height=260),
    #     dataset_root="/media/2TB_1/dataset/PokerEvent",
    #     num_workers=8  # 指定 worker 數量，None 則使用 CPU 全部核心
    # )
