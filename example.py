import os
import shutil
from time import time
from EventProcessing import EventFrameConverter
from EventProcessing import EventCountConverter
from EventProcessing import EventTimeSurfaceConverter
from EventProcessing import EventSpeedInvariantTimeSurfaceConverter
from EventProcessing import EventAFEConverter
from EventProcessing import EventGTEConverter
from EventProcessing import EventVoxelGridConverter


def event_frame_example():
    converter = EventFrameConverter(width=320, height=240, interval=0.5)

    in_path = testing_file
    output_path = testing_output_dir

    converter.events_to_event_images(input_filepath=in_path,
                                     output_file_dir=os.path.join(output_path, converter.__class__.__name__))


def event_count_example():
    converter = EventCountConverter(width=320, height=240, interval=0.5)

    in_path = testing_file
    output_path = testing_output_dir

    converter.events_to_event_images(input_filepath=in_path,
                                     output_file_dir=os.path.join(output_path, converter.__class__.__name__))


def event_time_surface_example():
    converter = EventTimeSurfaceConverter(width=320, height=240, interval=0.5)

    in_path = testing_file
    output_path = testing_output_dir

    converter.events_to_event_images(input_filepath=in_path,
                                     output_file_dir=os.path.join(output_path, converter.__class__.__name__))


def event_speed_invariant_time_surface_example():
    converter = EventSpeedInvariantTimeSurfaceConverter(width=320, height=240, interval=0.5)

    in_path = testing_file
    output_path = testing_output_dir

    converter.events_to_event_images(input_filepath=in_path,
                                     output_file_dir=os.path.join(output_path, converter.__class__.__name__))


def event_AFE_example():
    converter = EventAFEConverter(width=320, height=240,
                                  representation="rgb",
                                  sample_event_threshold=40,
                                  sample_event_num_min=100000)

    in_path = testing_file
    output_path = testing_output_dir

    converter.events_to_event_images(input_filepath=in_path,
                                     output_file_dir=os.path.join(output_path, converter.__class__.__name__))


def event_voxel_grid_example():
    converter = EventVoxelGridConverter(width=320, height=240,
                                        voxel_bin_num=9)

    in_path = testing_file
    output_path = testing_output_dir

    converter.events_to_event_images(input_filepath=in_path,
                                     output_file_dir=os.path.join(output_path, converter.__class__.__name__))


def event_GTE_npy_example():
    converter = EventGTEConverter(width=320, height=240,
                                  output_npy_or_frame="npy",
                                  patch_size=(4, 4),
                                  group_num=12)

    in_path = testing_file
    output_path = testing_output_dir

    converter.events_to_event_images(input_filepath=in_path,
                                     output_file_dir=os.path.join(output_path, f"{converter.__class__.__name__}_{converter.output_npy_or_frame}"))


def event_GTE_ori_image_example():
    converter = EventGTEConverter(width=320, height=240,
                                  output_npy_or_frame="ori_frame",
                                  patch_size=(4, 4),
                                  group_num=12)

    in_path = testing_file
    output_path = testing_output_dir

    converter.events_to_event_images(input_filepath=in_path,
                                     output_file_dir=os.path.join(output_path, f"{converter.__class__.__name__}_{converter.output_npy_or_frame}"))


def event_GTE_enhancement_image_example():
    converter = EventGTEConverter(width=320, height=240,
                                  output_npy_or_frame="enhancement_frame",
                                  patch_size=(4, 4),
                                  group_num=12)

    in_path = testing_file
    output_path = testing_output_dir

    converter.events_to_event_images(input_filepath=in_path,
                                     output_file_dir=os.path.join(output_path, f"{converter.__class__.__name__}_{converter.output_npy_or_frame}"))


testing_file = "src/C11P16M1S3_20231120_10_04_07.npy"  # .aedat4 or .npy
testing_output_dir = "src/figs"
# testing_file = r"E:\dataset\DailyDvs-200\test\DailyDvs-200\011\C11P7M1S1_20231117_10_52_44.npz.npy"  # .aedat4 or .npy
# testing_output_dir = r"E:\dataset\DailyDvs-200\test\a"

if __name__ == '__main__':
    if os.path.exists(testing_output_dir):
        print("Remove output dir")
        shutil.rmtree(testing_output_dir)

    start_time = time()

    print("event_frame_example")
    event_frame_example()
    print("event_count_example")
    event_count_example()
    print("event_time_surface_example")
    event_time_surface_example()
    print("event_speed_invariant_time_surface_example")
    event_speed_invariant_time_surface_example()
    print("event_AFE_example")
    event_AFE_example()
    print("event_voxel_grid_example")
    event_voxel_grid_example()
    print("event_GTE_npy_example")
    event_GTE_npy_example()
    print("event_GTE_ori_image_example")
    event_GTE_ori_image_example()
    print("event_GTE_enhancement_image_example")
    event_GTE_enhancement_image_example()

    end_time = time()

    print(f"cost time: %.3f s" % (end_time - start_time))
    print("finish")
