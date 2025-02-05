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


testing_file = "E:/dataset/DailyDvs-200/test/DailyDvs-200/011/C11P4M0S3_20231116_10_59_47.npz.npy"
testing_output_dir = "E:/dataset/DailyDvs-200/test/output/C11P4M0S3_20231116_10_59_47"


def event_frame_example():
    converter = EventFrameConverter(interval=0.5)

    in_path = testing_file
    output_path = testing_output_dir

    converter.events_to_event_images(input_filepath=in_path, output_file_dir=output_path)


def event_count_example():
    converter = EventCountConverter(interval=0.5)

    in_path = testing_file
    output_path = testing_output_dir

    converter.events_to_event_images(input_filepath=in_path, output_file_dir=output_path)


def event_time_surface_example():
    converter = EventTimeSurfaceConverter(interval=0.5)

    in_path = testing_file
    output_path = testing_output_dir

    converter.events_to_event_images(input_filepath=in_path, output_file_dir=output_path)


def event_speed_invariant_time_surface_example():
    converter = EventSpeedInvariantTimeSurfaceConverter(interval=0.5)

    in_path = testing_file
    output_path = testing_output_dir

    converter.events_to_event_images(input_filepath=in_path, output_file_dir=output_path)


def event_AFE_example():
    converter = EventAFEConverter(interval=0.5)

    in_path = testing_file
    output_path = testing_output_dir

    converter.events_to_event_images(input_filepath=in_path, output_file_dir=output_path)


def event_voxel_grid_example():
    converter = EventVoxelGridConverter(interval=0.5)

    in_path = testing_file
    output_path = testing_output_dir

    converter.events_to_event_images(input_filepath=in_path, output_file_dir=output_path)


def event_GTE_example():
    converter = EventGTEConverter(interval=0.5)

    in_path = testing_file
    output_path = testing_output_dir

    converter.events_to_event_images(input_filepath=in_path, output_file_dir=output_path)


if __name__ == '__main__':
    if os.path.exists(testing_output_dir):
        print("Remove output dir")
        shutil.rmtree(testing_output_dir)

    start_time = time()

    # event_frame_example()
    # event_count_example()
    # event_time_surface_example()
    # event_speed_invariant_time_surface_example()
    # event_AFE_example()
    event_voxel_grid_example()
    # event_GTE_example()

    end_time = time()

    print(f"cost time: %.3f s" % (end_time - start_time))
    print("finish")
