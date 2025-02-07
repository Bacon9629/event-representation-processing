# event-representation-processing

* Code for various methods to convert event-based camera data into RGB-like representations.
* This repository integrates various event representation methods from multiple research papers, refining and modifying them to provide a simple, intuitive, and user-friendly Python conversion tool.
* During the integration process, the original source code and algorithms are preserved as much as possible to ensure their effectiveness.
* This is the script for converting the DailyDVS200 dataset -> [DailyDVS200_dataset_use](DailyDVS200_dataset_use)
---
* 這個 repo 整合了多篇論文中的 Event Representation 方法，並進行改進與調整，旨在提供一個簡單、直觀且易於使用的 Python 轉換工具。
* 整合過程中，盡量保留來源原始碼與演算法，以確保其有效性
* 這裡順便附上dataset DailyDVS200 轉換用的腳本 -> [DailyDVS200_dataset_use](DailyDVS200_dataset_use)
---

## 🚀 Quick Start Guide

#### 🏗️ 1. Install Dependencies
```sh
conda create -n event_representation python=3.10
conda activate event_representation
pip install -r requirements.txt
```

#### 🔄 2. Convert Event Data

```python
from EventProcessing import EventFrameConverter

# Other representation converter
# from EventProcessing import EventCountConverter
# from EventProcessing import EventTimeSurfaceConverter
# from EventProcessing import EventSpeedInvariantTimeSurfaceConverter
# from EventProcessing import EventAFEConverter
# from EventProcessing import EventGTEConverter
# from EventProcessing import EventVoxelGridConverter

in_path = "./event_stream.aedat4"
output_path = "./output_event_representation_dir"

# width, height: event camera resolution
# interval: Controls how many time event data to be merged, in seconds, Some representations do not require this parameter
converter = EventFrameConverter(width=320, height=240, interval=0.5)
converter.events_to_event_images(input_filepath=in_path, output_file_dir=output_path)
```

## 📚 Current representation

### 📌 Event Frame

* Method reference: Unknown
* Published in: Unknown
* Code reference: https://github.com/QiWang233/DailyDVS-200/blob/main/utils/event_image_convert.py

| Hyperparameter | Default Value | Default Value Source |
|----------------|---------------|----------------------|
| Interval       | 0.5           | Paper: DailyDVS-200  |

![00000002.png](src%2Ffigs%2FEventFrameConverter%2F00000002.png)
![00000003.png](src%2Ffigs%2FEventFrameConverter%2F00000003.png)
![00000004.png](src%2Ffigs%2FEventFrameConverter%2F00000004.png)
![00000005.png](src%2Ffigs%2FEventFrameConverter%2F00000005.png)


### 📌 Event Count (Event Histogram, Event Count Histogram)

* Method reference: Event-based vision meets deep learning on steering prediction for self-driving cars
* Published in: CVPR 2018
* Code reference: https://dv-processing.inivation.com/rel_1_7/accumulators.html

| Hyperparameter | Default Value | Default Value Source |
|----------------|---------------|----------------------|
| Interval       | 0.5           | NO                   |

![00000002.png](src%2Ffigs%2FEventCountConverter%2F00000002.png)
![00000003.png](src%2Ffigs%2FEventCountConverter%2F00000003.png)
![00000004.png](src%2Ffigs%2FEventCountConverter%2F00000004.png)
![00000005.png](src%2Ffigs%2FEventCountConverter%2F00000005.png)

### 📌 Time Surface

* Method reference: Event-based visual flow
* Published in: IEEE Transactions on Neural Networks and Learning Systems 2014
* Code reference: https://dv-processing.inivation.com/rel_1_7/accumulators.html

| Hyperparameter | Default Value | Default Value Source |
|----------------|---------------|----------------------|
| Interval       | 0.5           | NO                   |

> This example event stream appears less distinct in the time surface representation series due to the high density of event data. However, a slight gradient can still be observed in the first image.

![00000002.png](src%2Ffigs%2FEventTimeSurfaceConverter%2F00000002.png)
![00000003.png](src%2Ffigs%2FEventTimeSurfaceConverter%2F00000003.png)
![00000004.png](src%2Ffigs%2FEventTimeSurfaceConverter%2F00000004.png)
![00000005.png](src%2Ffigs%2FEventTimeSurfaceConverter%2F00000005.png)

### 📌 Speed Invariant Time Surface

* Method reference: Speed invariant time surface for learning to detect corner points with event-based cameras
* Published in: CVPR 2019
* Code reference: https://dv-processing.inivation.com/rel_1_7/accumulators.html

| Hyperparameter | Default Value | Default Value Source |
|----------------|---------------|----------------------|
| Interval       | 0.5           | NO                   |

![00000002.png](src%2Ffigs%2FEventSpeedInvariantTimeSurfaceConverter%2F00000002.png)
![00000003.png](src%2Ffigs%2FEventSpeedInvariantTimeSurfaceConverter%2F00000003.png)
![00000004.png](src%2Ffigs%2FEventSpeedInvariantTimeSurfaceConverter%2F00000004.png)
![00000005.png](src%2Ffigs%2FEventSpeedInvariantTimeSurfaceConverter%2F00000005.png)

### 📌 Adaptive Fine-grained Event (AFE)

* Method reference: ExACT: Language-guided Conceptual Reasoning and Uncertainty Estimation for Event-based Action
  Recognition and More
* Published in: CVPR 2024
* Code reference: https://github.com/jiazhou-garland/ExACT.git

> The author mentioned that the hyperparameter of this method must be readjusted according to each dataset

| Hyperparameter         | Default Value | Default Value Source    |
|------------------------|---------------|-------------------------|
| sample_event_threshold | 40            | From the original paper |
| sample_event_num_min   | 100000        | From the original paper |

![00000015.png](src%2Ffigs%2FEventAFEConverter%2F00000015.png)
![00000016.png](src%2Ffigs%2FEventAFEConverter%2F00000016.png)
![00000017.png](src%2Ffigs%2FEventAFEConverter%2F00000017.png)
![00000018.png](src%2Ffigs%2FEventAFEConverter%2F00000018.png)

### 📌 Voxel Grid

* Method reference: Unsupervised event-based learning of optical flow, depth and egomotion.
* Published in: CVPRW 2019
* Code reference:  https://github.com/Peterande/GET-Group-Event-Transformer/blob/master/event_based/voxel_grid.py

| Hyperparameter | Default Value | Default Value Source    |
|----------------|---------------|-------------------------|
| voxel_bin_num  | 9             | From the original paper |

> The image appears to have nothing, but it's actually just too dark. You can refer to the plot below, which shows the pixel value distribution of the VoxelGrid image.  
> Because the image is barely visible, an additional .npy file has been saved for direct usage.  
> npy shape: [voxel_bin_num, height, width]  
> ![VoxelGrid image pixel value distribution.png](src%2FVoxelGrid%20image%20pixel%20value%20distribution.png)  
> Y-axis: Image pixel value  
> X-axis: Image pixel index  

![00000004.png](src%2Ffigs%2FEventVoxelGridConverter%2F00000004.png)
![00000005.png](src%2Ffigs%2FEventVoxelGridConverter%2F00000005.png)
![00000006.png](src%2Ffigs%2FEventVoxelGridConverter%2F00000006.png)
![00000007.png](src%2Ffigs%2FEventVoxelGridConverter%2F00000007.png)

### 📌 Group Token Embedding (GTE)

* Method reference: GET: Group Event Transformer for Event-Based Vision
* Published in: ICCV 2023
* Code reference:
    * https://github.com/Peterande/GET-Group-Event-Transformer/blob/master/event_based/event_token.py
    * https://github.com/QiWang233/DailyDVS-200/blob/main/models/GET/event_based/event_token.py

> In the original repository, the event stream undergoes [random augmentation](https://github.com/Peterande/GET-Group-Event-Transformer/blob/d979d7a243201d4c86cd1765636b167d7701d881/data/build.py#L182).  
> However, this repository does not apply the same augmentation due to the following reasons:
> 
> * **Comparability**: Ensuring that all representations share the same training dataset conditions, reducing experimental errors.
> * **Efficiency**: Using the author's augmentation method prevents dataset preprocessing, requiring augmentation to be applied during training, which increases processing time.

#### 提供三種輸出模式
> Since the number of channels is 4, it can't convert it directly into image. Therefore, provides three output modes for users to choose from

##### 1. npy
* It is saved as an `.npy` file with the shape: `[channel, H // patch_size, W // patch_size]`
* Follow the representation of the original paper.

##### 2. origin_frame
> Not the original paper provided
* The image appears to have nothing, but it's actually just too dark. You can refer to the plot below, which shows the pixel value distribution of the origin_frame image ([src image](src%2Ffigs%2FEventGTEConverter_ori_frame%2F00000000.png)).  
* Y-axis: Image pixel value | X-axis: Image pixel index  
* ![GTE image pixel value distribution before equalize.png](src%2FGTE%20image%20pixel%20value%20distribution%20before%20equalize.png)

![00000001.png](src%2Ffigs%2FEventGTEConverter_ori_frame%2F00000001.png)
![00000002.png](src%2Ffigs%2FEventGTEConverter_ori_frame%2F00000002.png)
![00000003.png](src%2Ffigs%2FEventGTEConverter_ori_frame%2F00000003.png)
![00000004.png](src%2Ffigs%2FEventGTEConverter_ori_frame%2F00000004.png)


##### 3. enhancement_frame
> Not the original paper provided
* Since the original frames are too dim, apply cv2.equalizeHist individually to each frame channel.
* Here is the pixel value distribution ([00000000.png](src%2Ffigs%2FEventGTEConverter_enhancement_frame%2F00000000.png))
* Y-axis: Image pixel value | X-axis: Image pixel index
* ![GTE image pixel value distribution after equalize.png](src%2FGTE%20image%20pixel%20value%20distribution%20after%20equalize.png)

![00000001.png](src%2Ffigs%2FEventGTEConverter_enhancement_frame%2F00000001.png)
![00000002.png](src%2Ffigs%2FEventGTEConverter_enhancement_frame%2F00000002.png)
![00000003.png](src%2Ffigs%2FEventGTEConverter_enhancement_frame%2F00000003.png)
![00000004.png](src%2Ffigs%2FEventGTEConverter_enhancement_frame%2F00000004.png)

| Hyperparameter | Default Value | Default Value Source    |
|----------------|---------------|-------------------------|
| patch_size     | (4, 4)        | From the original paper |
| group_num      | 12            | From the original paper |
