# event-representation-processing
Code for various methods to convert event-based camera data into RGB-like representations.

## Current representation

### Event Frame


### Event Count (Event Histogram, Event Count Histogram)
| src: https://dv-processing.inivation.com/rel_1_7/accumulators.html


### Time Surface
| src: https://dv-processing.inivation.com/rel_1_7/accumulators.html


### Speed Invariant Time Surface
| src: https://dv-processing.inivation.com/rel_1_7/accumulators.html


### Temporal Binary Representation (TBR)


### Adaptive Fine-grained Event (AFE)


### Voxel Grid
* Hyperparameter from Source paper: Unsupervised Event-based Learning of Optical Flow, Depth, and Egomotion
* Code from https://github.com/Peterande/GET-Group-Event-Transformer/blob/master/event_based/voxel_grid.py 

### Group Token Embedding (GTE)
| https://github.com/Peterande/GET-Group-Event-Transformer/blob/master/event_based/voxel_grid.py
* 只有他輸出的是npy
* npy shape: `[channel, H // patch_size, W // patch_size]`
* 在原始repo內，他有對event stream進行augmentation，但因為以下原因，本repo沒這麼做
  * 可比性: 使所有representation擁有同一個training dataset條件，降低實驗誤差
  * 效率: 若使用作者提供的augmentation，資料集無法預先處理，這樣就必須在training過程中花時間處理