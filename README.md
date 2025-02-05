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

### Group Token Embedding (GTE)
| https://github.com/Peterande/GET-Group-Event-Transformer/blob/master/event_based/voxel_grid.py
* 只有他輸出的是npy
* npy shape: `[channel, H // patch_size, W // patch_size]`