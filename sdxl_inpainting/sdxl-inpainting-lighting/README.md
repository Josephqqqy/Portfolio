---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: text
    dtype: string
  - name: masked_image
    dtype: image
  - name: mask
    dtype: image
  splits:
  - name: train
    num_bytes: 6618044944.875
    num_examples: 3131
  download_size: 6608891497
  dataset_size: 6618044944.875
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
