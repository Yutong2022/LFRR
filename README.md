# LFRR
The dataset and source codes for Disparity-guided Multi-view Interaction Network for Light Field Reflection Removal.

## Datasets
**Based on our self-collected LF reflection dataset, we use 400 (synthetic) + 50 (real-world) scenes for training and 20 (synthetic) + 20 (real-world) scenes for testing.
Please first download our datasets via [Baidu Drive](https://pan.baidu.com/s/1fkYbIVchBLBd5oGnniqgFA?pwd=vida), and place the datasets to the folder `./LFRR_DATA/`.**

* **Our project has the following structure:**
  ```
  ├──./LFRR_DATA/
  │    ├── LFRR_training
  │    │    ├── mixturelf_sys
  │    │    │    ├── mixturelf_syn (1).h5
  │    │    │    ├── mixturelf_syn (2).h5
  │    │    │    ├── ...
  │    │    ├── mixturelf_real
  │    │    │    ├── mixturelf_real (1).h5
  │    │    │    ├── mixturelf_real (2).h5
  │    │    │    ├── ...
  │    ├── LFRR_testing
  │    │    ├── synthetic
  │    │    │    ├── test_syn (1).h5
  │    │    │    ├── test_syn (2).h5
  │    │    │    ├── ...
  │    │    ├── realworld
  │    │    │    ├── test_real (1).h5
  │    │    │    ├── test_real (2).h5
  │    │    │    ├── ...
  ```

<br>
