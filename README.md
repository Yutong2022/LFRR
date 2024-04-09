# LFRR_TCI2024
The dataset and source codes for Disparity-guided Multi-view Interaction Network for Light Field Reflection Removal.
[Paper](https://ieeexplore.ieee.org/abstract/document/10268449) | [Bibtex](https://github.com/Yutong2022/LFRR)

## Dependencies

- Python 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- Pytorch 1.7.0
- einops
- Numpy
- Scipy
- matplotlib
- timm
- VGG19

## Datasets
**Based on our self-collected LF reflection dataset, we use 400 (synthetic) + 50 (real-world) scenes for training and 20 (synthetic) + 20 (real-world) scenes for testing.
Please first download our datasets via [Baidu Drive](https://pan.baidu.com/s/1fkYbIVchBLBd5oGnniqgFA?pwd=vida), and place the datasets to the folder `./LFRR_DATA/`. You can use `./read_lf_h5.py` to load the data.**

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


## Train & test

Before training or testing, please downland checkpoint from [Baidu Drive](https://pan.baidu.com/s/1GXKF9HzT0sKhN91z0gvUHw?pwd=vida) and put them into the folder `./pretrained_model/`.

* If you want to simply start training, you can execute the code:
```shell
bash train.sh 
```
* If you want to simply start testing, you can execute the code:
```shell
bash test.sh
```

## Citation

If you find this work helpful, please consider citing our paper.
<!-- ```latex
@ARTICLE{10268449,
  author={Liu, Yutong and Cheng, Zhen and Xiao, Zeyu and Xiong, Zhiwei},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Light Field Super-Resolution Using Decoupled Selective Matching}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2023.3321085}}
``` -->

## Related Projects

[BasicLFSR](https://github.com/ZhengyuLiang24/BasicLFSR)


## Contact

If you have any problem about the released code, please contact me with email (ustclyt@mail.ustc.edu.cn).
