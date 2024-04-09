# !/bin/bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
conda config --set show_channel_urls yes && \
pip install numpy && \
pip install opencv-python && \
pip install torchvision && \
pip install scikit-image && \
pip install easydict && \
pip install h5py && \
pip install einops && \
pip install timm && \
pip install tensorboardX && \
pip install argparse