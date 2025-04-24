
# mmsegmentation as an histo-miner submodule

Here is the presentation of mmsegmentation library, as a submodule of histo-miner. The original code is available following this [link](https://github.com/MengzhangLI/mmsegmentation). This fork contains some variation corresponding to the specific training used for histo-miner pipeline.
 

## Installation from yml file

One can install the required dependencies inside a conda env as follow:

**Step 1.** Open a terminal in your local or remote device, create hovernet_submodule environment using the corresponding yml congif and activate it: 
```shell
conda env create -f mmsegmentation_submodule.yml 
conda activate mmsegmentation_submodule
```

**Step 2.** Install PyTorch ([official instructions](https://pytorch.org/get-started/locally/)): 

- **Recommended:** install PyTorch version 1.13 with CUDA 1.16 
```shell
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
```

- *Not Recommended:** Like the instruction from [mmsegmentation repository]((https://github.com/MengzhangLI/mmsegmentation)), install last PyTorch (in this case probably use another Python version):
On GPU:
```shell
conda install pytorch torchvision -c pytorch
```
On CPU:
```shell
conda install pytorch torchvision cpuonly -c pytorch
```


## Installation from scratch

Alternatively the mmsegmentation_submodule env can be installed as follows:

**Step 1.** Open a terminal in your local or remote device, create a conda environment and activate it. 

```shell
conda create -n mmsegmentation_submodule python==3.8 
conda activate mmsegmentation_submodule
```

**Step 2.** Install PyTorch ([official instructions](https://pytorch.org/get-started/locally/)): 

- **Recommended:** install PyTorch version 1.13 with CUDA 1.16 
```shell
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
```

- *Not Recommended:** Like the instruction from [mmsegmentation repository]((https://github.com/MengzhangLI/mmsegmentation)), install last PyTorch (in this case probably use another Python version):
On GPU:
```shell
conda install pytorch torchvision -c pytorch
```
On CPU:
```shell
conda install pytorch torchvision cpuonly -c pytorch
```

**Step 3.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).
```shell
pip install -U openmim
mim install mmengine
mim install "mmcv==1.7.0"
```

**Step 4.** Install MMSegmentation.
```shell
pip install mmsegmentation==1.1.1
```

For more information on how to install or for a custom installation of the `mmsegmentation submodule`, the reader can refer to the original guidelines [here](https://github.com/MengzhangLI/mmsegmentation) that were alos used a the source of this README. 