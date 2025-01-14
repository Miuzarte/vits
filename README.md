### fork后根据[Cj佬](https://github.com/CjangCjengh)和[R佬](https://github.com/innnky)指点修改过的地方，解决（忽略掉）了一些报错

1. [data_utils.py](data_utils.py)

```diff
    batches = []
    for i in range(len(self.buckets)):
        bucket = self.buckets[i]
        len_bucket = len(bucket)
+       if len_bucket == 0:
+           continue
        ids_bucket = indices[i]
        num_samples_bucket = self.num_samples_per_bucket[i]
```

2. [train.py](train.py)

```diff
-   net_g = DDP(net_g, device_ids=[rank])
+   net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
-   net_d = DDP(net_d, device_ids=[rank])
+   net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
```

### 其他

1. [preprocess.py](preprocess.py)

```diff
-   parser.add_argument("--text_cleaners", nargs="+", default=["english_cleaners2"])
+   parser.add_argument("--text_cleaners", nargs="+", default=["chinese_cleaners"])
```

2. [train.py](train.py)

```diff
-   old_g=os.path.join(hps.model_dir, "G_{}.pth".format(global_step-2000))
+   old_g=os.path.join(hps.model_dir, "G_{}.pth".format(global_step-100000))
-   old_d=os.path.join(hps.model_dir, "D_{}.pth".format(global_step-2000))
+   old_d=os.path.join(hps.model_dir, "D_{}.pth".format(global_step-100000))
```

3. [train_ms.py](train_ms.py)

```diff
-   old_g=os.path.join(hps.model_dir, "G_{}.pth".format(global_step-2000))
+   old_g=os.path.join(hps.model_dir, "G_{}.pth".format(global_step-100000))
-   old_d=os.path.join(hps.model_dir, "D_{}.pth".format(global_step-2000))
+   old_d=os.path.join(hps.model_dir, "D_{}.pth".format(global_step-100000))
```

4. [utils.py](utils.py)

```diff
-   model_dir = os.path.join("../drive/MyDrive", args.model)
+   model_dir = os.path.join("./logs", args.model)
```

5. [requirements.txt](requirements.txt)

```diff
-   tensorboard==2.3.0
+   tensorboard
-   torch==1.6.0
+   torch
-   torchvision==0.7.0
+   torchvision
```

#### 一些备注

已知librosa需要使用版本0.8.0

#### 运行历史

1. 九天人工智能平台 V100 32G cuda11.0，python3.8，只安装requirement.txt上的环境

# How to use
(Suggestion) Python == 3.7
## Clone this repository
```sh
git clone https://github.com/CjangCjengh/vits.git
```
## Choose cleaners
- Fill "text_cleaners" in config.json
- Edit text/symbols.py
- Remove unnecessary imports from text/cleaners.py
## Install requirements
```sh
pip install -r requirements.txt
```
## Create datasets
### Single speaker
"n_speakers" should be 0 in config.json
```
path/to/XXX.wav|transcript
```
- Example
```
dataset/001.wav|こんにちは。
```
### Mutiple speakers
Speaker id should start from 0 
```
path/to/XXX.wav|speaker id|transcript
```
- Example
```
dataset/001.wav|0|こんにちは。
```
## Preprocess
If you have done this, set "cleaned_text" to true in config.json
```sh
# Single speaker
python preprocess.py --text_index 1 --filelists path/to/filelist_train.txt path/to/filelist_val.txt

# Mutiple speakers
python preprocess.py --text_index 2 --filelists path/to/filelist_train.txt path/to/filelist_val.txt
```
## Build monotonic alignment search
```sh
cd monotonic_align
python setup.py build_ext --inplace
cd ..
```
## Train
```sh
# Single speaker
python train.py -c <config> -m <folder>

# Mutiple speakers
python train_ms.py -c <config> -m <folder>
```
## Inference
### Online
See [inference.ipynb](inference.ipynb)
### Offline
See [MoeGoe](https://github.com/CjangCjengh/MoeGoe)

# Running in Docker

```sh
docker run -itd --gpus all --name "Container name" -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all "Image name"
```

