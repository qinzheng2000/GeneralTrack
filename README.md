# GeneralTrack

> [**Towards Generalizable Multi-Object Tracking**]()
> 
> Zheng Qin, Le Wang, Sanping Zhou, Panpan Fu, Gang Hua, Wei Tang
> 


## Installation
### 1. Installing on the host machine
```shell
git clone 
cd GeneralTrack
conda create -n generaltrack python=3.8 -y
conda activate generaltrack
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
python setup.py develop
pip install cython
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install cython_bbox
```


## Data preparation

Download [BDD100k](https://bdd-data.berkeley.edu/) for MOT 2020 Labels and MOT 2020 images. Unzip all of them to 
```datasets```. 

Also download [detections](https://vision.in.tum.de/webshare/u/seidensc/GHOST/detections_GHOST.zip) from GHOST and also extract into ```dataset```. 
```
datasets/
    - bdd100k
        - images
            - track
                - train
                - val
                - test
        - labels
            - box_track_20
                - train
                - val
    - detections_GHOST
        - bdd100k
            - train
            - val
            - test
```
Packaging detection results and inference files together.
```shell
cd <GeneralTrack_HOME>
python3 tools/convert_bdd100k_to_coco.py
```

## Tracking

**Evaluation on BDD100K**
* **Validation set**
```shell
cd <GeneralTrack_HOME>
python3 tools/track.py
python3 tools/txt2json_trackeval.py

# Unzip 'data.zip'(https://drive.google.com/file/d/1ZAemZSiRtJNIL68g2mYViBDfVMt4igL1/view?usp=drive_link). Put the json file into 'TrackEval/data/trackers/bdd100k/bdd100k_val/xxtrack/data'
python3 TrackEval/scripts/run_bdd.py --USE_PARALLEL True --NUM_PARALLEL_CORES 64
```

* **Test set**

```shell
cd <GeneralTrack_HOME>
python3 tools/track.py --test
python3 tools/txt2json_web.py
```
Submit to [BDD server](https://eval.ai/web/challenges/challenge-page/1836/overview)



## Citation

```

```

## Acknowledgement

A large part of the code and the detection results are borrowed from [ByteTrack](https://github.com/ifzhang/ByteTrack), [RAFT](https://github.com/princeton-vl/RAFT), [GHOST](https://github.com/dvl-tum/GHOST). Many thanks for their wonderful works.
