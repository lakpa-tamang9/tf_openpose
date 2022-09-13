# Tensorflow implementation of openpose.
Original work: https://www.github.com/ildoonet/tf-openpose

## Usage
1. Create new environment and activate it
```
conda create -n tfpose python=3.7.6 -y
conda activate tfpose
```
2. Install conda packages:
```
conda install -c anaconda numpy
conda instal swig
```

3. Install requirements
```
pip install -r requirements.txt
```

4. Build from source using swig
```
cd tf_pose/pafprocess/
swig -python -c++ pafprocess.i 
python setup.py build_ext --inplace
```
5. Run inference in webcam using ```run_webcam.py``` and for pre-recorded video using ```run_video.py```
