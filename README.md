# Code Search

This is source code for "How to make effective semantic code search efficient"

## Dependency

```sh
pip install faiss
pip install torch
pip install more_itertools
```
## Usage 

   ### Data Preparation 
  We provide a small dummy dataset for quick deployment in path './data/github'.  
  To train and test our model:

  1) Download and unzip real dataset from Google Drive.
  Note:
  2) Replace each file in the path with the corresponding real file. 

## Train Models


### Train

```sh
python train_test.py
```
The models will be generated in path'\output\JointEmbeder\github\models'.

### Get best model
Run the following code to get the best batch of models.
```sh
python test_best.py
```



## Get Hash Vector
Convert the code vector and description vector into hash vectors using the trained model.

```sh
python getvec_hash.py
```


## Evaluation
Use all descriptions as queries to evaluate various metrics for searching on hash vectors.
```sh
python auto_time_all.py
```
