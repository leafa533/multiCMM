#!/bin/bash -u

datasetdir=./data	# The data source dir
modeldir=./model	# The dir saves the trained models
results=./results	# The dir saves the results, which is predicted by the trained model. The evaluation results are also saved here
ifgpu=true		    # ifgpu is true, then using GPU for training
ifdebug=false		# ifdebug is true, when the test set haven't released. Split a part of training set as the test set
num=100			    # ifdebug is true, split $num samples from training set as the test set
best_model=acc		# we can selected choose best accuracy model or best loss model



featype=bert #xlm
language=en #zh
iffinetune=false

python ./train.py $featype $language $modeldir $results $best_model $ifgpu $iffinetune $datasetdir


#
#python ./train.py $featype $language $modeldir $results $best_model $ifgpu $iffinetune $datasetdir

#for language in "en" "zh"; do
#  echo $language
#  python ./train.py $featype $language $modeldir $results $best_model $ifgpu $iffinetune $datasetdir
#done