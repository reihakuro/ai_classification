.PHONY: all data train quantize eval clean deploy

all: eval

data:data
	@echo " Preprocessing data..."
	python src/data_prep/split_data.py

train: train data
	@echo " Training model..."
	python src/train/train_tf_cnn.py

quantize: quantize train
	@echo " Quantization..."
	python src/train/quantization.py

eval: benchmarking
	@echo " Benchmarking..."
	python src/inference/pi/process-tfl/benchmark_comparison.py
	@echo "\nFinished!"
