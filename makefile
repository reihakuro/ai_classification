.PHONY: all data train quantize eval clean deploy

VERSION := $(shell date +%Y%m%d_%H%M%S)

all: eval

data:
	@echo " Preprocessing data..."
	python src/data_prep/split_data.py

train: data
	@echo " Training model $(VERSION)..."
	python src/train/train_tf_cnn.py --version $(VERSION)

quantize: train
	@echo " Quantization..."
	python src/train/quantization.py --version $(VERSION)

eval: quantize
	@echo " Benchmarking..."
	python src/eval/benchmark_keras.py --version $(VERSION)
	@echo "\nFinished!"
