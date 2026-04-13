.PHONY: all data train quantize eval clean deploy

all: eval

data:
	@echo " Preprocessing data..."
	python src/data_prep/split_data.py

train: data
	@echo " Training model..."
	python src/train/train_tf_cnn.py

quantize: train
	@echo " Quantization..."
	python src/train/quantization.py

eval: quantize
	@echo " Benchmarking..."
	python src/inference/pi/process-tfl/benchmark_comparison.py
	@echo "\nFinished!"
