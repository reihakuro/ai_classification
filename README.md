# IMAGE CLASSIFICATION ON RASPBERRY PI
## Overview
This repository presents a machine learning-based image classification pipeline optimized for edge computing environments like Raspberry Pi. 

The system architecture adopts a decoupled computational approach:   
- Training and optimize phase is executed on a high-performance host machine
- Trained and optimized model is exported and deployed onto Raspberry Pi 

This methodology enables efficient, real-time visual inference with minimal latency without relying on cloud-based processing.

## Features
- Edge-based Inference: real-time image classification executed locally on the Raspberry Pi hardware.
- Host training: Utilization of host computing resources for robust model training, hyperparameter tuning, and weight optimization.
- Model optimization : integration of pruning techniques to reduce model footprint, benchmarking to ensure an optimal balance between accuracy and computational efficiency.

## Repository Structure
- data_prep: create dataset and pre-processing scripts for training step
- train: training and package models script 
- inference: deploy, inference and benchmark
- eval: performance comparison between baseline model and pruned model

## System Specifications
The system is developed and evaluated on the following hardware and software stack:
- **Edge Hardware:** Raspberry Pi 4 Model B - 8GiB RAM with Logitech Webcam C505E
- **Software Environment:** Raspberry Pi OS, Python 3.11, TensorFlow 2.x, OpenCV, and TensorFlow Lite for edge inference.

