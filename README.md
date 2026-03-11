# IMAGE CLASSIFICATION ON RASPBERRY PI
## Overview
This repository contains a basic Machine Learning classification system running on a Raspberry Pi. The model is trained on a host PC using a machine learning framework, then the trained model is exported and deployed on the Raspberry Pi for real-time inference.
## Features
- Image classification using a trained ML model
- Model training on host
- Real-time inference running on Raspberry Pi
## Repository Structure
- data: create dataset and pre-processing scripts for training step
- train: training and package models script 
- inference_host: running ML models on PC
- benchmark: performance comparison between baseline model and pruned model
- inference_pi: deploy, inference and benchmark on Raspberry Pi hardware

