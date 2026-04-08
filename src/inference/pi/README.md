This directory contains necessary scripts and models for 
running inference on Raspberry Pi.

The system supports two pipelines:
- TFLite INT8: optimized for edge deployment.
- PKL model: used for comparison and legacy testing.

Notice: 
- All these script should run on Raspberry Pi for accurate result benchmark and performance, running on host PC may give different result. - Model deployed must be optimize and quantization for ARM structure.
- Make sure using `tflite-runtime`
