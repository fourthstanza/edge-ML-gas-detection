## This is a set of python scripts for training and optimizing Keras/tflite models.

---

I've put these together to simplify my workflow training models and processing them through Edge Impulse for deployment on microcontrollers.

### Before running, you'll need

* **Python version 3.11** (tested, will likely work fine with other releases which tflite supports. check tflite docs!)
* **TensorFlow and TensorFlow lite libraries** with matching versions. TFLite will either be behind or current with tensorflow, so take the current TFLite version and match. e.g. ```pip install tensorflow==2.18 tflite==2.18```
* **Numpy version numpy-1.26.0** comes with Tensorflow.
* **Pyarrow, FastParquet** for parquet support, if you are using parquetToPd.py
* **SKL2ONNX** For conversion from Scikit-Learn models to ONNX. This is required to feed them into the STM deployment pipeline.
* **PySTM32** Wrapper for STM32 command line tool for converting 
* **ONNX 1.19** downgraded for compatibility with ml_dtypes 0.4.1, which is required for tensorflow 2.18. Reinstall the correct verion of ml_dtypes after installing ONNX 1.19.

The DataSplit class is used to handle randomly splitting datasets into training, testing, and validation. Initialize it as an object & use test_train_shuffle to shuffle the test/train split during hyperparameter optimization.

