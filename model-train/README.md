## This is a set of python scripts for training and optimizing Keras/tflite models.

---

I've put these together to simplify my workflow training models and processing them through Edge Impulse for deployment on microcontrollers.

### Before running, you'll need

* **Python version 3.11** (tested, will likely work fine with other releases)
* **TensorFlow and TensorFlow lite libraries** with matching versions. TFLite will either be behind or current with tensorflow, so take the current TFLite version and match. e.g. ```pip install tensorflow==2.18 tflite==2.18```
* **Numpy version numpy-1.26.0** comes with Tensorflow.
* **Pyarrow, FastParquet** for parquet support, if you are using parquetToPd.py

