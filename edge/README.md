# Edge deployment for SkyGuard-AWS

This directory contains a minimal TensorFlow Lite example for embedded or microcontroller deployment.

Overview
- Train the model with `python train.py`
- Convert the trained Keras model to TFLite with post-training quantization
- Load the model on ESP32 or another embedded microcontroller
- Feed a sliding window of sensor values and compute a local anomaly score

Example conversion

```bash
python - <<'PY'
import tensorflow as tf
model = tf.keras.models.load_model('model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
model_tflite = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(model_tflite)
print('Saved model.tflite')
PY
```

Recommended deployment strategy
- Use a small window of 30-60 samples
- Keep the model compact: 1 LSTM or 1 GRU layer with small hidden size
- Quantize to float16 or int8 for memory savings
- Prefer one model per station, with central server aggregation

Notes
- For ESP32, use TensorFlow Lite Micro with a minimal interpreter.
- For production, combine lightweight local checks (spikes, frozen sensor, comm loss) with server-side deep-learning inference.
