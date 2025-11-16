#!/usr/bin/env python3
import tensorflow as tf # type: ignore

print("TensorFlow version:", tf.__version__)

for device in ['CPU', 'GPU']:
    devices = tf.config.list_physical_devices(device) # type: ignore
    print(f"Available {device} devices:", devices) # type: ignore
