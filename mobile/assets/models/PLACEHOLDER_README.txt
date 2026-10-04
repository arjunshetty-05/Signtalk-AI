movenet.tflite and bilstm.tflite are NOT included here (binary model
weights don't belong in a code scaffold).

Before building the offline-mode path:
1. Run backend/convert_to_tflite.py --download_movenet
   -> copy the cached backend/models/movenet.tflite here.
2. Train the classifier (backend/train_bilstm.py) and copy the resulting
   bilstm.tflite + labels.json here, replacing the placeholder labels.json.

Until both real .tflite files are present, TFLiteInferenceService.initialize()
will throw when Interpreter.fromAsset() can't find them — that's expected
and intentional (fail loudly rather than silently running on garbage
weights).
