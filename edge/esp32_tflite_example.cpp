#include <Arduino.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

// Minimal TFLite inference example for an ESP32 or embedded board.
// This is intentionally simple and should be adapted to the exact model structure.

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
tflite::MicroMutableOpResolver<8> resolver;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
constexpr int kWindowSize = 60;
constexpr int kNumFeatures = 3;
constexpr int kInputSize = kWindowSize * kNumFeatures;

float inputBuffer[kInputSize] = {0.0f};

void setup() {
  Serial.begin(115200);
  model = tflite::GetModel(modelData);  // replace with generated model binary
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  // Use optimized ops depending on the model; this is a minimal template.
  resolver.AddFullyConnected();
  resolver.AddTanh();
  resolver.AddRelu();

  static tflite::MicroInterpreter staticInterpreter(model, resolver, tensorArena, kTensorArenaSize);
  interpreter = &staticInterpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Allocation failed");
    while (1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
  Serial.println("Model ready.");
}

void loop() {
  // Fill the inputBuffer with a sliding window of [temperature, pressure, humidity] values.
  for (int i = 0; i < kInputSize; ++i) {
    inputBuffer[i] = 0.0f;
  }

  for (int i = 0; i < kInputSize; ++i) {
    input->data.f[i] = inputBuffer[i];
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Inference failed.");
    delay(1000);
    return;
  }

  float anomalyScore = 0.0f;
  for (int i = 0; i < output->bytes / sizeof(float); ++i) {
    anomalyScore += output->data.f[i];
  }

  Serial.print("Anomaly score: ");
  Serial.println(anomalyScore);
  delay(5000);
}
