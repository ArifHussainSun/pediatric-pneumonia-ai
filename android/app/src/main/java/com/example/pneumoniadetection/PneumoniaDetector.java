package com.example.pneumoniadetection;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

/**
 * TensorFlow Lite wrapper for pneumonia detection inference.
 *
 * Handles model loading, image preprocessing, and prediction logic
 * for pediatric chest X-ray analysis on Android devices.
 */
public class PneumoniaDetector {
    private static final String TAG = "PneumoniaDetector";

    private static final int INPUT_SIZE = 224;
    private static final int NUM_CLASSES = 2;
    private static final String[] CLASS_LABELS = {"NORMAL", "PNEUMONIA"};

    private Interpreter tflite;
    private ImageProcessor imageProcessor;

    public PneumoniaDetector(Context context, String modelPath) throws IOException {
        // Load TensorFlow Lite model
        MappedByteBuffer tfliteModel = loadModelFile(context, modelPath);
        tflite = new Interpreter(tfliteModel);

        // Initialize image processor
        imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeOp(INPUT_SIZE, INPUT_SIZE, ResizeOp.ResizeMethod.BILINEAR))
                .build();

        Log.i(TAG, "PneumoniaDetector initialized successfully");
    }

    /**
     * Predict pneumonia from chest X-ray image.
     *
     * @param bitmap Input chest X-ray image
     * @return PredictionResult containing prediction and confidence
     */
    public PredictionResult predict(Bitmap bitmap) {
        try {
            // Assess image quality and apply preprocessing if needed
            ImageQualityUtils.QualityAssessment quality = ImageQualityUtils.assessImageQuality(bitmap);

            if (!quality.isAcceptable) {
                Log.i(TAG, String.format("Poor image quality detected (%s), applying enhancement", quality.overallQuality));
                bitmap = ImageQualityUtils.applyBasicEnhancement(bitmap);
            }

            // Preprocess image
            TensorImage tensorImage = new TensorImage();
            tensorImage.load(bitmap);
            tensorImage = imageProcessor.process(tensorImage);

            // Run inference
            float[][] output = new float[1][NUM_CLASSES];
            tflite.run(tensorImage.getBuffer(), output);

            // Process results
            float normalProb = output[0][0];
            float pneumoniaProb = output[0][1];

            // Apply softmax if needed
            float maxProb = Math.max(normalProb, pneumoniaProb);
            normalProb = (float) Math.exp(normalProb - maxProb);
            pneumoniaProb = (float) Math.exp(pneumoniaProb - maxProb);

            float sum = normalProb + pneumoniaProb;
            normalProb /= sum;
            pneumoniaProb /= sum;

            // Determine prediction
            String prediction;
            float confidence;

            if (pneumoniaProb > normalProb) {
                prediction = "PNEUMONIA";
                confidence = pneumoniaProb;
            } else {
                prediction = "NORMAL";
                confidence = normalProb;
            }

            return new PredictionResult(prediction, confidence, normalProb, pneumoniaProb);

        } catch (Exception e) {
            Log.e(TAG, "Prediction failed", e);
            return new PredictionResult("ERROR", 0.0f, 0.0f, 0.0f);
        }
    }

    /**
     * Load TensorFlow Lite model from assets.
     */
    private MappedByteBuffer loadModelFile(Context context, String modelPath) throws IOException {
        FileInputStream inputStream = new FileInputStream(context.getAssets().openFd(modelPath).getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = context.getAssets().openFd(modelPath).getStartOffset();
        long declaredLength = context.getAssets().openFd(modelPath).getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /**
     * Get model information.
     */
    public ModelInfo getModelInfo() {
        return new ModelInfo(
            INPUT_SIZE,
            NUM_CLASSES,
            CLASS_LABELS
        );
    }

    /**
     * Clean up resources.
     */
    public void close() {
        if (tflite != null) {
            tflite.close();
            tflite = null;
        }
    }

    /**
     * Result class for prediction output.
     */
    public static class PredictionResult {
        public final String prediction;
        public final float confidence;
        public final float normalProbability;
        public final float pneumoniaProbability;

        public PredictionResult(String prediction, float confidence,
                              float normalProbability, float pneumoniaProbability) {
            this.prediction = prediction;
            this.confidence = confidence;
            this.normalProbability = normalProbability;
            this.pneumoniaProbability = pneumoniaProbability;
        }

        @Override
        public String toString() {
            return String.format("Prediction: %s (%.2f%% confidence)",
                               prediction, confidence * 100);
        }
    }

    /**
     * Model information class.
     */
    public static class ModelInfo {
        public final int inputSize;
        public final int numClasses;
        public final String[] classLabels;

        public ModelInfo(int inputSize, int numClasses, String[] classLabels) {
            this.inputSize = inputSize;
            this.numClasses = numClasses;
            this.classLabels = classLabels;
        }
    }
}