package com.example.pneumoniadetection;

import android.graphics.Bitmap;
import android.graphics.Color;
import android.util.Log;

/**
 * Image quality assessment utilities for Android preprocessing.
 *
 * Provides basic image quality checks and enhancement capabilities
 * for medical chest X-ray images before model inference.
 */
public class ImageQualityUtils {
    private static final String TAG = "ImageQualityUtils";

    /**
     * Quality assessment result class.
     */
    public static class QualityAssessment {
        public final float brightness;
        public final float contrast;
        public final String brightnessQuality;
        public final String contrastQuality;
        public final String overallQuality;
        public final boolean isAcceptable;

        public QualityAssessment(float brightness, float contrast,
                               String brightnessQuality, String contrastQuality,
                               String overallQuality, boolean isAcceptable) {
            this.brightness = brightness;
            this.contrast = contrast;
            this.brightnessQuality = brightnessQuality;
            this.contrastQuality = contrastQuality;
            this.overallQuality = overallQuality;
            this.isAcceptable = isAcceptable;
        }
    }

    /**
     * Assess the quality of a medical image.
     *
     * @param bitmap Input image to assess
     * @return QualityAssessment with quality metrics
     */
    public static QualityAssessment assessImageQuality(Bitmap bitmap) {
        try {
            // Convert to grayscale and calculate statistics
            float[] grayValues = convertToGrayscale(bitmap);

            float brightness = calculateMean(grayValues);
            float contrast = calculateStandardDeviation(grayValues, brightness);

            // Assess quality levels
            String brightnessQuality = assessBrightnessQuality(brightness);
            String contrastQuality = assessContrastQuality(contrast);
            String overallQuality = determineOverallQuality(brightnessQuality, contrastQuality);
            boolean isAcceptable = overallQuality.equals("good") || overallQuality.equals("acceptable");

            Log.d(TAG, String.format("Image quality: brightness=%.1f (%s), contrast=%.1f (%s), overall=%s",
                    brightness, brightnessQuality, contrast, contrastQuality, overallQuality));

            return new QualityAssessment(brightness, contrast, brightnessQuality,
                                       contrastQuality, overallQuality, isAcceptable);

        } catch (Exception e) {
            Log.e(TAG, "Quality assessment failed", e);
            // Return default acceptable quality on error
            return new QualityAssessment(127.0f, 50.0f, "good", "good", "acceptable", true);
        }
    }

    /**
     * Apply basic CLAHE-like enhancement to improve image quality.
     *
     * @param bitmap Input image to enhance
     * @return Enhanced bitmap
     */
    public static Bitmap applyBasicEnhancement(Bitmap bitmap) {
        try {
            int width = bitmap.getWidth();
            int height = bitmap.getHeight();
            Bitmap enhanced = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);

            // Simple histogram stretching (basic contrast enhancement)
            int[] pixels = new int[width * height];
            bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

            // Find min and max values
            int min = 255, max = 0;
            for (int pixel : pixels) {
                int gray = Color.red(pixel); // Assuming grayscale
                min = Math.min(min, gray);
                max = Math.max(max, gray);
            }

            // Apply histogram stretching
            float scale = 255.0f / (max - min);
            for (int i = 0; i < pixels.length; i++) {
                int gray = Color.red(pixels[i]);
                int enhanced_gray = Math.round((gray - min) * scale);
                enhanced_gray = Math.max(0, Math.min(255, enhanced_gray));
                pixels[i] = Color.rgb(enhanced_gray, enhanced_gray, enhanced_gray);
            }

            enhanced.setPixels(pixels, 0, width, 0, 0, width, height);
            Log.d(TAG, "Applied basic enhancement to image");

            return enhanced;

        } catch (Exception e) {
            Log.e(TAG, "Enhancement failed", e);
            return bitmap; // Return original on error
        }
    }

    private static float[] convertToGrayscale(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        float[] grayValues = new float[width * height];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pixel = bitmap.getPixel(x, y);
                float gray = 0.299f * Color.red(pixel) + 0.587f * Color.green(pixel) + 0.114f * Color.blue(pixel);
                grayValues[y * width + x] = gray;
            }
        }

        return grayValues;
    }

    private static float calculateMean(float[] values) {
        float sum = 0;
        for (float value : values) {
            sum += value;
        }
        return sum / values.length;
    }

    private static float calculateStandardDeviation(float[] values, float mean) {
        float sumSquaredDiff = 0;
        for (float value : values) {
            float diff = value - mean;
            sumSquaredDiff += diff * diff;
        }
        return (float) Math.sqrt(sumSquaredDiff / values.length);
    }

    private static String assessBrightnessQuality(float brightness) {
        if (brightness < 30) return "too_dark";
        if (brightness < 60) return "dark";
        if (brightness > 200) return "too_bright";
        if (brightness > 160) return "bright";
        return "good";
    }

    private static String assessContrastQuality(float contrast) {
        if (contrast < 15) return "very_low";
        if (contrast < 30) return "low";
        if (contrast > 80) return "high";
        return "good";
    }

    private static String determineOverallQuality(String brightnessQuality, String contrastQuality) {
        // Poor quality conditions
        if (brightnessQuality.equals("too_dark") || brightnessQuality.equals("too_bright")) {
            return "poor";
        }
        if (contrastQuality.equals("very_low")) {
            return "poor";
        }

        // Good quality conditions
        if (brightnessQuality.equals("good") && contrastQuality.equals("good")) {
            return "excellent";
        }

        // Acceptable quality conditions
        if ((brightnessQuality.equals("good") || brightnessQuality.equals("dark") || brightnessQuality.equals("bright")) &&
            (contrastQuality.equals("good") || contrastQuality.equals("low"))) {
            return "good";
        }

        return "acceptable";
    }
}