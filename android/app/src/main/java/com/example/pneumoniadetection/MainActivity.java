package com.example.pneumoniadetection;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.FileNotFoundException;
import java.io.InputStream;

/**
 * Main activity for pneumonia detection app.
 *
 * Handles image upload from gallery/files and runs offline TensorFlow Lite
 * inference
 * for pediatric chest X-ray pneumonia detection.
 */
public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    private static final int PERMISSION_REQUEST_CODE = 1;

    private PneumoniaDetector detector;
    private ImageView imageView;
    private TextView resultText;
    private Button selectImageButton;
    private Button analyzeButton;

    private Bitmap currentImage;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Initialize UI components
        imageView = findViewById(R.id.imageView);
        resultText = findViewById(R.id.resultText);
        selectImageButton = findViewById(R.id.selectImageButton);
        analyzeButton = findViewById(R.id.analyzeButton);

        // Initialize TensorFlow Lite detector
        initializeDetector();

        // Setup button listeners
        selectImageButton.setOnClickListener(v -> selectImage());
        analyzeButton.setOnClickListener(v -> analyzeImage());

        // Check permissions
        checkPermissions();
    }

    private void initializeDetector() {
        try {
            // Use tablet model by default, fallback to phone model
            String modelPath = "mobilenet_android_tablet.tflite";
            detector = new PneumoniaDetector(this, modelPath);

            Log.i(TAG, "TensorFlow Lite detector initialized successfully");

        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize detector", e);
            Toast.makeText(this, "Model loading failed", Toast.LENGTH_LONG).show();
        }
    }

    private void checkPermissions() {
        if (ContextCompat.checkSelfPermission(this,
                Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[] { Manifest.permission.READ_EXTERNAL_STORAGE },
                    PERMISSION_REQUEST_CODE);
        }
    }

    private final ActivityResultLauncher<Intent> imagePickerLauncher = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(), result -> {
                if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                    Uri imageUri = result.getData().getData();
                    loadImageFromUri(imageUri);
                }
            });

    private void selectImage() {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("image/*");
        intent.putExtra(Intent.EXTRA_TITLE, "Select Chest X-Ray Image");
        imagePickerLauncher.launch(Intent.createChooser(intent, "Select X-Ray Image"));
    }

    private void loadImageFromUri(Uri uri) {
        try {
            InputStream inputStream = getContentResolver().openInputStream(uri);
            currentImage = BitmapFactory.decodeStream(inputStream);

            if (currentImage != null) {
                imageView.setImageBitmap(currentImage);
                analyzeButton.setEnabled(true);
                resultText.setText("Image loaded. Tap 'Analyze' to detect pneumonia.");

                Log.i(TAG, String.format("Image loaded: %dx%d",
                        currentImage.getWidth(), currentImage.getHeight()));
            } else {
                Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show();
            }

        } catch (FileNotFoundException e) {
            Log.e(TAG, "Failed to load image from URI", e);
            Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show();
        }
    }

    private void analyzeImage() {
        if (currentImage == null || detector == null) {
            Toast.makeText(this, "Please select an image first", Toast.LENGTH_SHORT).show();
            return;
        }

        // Show processing state
        analyzeButton.setEnabled(false);
        resultText.setText("Analyzing chest X-ray...");

        // Run inference in background thread
        new Thread(() -> {
            try {
                long startTime = System.currentTimeMillis();

                PneumoniaDetector.PredictionResult result = detector.predict(currentImage);

                long inferenceTime = System.currentTimeMillis() - startTime;

                // Update UI on main thread
                runOnUiThread(() -> {
                    displayResults(result, inferenceTime);
                    analyzeButton.setEnabled(true);
                });

            } catch (Exception e) {
                Log.e(TAG, "Inference failed", e);
                runOnUiThread(() -> {
                    resultText.setText("Analysis failed. Please try again.");
                    analyzeButton.setEnabled(true);
                });
            }
        }).start();
    }

    private void displayResults(PneumoniaDetector.PredictionResult result, long inferenceTime) {
        String resultString = String.format(
                "DIAGNOSIS: %s\n" +
                        "Confidence: %.1f%%\n\n" +
                        "Probabilities:\n" +
                        "• Normal: %.1f%%\n" +
                        "• Pneumonia: %.1f%%\n\n" +
                        "Inference time: %d ms",
                result.prediction,
                result.confidence * 100,
                result.normalProbability * 100,
                result.pneumoniaProbability * 100,
                inferenceTime);

        resultText.setText(resultString);

        Log.i(TAG, String.format("Prediction: %s (%.2f confidence) in %d ms",
                result.prediction, result.confidence, inferenceTime));
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (detector != null) {
            detector.close();
        }
    }
}