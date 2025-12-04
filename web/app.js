// Pneumonia Detection Web Interface
class PneumoniaDetector {
    constructor() {
        this.apiUrl = 'http://localhost:8000';
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.checkModelStatus();
    }

    setupEventListeners() {
        const fileInput = document.getElementById('fileInput');
        const uploadArea = document.getElementById('uploadArea');

        // File input change
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileSelect(e.target.files[0]);
            }
        });

        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#4a9eff';
        });

        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#555';
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#555';

            if (e.dataTransfer.files.length > 0) {
                this.handleFileSelect(e.dataTransfer.files[0]);
            }
        });

        // Click to upload
        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });
    }

    async checkModelStatus() {
        const statusElement = document.getElementById('modelStatus');

        try {
            const response = await fetch(`${this.apiUrl}/health`);

            if (response.ok) {
                const data = await response.json();
                statusElement.textContent = '✅ API Ready';
                statusElement.style.color = '#4caf50';
            } else {
                throw new Error('API not responding');
            }
        } catch (error) {
            statusElement.textContent = '❌ API Offline';
            statusElement.style.color = '#f44336';
            console.error('Model status check failed:', error);
        }
    }

    handleFileSelect(file) {
        // Validate file
        if (!this.validateFile(file)) {
            return;
        }

        // Show image preview
        this.showImagePreview(file);

        // Analyze image
        this.analyzeImage(file);
    }

    validateFile(file) {
        const errorElement = document.getElementById('errorMessage');

        // Check file type
        if (!file.type.startsWith('image/')) {
            this.showError('Please select a valid image file');
            return false;
        }

        // Check file size (max 50MB)
        if (file.size > 50 * 1024 * 1024) {
            this.showError('File size too large. Please select an image smaller than 50MB');
            return false;
        }

        // Hide error if validation passes
        errorElement.style.display = 'none';
        return true;
    }

    showImagePreview(file) {
        const preview = document.getElementById('imagePreview');
        const reader = new FileReader();

        reader.onload = (e) => {
            preview.src = e.target.result;
            preview.style.display = 'block';
        };

        reader.readAsDataURL(file);
    }

    async analyzeImage(file) {
        const loadingElement = document.getElementById('loadingIndicator');
        const resultCard = document.getElementById('resultCard');

        // Show loading
        loadingElement.style.display = 'block';
        resultCard.style.display = 'none';

        try {
            // Prepare form data
            const formData = new FormData();
            formData.append('file', file);
            formData.append('model_type', 'mobilenet');

            // Make API request
            const response = await fetch(`${this.apiUrl}/predict`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`API request failed: ${response.status}`);
            }

            const result = await response.json();

            // Display results
            this.displayResults(result, file.name);

        } catch (error) {
            console.error('Analysis failed:', error);
            this.showError(`Analysis failed: ${error.message}`);
        } finally {
            // Hide loading
            loadingElement.style.display = 'none';
        }
    }

    displayResults(result, fileName) {
        const resultCard = document.getElementById('resultCard');
        const diagnosis = document.getElementById('diagnosis');
        const confidenceValue = document.getElementById('confidenceValue');
        const normalProb = document.getElementById('normalProb');
        const pneumoniaProb = document.getElementById('pneumoniaProb');
        const inferenceTime = document.getElementById('inferenceTime');
        const modelName = document.getElementById('modelName');

        // Set diagnosis
        diagnosis.textContent = result.prediction;
        diagnosis.className = 'diagnosis ' + result.prediction.toLowerCase();

        // Set confidence
        const confidence = (result.confidence * 100).toFixed(1);
        confidenceValue.textContent = `${confidence}%`;

        // Set probabilities
        const normalPct = (result.probabilities.NORMAL * 100).toFixed(1);
        const pneumoniaPct = (result.probabilities.PNEUMONIA * 100).toFixed(1);
        normalProb.textContent = `${normalPct}%`;
        pneumoniaProb.textContent = `${pneumoniaPct}%`;

        // Set metrics
        inferenceTime.textContent = Math.round(result.processing_time_ms || 0);
        modelName.textContent = result.model_name || 'MobileNet';

        // Handle image quality information
        if (result.image_quality) {
            this.displayImageQuality(result.image_quality);
        }

        // Show user feedback if available
        if (result.user_feedback) {
            this.showQualityFeedback(result.user_feedback);
        }

        // Show result card
        resultCard.style.display = 'block';

    }

    displayImageQuality(quality) {
        // Update quality metrics in the web interface
        const qualityElements = {
            'qualityOverall': quality.overall_quality,
            'qualityBrightness': (quality.brightness_score * 100).toFixed(0) + '%',
            'qualityContrast': (quality.contrast_score * 100).toFixed(0) + '%',
            'qualitySharpness': (quality.sharpness_score * 100).toFixed(0) + '%',
            'qualityNoise': (quality.noise_level * 100).toFixed(0) + '%',
            'qualityEnhancement': quality.enhancement_applied ? 'Applied' : 'None'
        };

        // Update elements if they exist
        Object.entries(qualityElements).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value;
            }
        });

        // Show recommendations if available
        if (quality.recommendations && quality.recommendations.length > 0) {
            const recommendationsElement = document.getElementById('qualityRecommendations');
            if (recommendationsElement) {
                recommendationsElement.innerHTML = quality.recommendations
                    .map(rec => `<li>${rec}</li>`)
                    .join('');
            }
        }
    }

    showQualityFeedback(message) {
        // Create and show a quality feedback notification
        const notification = document.createElement('div');
        notification.className = 'quality-notification';
        notification.innerHTML = `
            <div class="notification-content">
                <h4>Image Quality Notice</h4>
                <p>${message}</p>
                <button onclick="this.parentElement.parentElement.remove()">OK</button>
            </div>
        `;

        document.body.appendChild(notification);

        // Auto-remove after 10 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 10000);
    }

    showError(message) {
        const errorElement = document.getElementById('errorMessage');
        errorElement.textContent = message;
        errorElement.style.display = 'block';
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    new PneumoniaDetector();
});

// Add some utility functions
function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function getImageDimensions(file) {
    return new Promise((resolve) => {
        const img = new Image();
        const url = URL.createObjectURL(file);

        img.onload = () => {
            URL.revokeObjectURL(url);
            resolve({ width: img.naturalWidth, height: img.naturalHeight });
        };

        img.src = url;
    });
}