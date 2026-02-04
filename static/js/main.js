// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');
const removeBtn = document.getElementById('removeBtn');
const predictBtn = document.getElementById('predictBtn');
const btnText = document.getElementById('btnText');
const btnLoader = document.getElementById('btnLoader');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');
const retryBtn = document.getElementById('retryBtn');

// State
let selectedFile = null;

// Event Listeners
uploadArea.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);
removeBtn.addEventListener('click', resetUpload);
predictBtn.addEventListener('click', handlePredict);
retryBtn.addEventListener('click', resetAll);

// Drag and Drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

// File Handling
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp', 'image/webp'];
    if (!validTypes.includes(file.type)) {
        showError('Invalid file type. Please upload an image file.');
        return;
    }

    // Validate file size (16MB)
    if (file.size > 16 * 1024 * 1024) {
        showError('File too large. Maximum size is 16MB.');
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        uploadArea.style.display = 'none';
        previewContainer.style.display = 'block';
        predictBtn.style.display = 'block';
        hideError();
        hideResults();
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    imagePreview.src = '';
    uploadArea.style.display = 'block';
    previewContainer.style.display = 'none';
    predictBtn.style.display = 'none';
    hideResults();
    hideError();
}

function resetAll() {
    resetUpload();
    hideError();
    hideResults();
}

// Prediction
async function handlePredict() {
    if (!selectedFile) return;

    // Show loading state
    predictBtn.disabled = true;
    btnText.style.display = 'none';
    btnLoader.style.display = 'block';
    hideError();
    hideResults();

    try {
        // Create form data
        const formData = new FormData();
        formData.append('file', selectedFile);

        // Make prediction request
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || data.message || 'Prediction failed');
        }

        // Display results
        displayResults(data);

    } catch (error) {
        console.error('Prediction error:', error);
        showError(error.message || 'Failed to make prediction. Please try again.');
    } finally {
        // Reset button state
        predictBtn.disabled = false;
        btnText.style.display = 'block';
        btnLoader.style.display = 'none';
    }
}

// Display Results
function displayResults(data) {
    const { prediction, top_predictions, all_predictions } = data;

    // Top prediction
    document.getElementById('topClass').textContent = prediction.class;
    document.getElementById('topConfidence').textContent = `${prediction.confidence.toFixed(2)}%`;
    document.getElementById('topConfidenceFill').style.width = `${prediction.confidence}%`;

    // Top 5 predictions
    const predictionsList = document.getElementById('predictionsList');
    predictionsList.innerHTML = '';

    top_predictions.forEach((pred, index) => {
        const item = document.createElement('div');
        item.className = 'prediction-item';
        item.style.animationDelay = `${index * 0.1}s`;
        item.innerHTML = `
            <span class="prediction-name">${index + 1}. ${pred.class}</span>
            <span class="prediction-confidence">${pred.confidence.toFixed(2)}%</span>
        `;
        predictionsList.appendChild(item);
    });

    // All predictions
    const allPredictionsGrid = document.getElementById('allPredictions');
    allPredictionsGrid.innerHTML = '';

    // Sort by confidence
    const sortedPredictions = Object.entries(all_predictions)
        .sort((a, b) => b[1] - a[1]);

    sortedPredictions.forEach(([className, confidence], index) => {
        const chip = document.createElement('div');
        chip.className = 'prediction-chip';
        chip.style.animationDelay = `${index * 0.05}s`;
        chip.innerHTML = `
            <div class="chip-name">${className}</div>
            <div class="chip-confidence">${confidence.toFixed(1)}%</div>
        `;
        allPredictionsGrid.appendChild(chip);
    });

    // Show results section
    resultsSection.style.display = 'block';

    // Scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);
}

// Error Handling
function showError(message) {
    errorMessage.textContent = message;
    errorSection.style.display = 'block';
    hideResults();

    // Scroll to error
    setTimeout(() => {
        errorSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);
}

function hideError() {
    errorSection.style.display = 'none';
}

function hideResults() {
    resultsSection.style.display = 'none';
}

// Check server health on load
window.addEventListener('load', async () => {
    try {
        const response = await fetch('/health');
        const data = await response.json();

        if (!data.checkpoint_exists) {
            console.warn('Model checkpoint not found. Train the model first.');
        }
    } catch (error) {
        console.error('Health check failed:', error);
    }
});
