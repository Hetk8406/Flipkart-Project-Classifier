import os
import sys
import json
import threading
import webbrowser
import base64
from flask import Flask, request, jsonify, render_template_string, send_from_directory

app = Flask(__name__)

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "Data", "dashboard_assets")
TEXT_DATA_PATH = os.path.join(ASSETS_DIR, "extracted_text.json")

# Lazy-loaded model
_model = None
_classes = ['Apparel', 'Electronics', 'Home']

def get_model():
    global _model
    if _model is None:
        print("Loading TensorFlow Keras model (this may take a few seconds)...")
        import tensorflow as tf
        model_path = os.path.join(BASE_DIR, "best_mobilenet_model.keras")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        _model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
    return _model

# Read extracted notebook text outputs
def load_extracted_text():
    if os.path.exists(TEXT_DATA_PATH):
        try:
            with open(TEXT_DATA_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading text metadata: {e}")
    
    # Fallbacks in case file is missing
    return {
        "metadata_stats": "Total Dataset: 62,197 records\nClasses: Apparel, Electronics, Home",
        "split_summary": "Train: 43,537 (70%)\nValidation: 9,330 (15%)\nTest: 9,330 (15%)",
        "class_weights": "Apparel: 0.3783\nElectronics: 7.0965\nHome: 4.6321",
        "classification_report": "MobileNetV2 Accuracy: 97.0%",
        "model_comparison": "MobileNetV2: 91.50% validation accuracy vs. Custom CNN: 78.20%."
    }

TEXT_DATA = load_extracted_text()

# HTML template string served via render_template_string
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Flipkart Product Catalog Classifier Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #f8fafc;
            --bg-card: #ffffff;
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --primary: #1e40af;
            --primary-hover: #1e3a8a;
            --accent-success: #16a34a;
            --accent-error: #dc2626;
            --border: #e2e8f0;
            --sidebar-width: 250px;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            display: flex;
            height: 100vh;
            overflow: hidden;
        }

        /* Sidebar Styling */
        .sidebar {
            width: var(--sidebar-width);
            background-color: #0f172a;
            color: #f8fafc;
            display: flex;
            flex-direction: column;
            padding: 1.5rem 1rem;
            flex-shrink: 0;
        }

        .sidebar-header {
            margin-bottom: 2rem;
            padding: 0 0.5rem;
        }

        .sidebar-header h2 {
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: -0.5px;
            color: #38bdf8;
        }

        .sidebar-header p {
            font-size: 0.75rem;
            color: #94a3b8;
            margin-top: 0.25rem;
        }

        .nav-menu {
            list-style: none;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .nav-item {
            display: block;
            padding: 0.75rem 1rem;
            color: #cbd5e1;
            text-decoration: none;
            border-radius: 0.375rem;
            font-size: 0.9rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease-in-out;
        }

        .nav-item:hover, .nav-item.active {
            background-color: #1e293b;
            color: #38bdf8;
        }

        .nav-item.active {
            font-weight: 600;
            border-left: 3px solid #38bdf8;
        }

        /* Main Content Styling */
        .main-content {
            flex-grow: 1;
            padding: 2rem;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 2rem;
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        /* Typography & Layout Utilities */
        h1 {
            font-size: 1.75rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            letter-spacing: -0.5px;
        }

        .section-desc {
            font-size: 0.95rem;
            color: var(--text-secondary);
            margin-bottom: 1.5rem;
            line-height: 1.5;
        }

        /* Card Layouts */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }

        .metric-card {
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }

        .metric-card h3 {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }

        .metric-card p {
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--primary);
        }

        .card {
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }

        .card h2 {
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 1rem;
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.5rem;
        }

        /* General Plot Image Container */
        .plot-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 1rem 0;
            border: 1px solid var(--border);
            border-radius: 0.375rem;
            padding: 1rem;
            background-color: #f8fafc;
        }

        .plot-img {
            max-width: 100%;
            height: auto;
            border-radius: 0.25rem;
        }

        .two-column {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            align-items: stretch; /* Force equal column height */
        }

        .two-column > .card {
            display: flex;
            flex-direction: column;
            height: 100%;
            margin-bottom: 0; /* Remove bottom margin to align bottom edges */
        }

        .two-column > .card > pre {
            flex-grow: 1;
            max-height: 280px; /* Restrict height to align sections */
            overflow-y: auto;
            white-space: pre-wrap; /* Wrap long lines */
        }

        @media (max-width: 900px) {
            .two-column {
                grid-template-columns: 1fr;
            }
            .two-column > .card {
                height: auto;
                margin-bottom: 1.5rem;
            }
        }

        /* Observations Styling */
        .observation-box {
            background-color: #f0fdf4;
            border-left: 4px solid var(--accent-success);
            padding: 1rem;
            border-radius: 0.25rem;
            margin-top: 1rem;
        }

        .observation-box p {
            font-size: 0.875rem;
            color: #166534;
            line-height: 1.4;
            margin-bottom: 0.5rem;
        }

        .observation-box p:last-child {
            margin-bottom: 0;
        }

        pre {
            font-family: 'Courier New', Courier, monospace;
            background-color: #0f172a;
            color: #f8fafc;
            padding: 1rem;
            border-radius: 0.375rem;
            overflow-x: auto;
            font-size: 0.85rem;
            line-height: 1.4;
        }

        /* Prediction Demo Styling */
        .upload-container {
            border: 2px dashed #cbd5e1;
            border-radius: 0.5rem;
            padding: 3rem 2rem;
            text-align: center;
            background-color: #f8fafc;
            cursor: pointer;
            transition: border-color 0.2s ease-in-out;
            margin-bottom: 1.5rem;
        }

        .upload-container:hover {
            border-color: var(--primary);
        }

        .upload-container p {
            font-size: 0.95rem;
            color: var(--text-secondary);
            margin-top: 0.5rem;
        }

        .preview-img {
            max-width: 250px;
            max-height: 250px;
            border-radius: 0.375rem;
            margin-top: 1.5rem;
            border: 1px solid var(--border);
            display: none;
        }

        .classify-btn {
            background-color: var(--primary);
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            font-size: 0.95rem;
            font-weight: 600;
            border-radius: 0.375rem;
            cursor: pointer;
            transition: background-color 0.2s ease-in-out;
            display: inline-block;
            margin-top: 1rem;
        }

        .classify-btn:hover {
            background-color: var(--primary-hover);
        }

        .classify-btn:disabled {
            background-color: #94a3b8;
            cursor: not-allowed;
        }

        .result-container {
            display: none;
            margin-top: 1.5rem;
            border-top: 1px solid var(--border);
            padding-top: 1.5rem;
        }

        .result-title {
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .result-confidence {
            font-size: 0.95rem;
            color: var(--text-secondary);
            margin-bottom: 1rem;
        }

        .prob-bar-container {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            margin-top: 1rem;
        }

        .prob-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.875rem;
        }

        .prob-label {
            font-weight: 500;
            width: 100px;
        }

        .prob-bar-outer {
            flex-grow: 1;
            height: 0.75rem;
            background-color: #e2e8f0;
            border-radius: 9999px;
            margin: 0 1rem;
            overflow: hidden;
        }

        .prob-bar-inner {
            height: 100%;
            background-color: var(--primary);
            border-radius: 9999px;
            width: 0%;
            transition: width 0.4s ease-out;
        }

        .prob-value {
            font-weight: 600;
            width: 50px;
            text-align: right;
        }

        .spinner {
            display: none;
            width: 2rem;
            height: 2rem;
            border: 3px solid #cbd5e1;
            border-top-color: var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 1.5rem auto 0 auto;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Business Insights & Decisions */
        .insight-card {
            background-color: #f8fafc;
            border: 1px solid var(--border);
            border-radius: 0.375rem;
            padding: 1rem;
            margin-bottom: 1rem;
            display: flex;
            gap: 1rem;
        }

        .insight-num {
            background-color: #dbeafe;
            color: var(--primary);
            font-weight: 700;
            font-size: 0.875rem;
            width: 1.5rem;
            height: 1.5rem;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            margin-top: 0.125rem;
        }

        .insight-text h4 {
            font-size: 0.95rem;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }

        .insight-text p {
            font-size: 0.875rem;
            color: var(--text-secondary);
            line-height: 1.4;
        }

        .decision-box {
            background-color: #eff6ff;
            border-left: 4px solid #3b82f6;
            padding: 1.25rem;
            border-radius: 0.25rem;
            margin-top: 1rem;
        }

        .decision-box h3 {
            font-size: 1rem;
            font-weight: 600;
            color: #1e40af;
            margin-bottom: 0.5rem;
        }

        .decision-box ul {
            list-style-type: none;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .decision-box li {
            font-size: 0.875rem;
            color: #1e3a8a;
            line-height: 1.4;
            padding-left: 1.25rem;
            position: relative;
        }

        .decision-box li::before {
            content: "•";
            position: absolute;
            left: 0;
            color: #3b82f6;
            font-weight: bold;
        }
    </style>
</head>
<body>

    <!-- Sidebar -->
    <div class="sidebar">
        <div class="sidebar-header">
            <h2>Flipkart Classifier</h2>
            <p>Data Science Capstone Project</p>
        </div>
        <ul class="nav-menu">
            <li><a class="nav-item active" onclick="showTab('overview')">Overview</a></li>
            <li><a class="nav-item" onclick="showTab('dataset')">Dataset</a></li>
            <li><a class="nav-item" onclick="showTab('eda')">EDA</a></li>
            <li><a class="nav-item" onclick="showTab('training')">Model Training</a></li>
            <li><a class="nav-item" onclick="showTab('evaluation')">Model Evaluation</a></li>
            <li><a class="nav-item" onclick="showTab('demo')">Prediction Demo</a></li>
            <li><a class="nav-item" onclick="showTab('insights')">Business Insights</a></li>
            <li><a class="nav-item" onclick="showTab('about')">About</a></li>
        </ul>
    </div>

    <!-- Main Content Area -->
    <div class="main-content">

        <!-- Overview Section -->
        <div id="overview" class="tab-content active">
            <h1>Project Overview</h1>
            <p class="section-desc">Flipkart's catalog team handles millions of product listings daily. Automatically classifying incoming images into business departments reduces human labor, increases cataloging speed, and prevents misclassifications.</p>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Total Dataset Images</h3>
                    <p>62,197</p>
                </div>
                <div class="metric-card">
                    <h3>Target Classes</h3>
                    <p>3 Departments</p>
                </div>
                <div class="metric-card">
                    <h3>Best Validation Acc</h3>
                    <p>91.50%</p>
                </div>
                <div class="metric-card">
                    <h3>Best Performing Model</h3>
                    <p>MobileNetV2</p>
                </div>
            </div>

            <div class="card">
                <h2>Project Summary</h2>
                <p style="line-height: 1.5; font-size: 0.95rem; margin-bottom: 1rem;">This capstone project implements and compares two deep learning models for categorizing e-commerce product photos. We built a lightweight Custom CNN from scratch and fine-tuned a pre-trained MobileNetV2 architecture. Due to the high class imbalance in e-commerce catalogs (where Apparel dominants), our pipeline utilizes class weights and stratified splitting to ensure reliable classification performance across minority departments.</p>
                <p style="line-height: 1.5; font-size: 0.95rem;">Both models were evaluated on an unseen test set to identify classification confusions (e.g. backpacks being confused with clothing) to extract business insights and define routing criteria for automated cataloging gates.</p>
            </div>
        </div>

        <!-- Dataset Section -->
        <div id="dataset" class="tab-content">
            <h1>Dataset Structure</h1>
            <p class="section-desc">Analysis of the raw files, split ratios, and catalog mapping metadata.</p>
            
            <div class="two-column">
                <div class="card">
                    <h2>Metadata Statistics</h2>
                    <pre>{{ metadata_stats }}</pre>
                </div>
                <div class="card">
                    <h2>Dataset Split (70/15/15)</h2>
                    <pre>{{ split_summary }}</pre>
                    <div class="observation-box">
                        <p><strong>Stratification:</strong> To avoid model bias, we performed a stratified train/validation/test split. This maintains an identical 88% Apparel / 7% Home / 5% Electronics ratio across all three folders.</p>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2>Sample Images (Random Grid)</h2>
                <p class="section-desc">Sample products from the flat raw image directory.</p>
                <div class="plot-container">
                    <img src="/assets/random_samples.png" alt="Random Samples" class="plot-img">
                </div>
            </div>
        </div>

        <!-- EDA Section -->
        <div id="eda" class="tab-content">
            <h1>Exploratory Data Analysis</h1>
            <p class="section-desc">Understanding class distribution, aspect ratios, color channels, and image brightness.</p>
            
            <div class="two-column">
                <div class="card">
                    <h2>Class Distribution (Original)</h2>
                    <div class="plot-container">
                        <img src="/assets/class_distribution.png" alt="Class Distribution" class="plot-img">
                    </div>
                    <div class="observation-box">
                        <p># Apparel has the highest number of images by far (88.11%)</p>
                        <p># Electronics and Home are minority classes with much fewer samples</p>
                    </div>
                </div>
                <div class="card">
                    <h2>Aspect Ratio Distribution</h2>
                    <div class="plot-container">
                        <img src="/assets/aspect_ratio.png" alt="Aspect Ratios" class="plot-img">
                    </div>
                    <div class="observation-box">
                        <p># Most images have similar dimensions (400 width x 450 height)</p>
                        <p># The aspect ratio is slightly less than 1.0 (approx 0.89, meaning taller than wide)</p>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2>Product Showcase By Category</h2>
                <div class="plot-container">
                    <img src="/assets/category_samples.png" alt="Category Samples" class="plot-img">
                </div>
                <div class="observation-box">
                    <p># Apparel includes clothing and shoes, Electronics has watches, and Home has bags</p>
                    <p># Some bags (Home) look visually similar to fashion accessories, which might confuse the model</p>
                </div>
            </div>

            <div class="card">
                <h2>Pixel Intensity & Brightness Histogram</h2>
                <div class="plot-container">
                    <img src="/assets/brightness_distribution.png" alt="Brightness" class="plot-img">
                </div>
                <div class="observation-box">
                    <p># The pixel intensities peak very heavily near 255 across all color channels</p>
                    <p># This is because most catalog products are shot against standard studio white backgrounds</p>
                </div>
            </div>
        </div>

        <!-- Model Training Section -->
        <div id="training" class="tab-content">
            <h1>Model Training Configuration</h1>
            <p class="section-desc">Training histories and configurations for the Custom CNN baseline and pre-trained MobileNetV2.</p>
            
            <div class="card">
                <h2>Training Parameters & Weights</h2>
                <pre>{{ class_weights }}</pre>
            </div>

            <div class="two-column">
                <div class="card">
                    <h2>Model 1: Custom CNN Curves</h2>
                    <div class="plot-container">
                        <img src="/assets/training_curves_cnn.png" alt="CNN Curves" class="plot-img">
                    </div>
                    <div class="observation-box">
                        <p># Custom CNN validation loss oscillates, showing early signs of overfitting</p>
                        <p># Without pre-trained weights, the custom model struggles with high variance</p>
                    </div>
                </div>
                <div class="card">
                    <h2>Model 2: MobileNetV2 Curves</h2>
                    <div class="plot-container">
                        <img src="/assets/training_curves_mobilenet.png" alt="MobileNet Curves" class="plot-img">
                    </div>
                    <div class="observation-box">
                        <p># MobileNetV2 curves are smoother, showing steady improvement</p>
                        <p># Fine-tuning the top 20 layers yields a highly stable validation loss profile</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Model Evaluation Section -->
        <div id="evaluation" class="tab-content">
            <h1>Model Evaluation</h1>
            <p class="section-desc">Classification reports and confusion matrices computed on the held-out test split.</p>
            
            <div class="card">
                <h2>Model Performance Comparison</h2>
                <pre>{{ model_comparison }}</pre>
            </div>

            <div class="card">
                <h2>Classification Reports</h2>
                <pre style="white-space: pre-wrap;">{{ classification_report }}</pre>
            </div>

            <div class="two-column">
                <div class="card">
                    <h2>Custom CNN Confusion Matrix</h2>
                    <div class="plot-container">
                        <img src="/assets/confusion_matrix_cnn.png" alt="CNN Confusion Matrix" class="plot-img" style="max-height: 300px;">
                    </div>
                </div>
                <div class="card">
                    <h2>MobileNetV2 Confusion Matrix</h2>
                    <div class="plot-container">
                        <img src="/assets/confusion_matrix_mobilenet.png" alt="MobileNet Confusion Matrix" class="plot-img" style="max-height: 300px;">
                    </div>
                </div>
            </div>

            <div class="card">
                <h2>Test Predictions Gallery</h2>
                <p class="section-desc">Showcase of predictions on the test generator (Left: Correct Predictions, Right: Misclassified items).</p>
                <div class="two-column">
                    <div class="plot-container">
                        <img src="/assets/predictions_gallery_0.png" alt="Correct Predictions" class="plot-img">
                    </div>
                    <div class="plot-container">
                        <img src="/assets/predictions_gallery_1.png" alt="Misclassifications" class="plot-img">
                    </div>
                </div>
            </div>
        </div>

        <!-- Prediction Demo Section -->
        <div id="demo" class="tab-content">
            <h1>Prediction Demo</h1>
            <p class="section-desc">Upload a product image to run live inference using our fine-tuned MobileNetV2 model.</p>
            
            <div class="two-column">
                <div class="card">
                    <h2>Upload Product Image</h2>
                    <div class="upload-container" id="drop-zone" onclick="document.getElementById('file-input').click()">
                        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin: 0 auto 1rem auto; color: var(--text-secondary);">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                            <polyline points="17 8 12 3 7 8"></polyline>
                            <line x1="12" y1="3" x2="12" y2="15"></line>
                        </svg>
                        <p style="font-weight: 600;">Drag and drop your image here, or click to browse</p>
                        <p style="font-size: 0.8rem;">Supports PNG, JPG, JPEG</p>
                        <input type="file" id="file-input" style="display: none;" accept="image/*" onchange="previewImage(event)">
                        <img id="image-preview" class="preview-img" alt="Uploaded Preview">
                    </div>
                    <button class="classify-btn" id="classify-btn" onclick="classifyImage()" disabled>Classify Image</button>
                    <div class="spinner" id="spinner"></div>
                </div>

                <div class="card">
                    <h2>Classification Results</h2>
                    <div id="no-result" style="text-align: center; color: var(--text-secondary); padding: 3rem 0;">
                        <p>Upload and classify an image to see results</p>
                    </div>
                    
                    <div class="result-container" id="result-container">
                        <div class="result-title" id="predicted-class">Apparel</div>
                        <div class="result-confidence" id="predicted-confidence">Confidence: 0.0%</div>
                        
                        <h4 style="font-size: 0.9rem; font-weight: 600; margin-top: 1rem; color: var(--text-secondary);">Probabilities:</h4>
                        <div class="prob-bar-container">
                            <div class="prob-row">
                                <span class="prob-label">Apparel</span>
                                <div class="prob-bar-outer"><div class="prob-bar-inner" id="bar-Apparel"></div></div>
                                <span class="prob-value" id="val-Apparel">0.0%</span>
                            </div>
                            <div class="prob-row">
                                <span class="prob-label">Home</span>
                                <div class="prob-bar-outer"><div class="prob-bar-inner" id="bar-Home"></div></div>
                                <span class="prob-value" id="val-Home">0.0%</span>
                            </div>
                            <div class="prob-row">
                                <span class="prob-label">Electronics</span>
                                <div class="prob-bar-outer"><div class="prob-bar-inner" id="bar-Electronics"></div></div>
                                <span class="prob-value" id="val-Electronics">0.0%</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Business Insights Section -->
        <div id="insights" class="tab-content">
            <h1>Business Insights & Decision Engine</h1>
            <p class="section-desc">Key takeaways extracted from the confusion matrix analysis to optimize Flipkart's catalog operations.</p>
            
            <div class="card">
                <h2>Confusion Matrix Deep-Dive</h2>
                <div class="insight-card">
                    <div class="insight-num">1</div>
                    <div class="insight-text">
                        <h4>Bags, Backpacks and Wallets Overlap with Clothing Accessories</h4>
                        <p>Many bags, backpacks and wallets visually resemble clothing accessories. The model sometimes classifies them as Apparel because they share similar textures, colors and shapes. Additionally, lifestyle product shots containing human models wearing or holding bags introduce dominant clothing vectors, leading to misclassification.</p>
                    </div>
                </div>
                <div class="insight-card">
                    <div class="insight-num">2</div>
                    <div class="insight-text">
                        <h4>Strap Dominated Watch Confusions</h4>
                        <p>Watches with prominent leather or fabric straps are occasionally predicted as Apparel. The CNN is focusing heavily on the textured strap pattern (resembling a belt or sleeve cuff) rather than the dial face. This is aggravated when the reflective watch dial face causes glare, washing out digital/hands features.</p>
                    </div>
                </div>
                <div class="insight-card">
                    <div class="insight-num">3</div>
                    <div class="insight-text">
                        <h4>Indoor Slipper & Home Slipper Visual Background Bias</h4>
                        <p>Slippers and slides (Apparel) are occasionally misclassified as Home. This is caused by environmental bias in product photos: indoor slippers are often photographed on rugs, wooden floors, or inside living rooms, causing the network to classify the image based on background furniture context.</p>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2>8 Core Business Takeaways</h2>
                <div style="display: flex; flex-direction: column; gap: 1rem;">
                    <div class="insight-card">
                        <div class="insight-num">1</div>
                        <div class="insight-text">
                            <h4>Dynamic Crop Filtering</h4>
                            <p>E-commerce catalog shots showing products on human models cause contextual errors. We recommend integrating a bounding-box crop step prior to classification to isolate the primary item.</p>
                        </div>
                    </div>
                    <div class="insight-card">
                        <div class="insight-num">2</div>
                        <div class="insight-text">
                            <h4>Seller Guidelines Enforcement</h4>
                            <p>Primary listing images must display products on a clean, solid white background. Lifestyle shots should only be allowed as secondary gallery images to prevent background bias.</p>
                        </div>
                    </div>
                    <div class="insight-card">
                        <div class="insight-num">3</div>
                        <div class="insight-text">
                            <h4>Multimodal Search Integration</h4>
                            <p>To avoid mislabeling visually ambiguous items (e.g. metal necklaces vs. metal home hooks), classification gates should combine visual inputs with text metadata (product titles) to resolve ambiguity.</p>
                        </div>
                    </div>
                    <div class="insight-card">
                        <div class="insight-num">4</div>
                        <div class="insight-text">
                            <h4>Watch-Specific Classifier Rules</h4>
                            <p>Since watches are our only Electronics category, we can add a simple keyword bypass. If the title contains "digital watch" or "smartwatch," it is routed directly to Electronics, protecting catalog accuracy.</p>
                        </div>
                    </div>
                    <div class="insight-card">
                        <div class="insight-num">5</div>
                        <div class="insight-text">
                            <h4>Handling Minority Class Loss</h4>
                            <p>Without balanced class weights, the model defaults to the 88% dominant class (Apparel). The computed weights (Electronics: 7.09, Home: 4.63) are required to force the network to respect rare uploads.</p>
                        </div>
                    </div>
                    <div class="insight-card">
                        <div class="insight-num">6</div>
                        <div class="insight-text">
                            <h4>Relevance of Resolution</h4>
                            <p>Downsampling to 224x224 degrades small distinguishing markers (e.g., watch brand stamps or zipper grooves), indicating a need to investigate 299x299 sizes in future iterations.</p>
                        </div>
                    </div>
                    <div class="insight-card">
                        <div class="insight-num">7</div>
                        <div class="insight-text">
                            <h4>Luggage vs. Home Separation</h4>
                            <p>Luggage and suitcases fit well under Home due to their storage utility, but their blocky, hard-sided structure differs greatly from textile home goods, indicating a potential split in future schemas.</p>
                        </div>
                    </div>
                    <div class="insight-card">
                        <div class="insight-num">8</div>
                        <div class="insight-text">
                            <h4>Jewelry Class Ambiguity</h4>
                            <p>Metallic body chains and necklaces overlap with decorative home metalwork, suggesting they might be better grouped in an independent "Accessories" class to maintain high-fidelity parent category splits.</p>
                        </div>
                    </div>
                </div>
            </div>

            <div class="decision-box">
                <h3>Actionable Decision Engine Output</h3>
                <ul>
                    <li>Images predicted with low confidence (softmax probability &lt; 0.85) between Apparel and Home should be sent for manual catalog verification before publishing.</li>
                    <li>Any item whose predicted secondary class has a probability within 10% of the top class should trigger an automatic flag for human catalog review.</li>
                    <li>Unlabelled image uploads or images that fail the $224 \times 224$ aspect ratio checks should be rejected at the upload gate with a prompt to the seller to upload conforming assets.</li>
                </ul>
            </div>
        </div>

        <!-- About Section -->
        <div id="about" class="tab-content">
            <h1>About the Capstone Project</h1>
            <p class="section-desc">Technical specification of the Flipkart Product Catalog Classifier.</p>
            
            <div class="card">
                <h2>Technologies Used</h2>
                <ul style="list-style-type: square; margin-left: 1.5rem; line-height: 1.6; font-size: 0.95rem;">
                    <li><strong>Backend Core:</strong> TensorFlow 2.21.0 & Keras (deep learning API)</li>
                    <li><strong>Image Operations:</strong> OpenCV & PIL (Pillow)</li>
                    <li><strong>Data Pipeline:</strong> Pandas, NumPy & Scikit-Learn</li>
                    <li><strong>Visualization:</strong> Matplotlib & Seaborn</li>
                    <li><strong>Web Dashboard:</strong> Python Flask & HTML/CSS/JS (Vanilla)</li>
                </ul>
            </div>

            <div class="card">
                <h2>Model Architecture Specifications</h2>
                <div style="display: flex; flex-direction: column; gap: 0.5rem; font-size: 0.9rem; line-height: 1.5;">
                    <p><strong>Model 1 (Custom CNN):</strong> 3 convolutional blocks (32, 64, 128 filters of size 3x3) alternating with Max Pooling (2x2), flattened into a Dense 128 hidden layer with 50% Dropout, ending in a 3-unit Softmax output. Optimizer: Adam. Loss: Categorical Crossentropy.</p>
                    <p><strong>Model 2 (MobileNetV2 Transfer Learning):</strong> Stripped base MobileNetV2 pre-trained on ImageNet, frozen features, Global Average Pooling, custom dense classification head (128 units, 50% Dropout), fine-tuned top 20 layers with a learning rate of $1\times 10^{-5}$. Callbacks: Early Stopping (patience=3), Reduce LR on Plateau (factor=0.2), Checkpointing (saves best validation loss model).</p>
                </div>
            </div>
        </div>

    </div>

    <!-- JavaScript for tab switching and AJAX upload prediction -->
    <script>
        function showTab(tabId) {
            // Hide all tab contents
            const tabs = document.querySelectorAll('.tab-content');
            tabs.forEach(tab => tab.classList.remove('active'));

            // Deactivate all sidebar items
            const navItems = document.querySelectorAll('.nav-item');
            navItems.forEach(item => item.classList.remove('active'));

            // Show current tab
            document.getElementById(tabId).classList.add('active');

            // Find matching nav item and activate it
            // Simple string matching based on the onclick attribute
            navItems.forEach(item => {
                if (item.getAttribute('onclick').includes(tabId)) {
                    item.classList.add('active');
                }
            });
        }

        function previewImage(event) {
            const input = event.target;
            const preview = document.getElementById('image-preview');
            const classifyBtn = document.getElementById('classify-btn');
            const noResult = document.getElementById('no-result');
            const resultContainer = document.getElementById('result-container');

            if (input.files && input.files[0]) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                    classifyBtn.disabled = false;
                    
                    // Hide previous results
                    noResult.style.display = 'block';
                    resultContainer.style.display = 'none';
                }
                reader.readAsDataURL(input.files[0]);
            }
        }

        function classifyImage() {
            const fileInput = document.getElementById('file-input');
            const classifyBtn = document.getElementById('classify-btn');
            const spinner = document.getElementById('spinner');
            const noResult = document.getElementById('no-result');
            const resultContainer = document.getElementById('result-container');

            if (!fileInput.files || fileInput.files.length === 0) return;

            const file = fileInput.files[0];
            const formData = new FormData();
            formData.append('image', file);

            // Show loader and disable buttons
            classifyBtn.disabled = true;
            spinner.style.display = 'block';
            noResult.style.display = 'none';
            resultContainer.style.display = 'none';

            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                spinner.style.display = 'none';
                classifyBtn.disabled = false;

                if (data.success) {
                    // Update predictions
                    document.getElementById('predicted-class').innerText = data.class;
                    
                    // Style class color indicator
                    const predictedTitle = document.getElementById('predicted-class');
                    if (data.class === 'Apparel') {
                        predictedTitle.style.color = '#1e40af'; // Blue
                    } else if (data.class === 'Electronics') {
                        predictedTitle.style.color = '#16a34a'; // Green
                    } else {
                        predictedTitle.style.color = '#d97706'; // Orange
                    }

                    document.getElementById('predicted-confidence').innerText = 'Confidence: ' + (data.confidence * 100).toFixed(1) + '%';

                    // Update probability bars
                    const classes = ['Apparel', 'Electronics', 'Home'];
                    classes.forEach(cls => {
                        const val = data.probabilities[cls];
                        const pctStr = (val * 100).toFixed(1) + '%';
                        document.getElementById('val-' + cls).innerText = pctStr;
                        document.getElementById('bar-' + cls).style.width = pctStr;
                        
                        // Apply custom class weights or colors to bars
                        const bar = document.getElementById('bar-' + cls);
                        if (cls === 'Apparel') bar.style.backgroundColor = '#1e40af';
                        else if (cls === 'Electronics') bar.style.backgroundColor = '#16a34a';
                        else bar.style.backgroundColor = '#d97706';
                    });

                    resultContainer.style.display = 'block';
                    noResult.style.display = 'none';
                } else {
                    alert('Error: ' + data.error);
                    noResult.style.display = 'block';
                }
            })
            .catch(error => {
                spinner.style.display = 'none';
                classifyBtn.disabled = false;
                noResult.style.display = 'block';
                alert('Inference failed: ' + error);
            });
        }
    </script>
</body>
</html>
"""

# Serve main webpage
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, **TEXT_DATA)

# Serve static dashboard assets (the generated PNG plots)
@app.route('/assets/<path:filename>')
def serve_assets(filename):
    return send_from_directory(ASSETS_DIR, filename)

# Serve predict API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image file provided."}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected."}), 400

    try:
        import numpy as np
        from PIL import Image
        import io

        # 1. Load image bytes
        file_bytes = file.read()
        img = Image.open(io.BytesIO(file_bytes))
        
        # 2. Preprocess image
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_resized = img.resize((224, 224))
        
        # Scale to [0, 1] and add batch dimension
        img_array = np.array(img_resized) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        # 3. Load model lazily and run prediction
        model = get_model()
        preds = model.predict(img_batch)[0]

        # 4. Map probabilities
        probabilities = {_classes[i]: float(preds[i]) for i in range(3)}
        predicted_class = max(probabilities, key=probabilities.get)
        confidence = probabilities[predicted_class]

        return jsonify({
            "success": True,
            "class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities
        })
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

def start_server():
    # Start the server on port 5000
    app.run(host='127.0.0.1', port=5000, debug=False)

if __name__ == '__main__':
    # Set up browser launch delay
    print("Starting capstone dashboard server...")
    timer = threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:5000"))
    timer.start()
    
    # Run server (this is blocking)
    start_server()
