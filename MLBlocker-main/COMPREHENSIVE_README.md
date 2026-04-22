# MLBlocker: Complete Step-by-Step Guide

## 🎯 Project Overview

MLBlocker is the world's first AI-powered anti-ad & tracker system that blocks ads and trackers based on machine learning models trained on 83K+ websites. This comprehensive guide explains the complete workflow from dataset training to Chrome extension deployment.

---

## 📊 Step 1: Dataset Training Process

### 1.1 Dataset Structure

The project uses multiple datasets for comprehensive training:

**Primary Training Datasets:**
- `all_df_883_train.csv` - Main training dataset with 883 features
- `all_df_883_test.csv` - Main testing dataset
- `AdFlush_train.csv` / `AdFlush_test.csv` - AdFlush-specific datasets

**Adversarial Testing Datasets:**
- `GAN_mutated_AdFlush.csv` - GAN-generated AdFlush mutations
- `GAN_mutated_AdGraph.csv` - GAN-generated AdGraph mutations  
- `GAN_mutated_WebGraph.csv` - GAN-generated WebGraph mutations

**JavaScript Obfuscation Datasets:**
- `JS_obfuscated_gnirts.csv` - String obfuscation attacks
- `JS_obfuscated_javascript_obfuscator.csv` - JS obfuscator attacks
- `JS_obfuscated_wobfuscator.csv` - Web obfuscator attacks

### 1.2 Feature Engineering Pipeline

**Step 1: Feature Categories (features.yaml)**

The system extracts 4 main feature categories:

1. **MLBlocker Features (28 features)**:
   - Content policy type encoding
   - URL characteristics (length, third-party detection)
   - JavaScript behavior analysis (storage operations, requests)
   - Character n-grams (ng_* features)
   - AST-based features (depth, breadth, identifier analysis)

2. **AdGraph Features (29 features)**:
   - Graph-based features (nodes, edges, degree metrics)
   - Script relationship analysis
   - Content policy and keyword detection

3. **WebGraph Features (120 features)**:
   - Comprehensive web graph analysis
   - Indirect graph metrics
   - Request/response pattern analysis

4. **WTAGraph Features (400+ features)**:
   - URL and FQDN character embeddings
   - API usage patterns
   - HTTP method and content type analysis

**Step 2: Feature Selection Process**

```bash
cd source
python main.py -p feature-eng
```

This executes a 4-step feature selection:

1. **Point-Biserial Correlation Analysis**: Removes features with p-value > 0.1
2. **Recursive Feature Elimination with Cross-Validation (RFECV)**: 
   - Uses RandomForest classifier
   - 5-fold cross-validation
   - Selects optimal feature subset
3. **Pearson & Spearman Correlation**: Removes highly correlated features
4. **Feature Importance Ranking**: Combines Random Forest Importance and Permutation Importance

**Step 3: Character Embeddings**

The system uses Word2Vec for character-level embeddings:
- Request URLs: 200-dimensional vectors
- FQDNs: 30-dimensional vectors
- Pre-trained embeddings stored in JSON files (`reqwordvec.json`, `fqdnwordvec.json`)

### 1.3 Model Training Process

**Step 1: Model Selection with H2O AutoML**

```bash
cd source
python main.py -p model-sel
```

**Training Configuration:**
- Uses H2O AutoML for automated model selection
- 1-hour runtime limit
- 5-fold cross-validation
- Excludes XGBoost and StackedEnsemble
- Primarily uses GBM (Gradient Boosting Machine)

**Step 2: Model Conversion**

The trained H2O model is converted to ONNX format:
- Converts H2O MOJO format to ONNX for browser deployment
- Target opset: 9
- Enables real-time inference in Chrome extension

**Model Files Generated:**
- `MLBlocker_mojo.zip` - H2O MOJO model (37MB)
- `MLBlocker.onnx` - ONNX model (69MB)

### 1.4 Adversarial Training

**Generate GAN-based adversarial examples:**

```bash
python main.py -p train-gan -s mlblocker
```

**Adversarial Training Process:**
- Uses `tabgan` library for synthetic data generation
- Creates adversarial examples to improve robustness
- Tests against various obfuscation techniques
- Generates `custom_GAN_mutated_mlblocker.csv`

### 1.5 Performance Evaluation

**Evaluate model performance on different datasets:**

```bash
# Standard test set
python main.py -p performance-eval -d testset -m onnx

# GAN attacks
python main.py -p performance-eval -d gan -m onnx

# JavaScript obfuscation attacks
python main.py -p performance-eval -d gnirts -m onnx
python main.py -p performance-eval -d javascript-obfuscator -m onnx
python main.py -p performance-eval -d wobfuscator -m onnx
```

**Evaluation Metrics:**
- Accuracy, Precision, Recall, F1 Score
- AUROC (Area under ROC curve)
- False Negative Rate (FNR) - Missed detections
- False Positive Rate (FPR) - Incorrect blocks
- Attack Success Rate (ASR) - For adversarial testing

---

## 🤖 Step 2: Model Architecture

### 2.1 Model Types

**1. H2O GBM Model (`MLBlocker_mojo.zip`):**
- Gradient Boosting Machine
- High accuracy for server-side deployment
- 37MB file size

**2. ONNX Model (`MLBlocker.onnx`):**
- Optimized for browser inference
- 69MB file size
- Real-time prediction capability

### 2.2 Model Features Used in Extension

The Chrome extension uses these 10 key features:
1. `is_third_party` - Third-party request detection
2. `url_77_max` - URL character embedding max value
3. `content_policy_type` - Resource type encoding
4. `keyword_char_present` - Ad keyword detection
5. `url_41_mean` - URL character embedding mean
6. `url_144_std` - URL character embedding std dev
7. `url_3_mean` - URL character embedding mean
8. `url_50_std` - URL character embedding std dev
9. `url_6_std` - URL character embedding std dev
10. `num_get_cookie` - Cookie access count

---

## 🌐 Step 3: Chrome Extension Architecture

### 3.1 Extension Structure

**Core Files:**
```
extension chrome/
├── manifest.json          # Extension configuration (Manifest V3)
├── background.js          # Service worker for request interception
├── model.js              # ONNX model inference engine
├── popup.html/js          # User interface for extension control
├── webpack.config.js     # Build configuration
└── dist_chrome/          # Built extension artifacts
    ├── MLBlocker.onnx    # Optimized ONNX model (7MB)
    ├── background.js     # Compiled service worker
    ├── model.js          # Compiled inference engine
    ├── *.wasm           # ONNX Runtime WASM modules
    ├── reqwordvec.json  # Request URL embeddings
    ├── fqdnwordvec.json # FQDN embeddings
    └── rules.json       # Blocking rules
```

### 3.2 Extension Workflow

**Step 1: Request Interception**
```javascript
// background.js intercepts web requests
chrome.webRequest.onBeforeSendHeaders.addListener(
  callback, {urls: ["<all_urls>"]}, ['requestHeaders']
);
```

**Step 2: Feature Extraction**
- **URL Analysis**: Extracts URL patterns and characteristics
- **JavaScript Parsing**: AST analysis for obfuscation detection
- **Content Policy Analysis**: Resource type classification
- **Character Embeddings**: Uses pre-trained vectors for URL/FQDN

**Step 3: Real-time Inference**
```javascript
// model.js performs ONNX inference
const session = await ort.InferenceSession.create(modelUrl);
const results = await session.run(feeds);
const prediction = results.label.data[0];
```

**Step 4: Blocking Decision**
- Uses Chrome's declarativeNetRequest API for efficient blocking
- Dynamic rule generation based on ML predictions
- Maintains block count and performance metrics

### 3.3 Key Technologies

**Frontend:**
- **Manifest V3**: Latest Chrome extension standard
- **Service Workers**: Background processing
- **WebAssembly**: ONNX Runtime execution
- **Webpack**: Module bundling and optimization

**ML Inference:**
- **ONNX Runtime Web**: Browser-based model execution
- **Tensor Operations**: Float32Array for efficient computation
- **WASM Optimization**: Hardware acceleration support

**JavaScript Analysis:**
- **Acorn**: AST parsing
- **Cherow**: Alternative AST parser
- **tldjs**: Domain extraction utilities

---

## 🚀 Step 4: Complete Setup Instructions

### 4.1 Prerequisites

- Python 3.7+
- Node.js 14+
- Chrome browser (for extension)
- H2O AI platform

### 4.2 Environment Setup

**Step 1: Clone and Setup Python Environment**
```bash
git clone <repository-url>
cd MLBlocker-main
pip install -r requirements.txt
```

**Step 2: Setup Node.js Dependencies**
```bash
cd "extension chrome"
npm install
```

### 4.3 Training Pipeline

**Step 1: Feature Engineering**
```bash
cd source
python main.py -p feature-eng
```

**Step 2: Model Training**
```bash
python main.py -p model-sel
```

**Step 3: Generate Adversarial Data (Optional)**
```bash
python main.py -p train-gan -s mlblocker
```

**Step 4: Performance Evaluation**
```bash
python main.py -p performance-eval -d testset -m onnx
```

### 4.4 Chrome Extension Installation

**Step 1: Build the Extension**
```bash
cd "extension chrome"
npm run build
```

**Step 2: Load in Chrome**
1. Open Chrome Extensions page (`chrome://extensions/`)
2. Enable Developer mode
3. Click "Load unpacked"
4. Select the `dist_chrome` folder

**Step 3: Configure Extension**
- Toggle MLBlocker on/off via popup
- Monitor blocked ads/trackers count
- View performance metrics

---

## 📈 Step 5: How It Works in Practice

### 5.1 Real-time Processing Flow

1. **User visits website**
2. **Extension intercepts all web requests**
3. **Features are extracted in real-time:**
   - URL characteristics and embeddings
   - Third-party detection
   - Content type analysis
   - JavaScript behavior analysis
4. **ONNX model makes prediction**
5. **If prediction = 1 (ad/tracker):**
   - Dynamic blocking rule is created
   - Request is blocked
   - Block counter increments
6. **Performance metrics are tracked**

### 5.2 Feature Extraction Details

**URL Features:**
- Length analysis
- Third-party detection using tldjs
- Character-level embeddings (200-dim vectors)
- Statistical features (mean, std, max)

**JavaScript Features:**
- Cookie access detection
- Storage operation counting
- AST-based obfuscation detection
- N-gram analysis

**Content Policy Features:**
- Resource type encoding
- Request method analysis
- Header analysis

### 5.3 Model Inference Process

1. **Feature vector creation** (10 features)
2. **Tensor conversion** to Float32Array
3. **ONNX runtime inference**
4. **Decision making** based on threshold
5. **Rule creation** for blocking/allowing

---

## 🔧 Step 6: Configuration and Customization

### 6.1 Feature Configuration
- Edit `features.yaml` to modify feature sets
- Adjust feature selection parameters in `main.py`
- Customize embedding dimensions in `mlblocker_encodings.py`

### 6.2 Model Parameters
- Modify H2O AutoML settings in `model_selection()`
- Adjust GAN training parameters in `trainGAN()`
- Tune inference thresholds in extension code

### 6.3 Extension Settings
- Customize blocking rules in `background.js`
- Adjust UI elements in `popup.html`
- Modify manifest permissions as needed

---

## 📊 Step 7: Performance and Evaluation

### 7.1 Benchmark Results

The system has been evaluated against:
- **Standard test set**: Baseline performance
- **GAN attacks**: Synthetic adversarial examples
- **JavaScript obfuscation**: Various obfuscation techniques
- **Real-world websites**: 83K+ website training data

### 7.2 Key Metrics
- **Inference Speed**: Real-time processing in browser
- **Memory Usage**: Optimized for extension constraints
- **Block Accuracy**: High precision to avoid false positives
- **Robustness**: Resistant to common evasion techniques

---

## 🛠️ Step 8: Development and Troubleshooting

### 8.1 Common Issues

**Model Loading Issues:**
- Ensure ONNX model is in `dist_chrome/` folder
- Check WebAssembly files are present
- Verify CSP settings allow WASM execution

**Feature Extraction Problems:**
- Check embedding JSON files are loaded
- Verify tldjs is working correctly
- Ensure JavaScript parsing is functioning

**Performance Issues:**
- Monitor inference time in popup
- Check memory usage in Chrome dev tools
- Optimize feature extraction if slow

### 8.2 Debugging Tools

**Chrome Extension Debugging:**
1. Open `chrome://extensions/`
2. Click "Inspect views: background page"
3. Check console for errors
4. Use Network tab to monitor requests

**Model Debugging:**
1. Check model outputs in background.js console
2. Verify feature values are being calculated correctly
3. Test with known ad/tracker URLs

---

## 📚 File Structure Summary

```
MLBlocker-main/
├── README.md                    # Original documentation
├── COMPREHENSIVE_README.md      # This detailed guide
├── requirements.txt             # Python dependencies
├── source/                      # ML pipeline code
│   ├── main.py                 # Main training/evaluation script
│   ├── features.yaml           # Feature definitions
│   ├── mlblocker_encodings.py  # Feature extraction utilities
│   └── processing/             # Data processing scripts
├── dataset/                     # Training and test datasets
│   ├── *_train.csv            # Training datasets
│   ├── *_test.csv             # Test datasets
│   └── GAN_mutated_*.csv      # Adversarial datasets
├── model/                       # Trained models
│   ├── AdFlush.onnx           # ONNX model for extension
│   └── AdFlush_mojo.zip       # H2O MOJO model
├── extension chrome/           # Chrome extension source
│   ├── background.js           # Service worker
│   ├── model.js               # ONNX inference
│   ├── popup.html/js          # User interface
│   ├── package.json           # Node dependencies
│   └── dist_chrome/           # Built extension
└── assets/                      # Documentation assets
```

---

## 🎯 Summary

MLBlocker represents a complete end-to-end machine learning pipeline for ad and tracker blocking:

1. **Data Collection**: 83K+ websites with labeled ads/trackers
2. **Feature Engineering**: Multi-modal feature extraction with 883+ features
3. **Model Training**: H2O AutoML with adversarial robustness
4. **Model Deployment**: ONNX conversion for browser inference
5. **Real-time Blocking**: Chrome extension with sub-second inference
6. **Continuous Learning**: GAN-based adversarial training

The system demonstrates state-of-the-art performance in blocking ads and trackers while maintaining low false positive rates and real-time performance in the browser environment.

**MLBlocker** - Intelligent ad and tracker blocking powered by machine learning 🚀
