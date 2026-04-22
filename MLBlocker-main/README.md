# MLBlocker: AI-Powered Ad and Tracker Blocking System

MLBlocker is a machine learning-based browser extension that uses advanced AI models to detect and block ads and trackers in real-time. It's the world's first AI-powered anti-ad & tracker system trained on 83K+ websites.

## 🏗️ Project Overview

MLBlocker combines sophisticated machine learning techniques with browser extension technology to provide intelligent ad and tracker blocking. The system uses feature extraction, model training, and real-time inference to identify malicious or unwanted web content.

### Key Components:
- **Machine Learning Pipeline**: Feature extraction, model training, and evaluation
- **Chrome Extension**: Real-time blocking with ONNX model inference
- **Dataset Processing**: Multiple datasets for training and adversarial testing
- **Model Architecture**: H2O AutoML with ONNX conversion for browser deployment

---

## 📊 Dataset and Training Process

### Dataset Structure

The project uses multiple datasets for comprehensive training and testing:

#### Primary Datasets:
- **`all_df_883_train.csv`**: Main training dataset (883 features)
- **`all_df_883_test.csv`**: Main testing dataset
- **`AdFlush_train.csv` / `AdFlush_test.csv`**: AdFlush-specific datasets
- **`GAN_mutated_*.csv`**: Adversarial datasets for robustness testing
  - `GAN_mutated_AdFlush.csv`
  - `GAN_mutated_AdGraph.csv` 
  - `GAN_mutated_WebGraph.csv`
- **JavaScript Obfuscation Datasets**:
  - `JS_obfuscated_gnirts.csv`
  - `JS_obfuscated_javascript_obfuscator.csv`
  - `JS_obfuscated_wobfuscator.csv`

### Feature Engineering Process

The system extracts and processes multiple types of features:

#### 1. **Feature Categories** (defined in `features.yaml`):

**MLBlocker Features (28 features)**:
- Content policy type encoding
- URL characteristics (length, third-party detection)
- JavaScript behavior analysis (storage operations, requests)
- Character n-grams (ng_* features)
- AST-based features (depth, breadth, identifier analysis)

**AdGraph Features (29 features)**:
- Graph-based features (nodes, edges, degree metrics)
- Script relationship analysis
- Content policy and keyword detection

**WebGraph Features (120 features)**:
- Comprehensive web graph analysis
- Indirect graph metrics
- Request/response pattern analysis

**WTAGraph Features (400+ features)**:
- URL and FQDN character embeddings
- API usage patterns
- HTTP method and content type analysis

#### 2. **Feature Selection Pipeline**:

1. **Point-Biserial Correlation Analysis**: Removes features with p-value > 0.1
2. **Recursive Feature Elimination with Cross-Validation (RFECV)**: 
   - Uses RandomForest classifier
   - 5-fold cross-validation
   - Selects optimal feature subset
3. **Pearson & Spearman Correlation**: Removes highly correlated features
4. **Feature Importance Ranking**: Combines Random Forest Importance and Permutation Importance

#### 3. **Character Embeddings**:

The system uses Word2Vec for character-level embeddings:
- **Request URLs**: 200-dimensional vectors
- **FQDNs**: 30-dimensional vectors
- Pre-trained embeddings stored in JSON files (`reqwordvec.json`, `fqdnwordvec.json`)

### Training Pipeline

#### Model Training Process:

1. **Data Preparation**:
   ```python
   # Load and preprocess datasets
   train_df = pd.read_csv('MLBlocker_train.csv')
   test_df = pd.read_csv('MLBlocker_test.csv')
   ```

2. **Feature Engineering**:
   ```bash
   python main.py -p feature-eng
   ```

3. **Model Selection with H2O AutoML**:
   ```bash
   python main.py -p model-sel
   ```
   - Uses H2O AutoML for automated model selection
   - 1-hour runtime limit
   - 5-fold cross-validation
   - Excludes XGBoost and StackedEnsemble
   - Primarily uses GBM (Gradient Boosting Machine)

4. **Model Conversion**:
   - Converts H2O MOJO format to ONNX for browser deployment
   - Target opset: 9
   - Enables real-time inference in Chrome extension

#### Adversarial Training:

The system includes GAN-based adversarial training:
```bash
python main.py -p train-gan -s mlblocker
```

- Uses `tabgan` library for synthetic data generation
- Creates adversarial examples to improve robustness
- Tests against various obfuscation techniques

---

## 🤖 Model Architecture and Deployment

### Model Types:

1. **H2O GBM Model** (`MLBlocker_mojo.zip`):
   - Gradient Boosting Machine
   - High accuracy for server-side deployment
   - 37MB file size

2. **ONNX Model** (`MLBlocker.onnx`):
   - Optimized for browser inference
   - 69MB file size
   - Real-time prediction capability

### Model Performance Metrics:

The system evaluates models using:
- **Accuracy**: Overall prediction accuracy
- **Precision/Recall**: Balance between false positives/negatives
- **F1 Score**: Harmonic mean of precision and recall
- **AUROC**: Area under ROC curve
- **False Negative Rate (FNR)**: Missed detections
- **False Positive Rate (FPR)**: Incorrect blocks
- **Attack Success Rate (ASR)**: For adversarial testing

### Performance Evaluation:

```bash
python main.py -p performance-eval -d testset -m onnx
```

Available datasets for evaluation:
- `testset`: Standard test dataset
- `gan`: GAN-generated adversarial examples
- `gnirts`: String obfuscation attacks
- `javascript-obfuscator`: JS obfuscation attacks
- `wobfuscator`: Web obfuscation attacks

---

## 🌐 Chrome Extension Architecture

### Extension Structure:

The Chrome extension is located in `extension chrome/` with the following structure:

#### Core Files:
- **`manifest.json`**: Extension configuration (Manifest V3)
- **`background.js`**: Service worker for request interception
- **`model.js`**: ONNX model inference engine
- **`popup.html/js`**: User interface for extension control
- **`webpack.config.js`**: Build configuration

#### Build Artifacts (`dist_chrome/`):
- **`AdFlush.onnx`**: Optimized ONNX model (7MB)
- **`background.js`**: Compiled service worker
- **`model.js`**: Compiled inference engine
- **WebAssembly files**: ONNX Runtime WASM modules
- **Pre-trained embeddings**: JSON files for character vectors

### Extension Workflow:

#### 1. **Request Interception**:
```javascript
// background.js intercepts web requests
chrome.webRequest.onBeforeRequest.addListener(
  callback, {urls: ["<all_urls>"]}, ["blocking"]
);
```

#### 2. **Feature Extraction**:
- **URL Analysis**: Extracts URL patterns and characteristics
- **JavaScript Parsing**: AST analysis for obfuscation detection
- **Content Policy Analysis**: Resource type classification
- **Character Embeddings**: Uses pre-trained vectors for URL/FQDN

#### 3. **Real-time Inference**:
```javascript
// model.js performs ONNX inference
const session = await ort.InferenceSession.create(modelUrl);
const results = await session.run(feeds);
const prediction = results.label.data[0];
```

#### 4. **Blocking Decision**:
- Uses Chrome's declarativeNetRequest API for efficient blocking
- Dynamic rule generation based on ML predictions
- Maintains block count and performance metrics

### Key Technologies:

#### Frontend:
- **Manifest V3**: Latest Chrome extension standard
- **Service Workers**: Background processing
- **WebAssembly**: ONNX Runtime execution
- **Webpack**: Module bundling and optimization

#### ML Inference:
- **ONNX Runtime Web**: Browser-based model execution
- **Tensor Operations**: Float32Array for efficient computation
- **WASM Optimization**: Hardware acceleration support

#### JavaScript Analysis:
- **Acorn**: AST parsing
- **Cherow**: Alternative AST parser
- **tldjs**: Domain extraction utilities

---

## 🚀 Installation and Setup

### Prerequisites:

- Python 3.7+
- Node.js 14+
- Chrome browser (for extension)
- H2O AI platform

### Environment Setup:

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd MLBlocker-main
   ```

2. **Set up Python environment**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Node.js dependencies**:
   ```bash
   cd "extension chrome"
   npm install
   ```

### Training Pipeline:

1. **Feature Engineering**:
   ```bash
   cd source
   python main.py -p feature-eng
   ```

2. **Model Training**:
   ```bash
   python main.py -p model-sel
   ```

3. **Generate Adversarial Data**:
   ```bash
   python main.py -p train-gan -s mlblocker
   ```

4. **Performance Evaluation**:
   ```bash
   python main.py -p performance-eval -d testset -m onnx
   ```

### Chrome Extension Installation:

1. **Build the extension**:
   ```bash
   cd "extension chrome"
   npm run build
   ```

2. **Load in Chrome**:
   - Open Chrome Extensions page (`chrome://extensions/`)
   - Enable Developer mode
   - Click "Load unpacked"
   - Select the `dist_chrome` folder

3. **Configure Extension**:
   - Toggle MLBlocker on/off via popup
   - Monitor blocked ads/trackers count
   - View performance metrics

---

## 📈 Performance and Evaluation

### Benchmark Results:

The system has been evaluated against:
- **Standard test set**: Baseline performance
- **GAN attacks**: Synthetic adversarial examples
- **JavaScript obfuscation**: Various obfuscation techniques
- **Real-world websites**: 83K+ website training data

### Key Metrics:
- **Inference Speed**: Real-time processing in browser
- **Memory Usage**: Optimized for extension constraints
- **Block Accuracy**: High precision to avoid false positives
- **Robustness**: Resistant to common evasion techniques

---

## 🔧 Configuration and Customization

### Feature Configuration:
- Edit `features.yaml` to modify feature sets
- Adjust feature selection parameters in `main.py`
- Customize embedding dimensions in `mlblocker_encodings.py`

### Model Parameters:
- Modify H2O AutoML settings in `model_selection()`
- Adjust GAN training parameters in `trainGAN()`
- Tune inference thresholds in extension code

### Extension Settings:
- Customize blocking rules in `background.js`
- Adjust UI elements in `popup.html`
- Modify manifest permissions as needed

---

## 📚 File Structure

```
MLBlocker-main/
├── README.md                    # This documentation
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

## 🛠️ Development and Contributing

### Code Style:
- Python: Follow PEP 8 guidelines
- JavaScript: Use ESLint configuration
- Comments: Document complex algorithms

### Testing:
- Unit tests for feature extraction
- Integration tests for model pipeline
- Extension testing in Chrome developer mode

### Performance Optimization:
- Monitor inference latency
- Optimize feature extraction speed
- Minimize extension memory usage

---

## 📄 License and Citation

This project represents cutting-edge research in AI-based ad blocking. Please cite appropriately if used in academic work.

## 🤝 Support and Community

For issues, questions, or contributions:
- Check existing documentation
- Review feature configuration options
- Test with different datasets
- Contribute to model improvement

---

**MLBlocker** - Intelligent ad and tracker blocking powered by machine learning 🚀
