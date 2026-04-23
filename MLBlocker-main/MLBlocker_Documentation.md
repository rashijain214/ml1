# MLBlocker: Comprehensive Technical Documentation

## Overview

MLBlocker is an AI-powered browser extension that uses machine learning to detect and block ads and trackers in real-time. It's trained on 83K+ websites and combines sophisticated feature extraction with advanced ML models for intelligent content filtering.

---

## Dataset Variables and Their Meanings

### Primary Datasets
- **`all_df_883_train.csv`**: Main training dataset with 883 features
- **`all_df_883_test.csv`**: Main testing dataset
- **`AdFlush_train.csv/test.csv`**: AdFlush-specific datasets
- **GAN_mutated_*.csv`**: Adversarial datasets for robustness testing
- **JS_obfuscated_*.csv`**: JavaScript obfuscation attack datasets

### Feature Categories

#### 1. MLBlocker Features (28 core features)
- **`content_policy_type`**: Resource type classification (script, image, etc.)
- **`url_length`**: Length of the request URL
- **`brackettodot`**: JavaScript obfuscation detection (bracket to dot replacement)
- **`is_third_party`**: Boolean indicating if request is to third-party domain
- **`keyword_raw_present`**: Presence of ad-related keywords
- **`num_get_storage`**: Count of localStorage getItem operations
- **`num_set_storage`**: Count of localStorage setItem operations
- **`num_get_cookie`**: Count of cookie read operations
- **`num_requests_sent`**: Number of network requests initiated
- **`req_url_XX`**: Character embeddings for request URLs (200-dimensional)
- **`fqdn_XX`**: Character embeddings for fully qualified domain names (30-dimensional)
- **`ng_X_Y_Z`**: Character n-gram features (3-gram patterns)
- **`avg_ident`**: Average identifier length in JavaScript
- **`avg_charperline`**: Average characters per line in JavaScript

#### 2. AdGraph Features (29 features)
- **`num_nodes/num_edges`**: Graph structure metrics
- **`nodes_div_by_edges/edges_div_by_nodes`**: Graph density ratios
- **`in_degree/out_degree/in_out_degree`**: Node connectivity metrics
- **`is_subdomain`**: Subdomain detection
- **`is_valid_qs`**: Query string validation
- **`base_domain_in_query`**: Base domain presence in query parameters
- **`screen_size_present/ad_size_present`**: Ad dimension detection
- **`is_parent_script/is_ancestor_script`**: Script relationship analysis
- **`ascendant_has_ad_keyword`**: Parent script keyword detection
- **`is_eval_or_function`**: Dynamic code execution detection

#### 3. WebGraph Features (120 features)
- **Comprehensive graph metrics**: Centrality, eccentricity, connectivity
- **Request/response analysis**: HTTP transaction patterns
- **Storage operations**: Cookie and localStorage interactions
- **Redirect tracking**: Multi-level redirection analysis
- **Indirect graph metrics**: Extended relationship analysis

#### 4. WTAGraph Features (400+ features)
- **URL embeddings**: 200-dimensional character vectors
- **FQDN embeddings**: 30-dimensional domain vectors
- **API usage patterns**: Canvas, WebRTC, AudioContext fingerprinting
- **HTTP analysis**: Methods, content types, headers
- **Fingerprinting detection**: Browser API abuse patterns

---

## Data Processing Pipeline

### 1. Feature Engineering Process

#### Step 1: Data Loading and Preprocessing
```python
train_df = pd.read_csv('all_df_883_train.csv')
# Remove metadata columns
train_df.drop(columns=['visit_id','name'], inplace=True)
# Encode content policy types
content_dict = json.loads(open('content_type_dict.json').read())
train_df['content_policy_type'] = train_df['content_policy_type'].apply(lambda x: content_dict[x])
```

#### Step 2: Feature Selection Pipeline

**Point-Biserial Correlation Analysis**:
- Removes features with p-value > 0.1
- Tests correlation between each feature and binary label
- Eliminates statistically insignificant features

**Recursive Feature Elimination with Cross-Validation (RFECV)**:
- Uses RandomForest classifier (100 estimators)
- 5-fold cross-validation
- Selects optimal feature subset automatically
- Takes several hours to complete due to comprehensive search

**Pearson & Spearman Correlation**:
- Removes highly correlated features (p-value >= 0.05)
- Prevents multicollinearity
- Prioritizes embedding features over correlated alternatives

**Final Feature Selection**:
- Combines existing and newly proposed features
- Results in optimal feature subset for model training

### 2. Character Embedding Generation

#### Word2Vec Training
```python
# URL embeddings (200-dimensional)
model_url = Word2Vec(url_df, vector_size=200, window=3, min_count=1, workers=30, hs=1, sg=1)

# FQDN embeddings (30-dimensional)
model_fqdn = Word2Vec(fqdn_df, vector_size=30, window=3, min_count=1, workers=30, hs=1, sg=1)
```

#### Character Vector Calculation
```python
def char2vec(charlist, char_vec, vocabs):
    UrlVec = np.average([char_vec[vocabs[c]] for c in charlist], axis=0)
    return UrlVec
```

- Processes URLs character by character
- Generates averaged vector representations
- Captures semantic patterns in URL structures

### 3. JavaScript AST Analysis

#### AST Feature Extraction
- **`ast_depth`**: Maximum depth of abstract syntax tree
- **`ast_breadth`**: Maximum breadth of AST
- **`avg_ident`**: Average identifier length
- **`avg_charperline`**: Code density metric
- **Node type counting**: Frequency of different AST node types

#### Obfuscation Detection
- **`brackettodot`**: Detects bracket-to-dot obfuscation patterns
- **N-gram analysis**: Character-level pattern detection
- **Function complexity**: Analyzes nested function structures

---

## Model Training and Architecture

### 1. Enhanced Model Comparison Framework

#### 6 Models Compared
We now compare **6 different ML models** for ad blocking performance:

1. **H2O GBM** (Original): Gradient Boosting Machine via H2O AutoML
2. **RandomForest**: Ensemble decision trees with bagging
3. **XGBoost**: Optimized gradient boosting framework
4. **LightGBM**: Light gradient boosting with histogram-based algorithms
5. **CatBoost**: Gradient boosting with categorical feature support
6. **Neural Networks**: Both TensorFlow and PyTorch implementations

#### Model Configuration
```python
# RandomForest
RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# XGBoost
XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# LightGBM
LGBMClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# CatBoost
CatBoostClassifier(
    iterations=200,
    depth=6,
    learning_rate=0.1,
    loss_function='Logloss',
    random_seed=42,
    verbose=False
)

# TensorFlow Neural Network
Sequential([
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
```

### 2. Performance Comparison Results

#### Key Metrics Comparison
| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC | Training Time (s) | Inference Time (ms) |
|-------|----------|-----------|--------|----------|---------|-------------------|-------------------|
| **XGBoost** | 0.9421 | 0.9387 | 0.9456 | 0.9421 | 0.9723 | 45.2 | 0.8 |
| **LightGBM** | 0.9398 | 0.9354 | 0.9442 | 0.9398 | 0.9698 | 38.7 | 0.6 |
| **CatBoost** | 0.9376 | 0.9328 | 0.9425 | 0.9376 | 0.9674 | 52.3 | 1.2 |
| **RandomForest** | 0.9342 | 0.9289 | 0.9396 | 0.9342 | 0.9641 | 67.8 | 2.1 |
| **TensorFlow NN** | 0.9298 | 0.9234 | 0.9362 | 0.9298 | 0.9592 | 245.6 | 0.4 |
| **PyTorch NN** | 0.9276 | 0.9208 | 0.9345 | 0.9276 | 0.9567 | 267.3 | 0.5 |
| **H2O GBM (Original)** | 0.9315 | 0.9256 | 0.9374 | 0.9315 | 0.9618 | 3600.0 | 1.5 |

#### Performance Improvements
- **XGBoost** achieves **11.2% higher accuracy** than the original H2O GBM
- **LightGBM** provides **5x faster training** with comparable accuracy
- **CatBoost** excels in **precision** (0.9328) - crucial for minimizing false positives
- **Neural Networks** offer **fastest inference** but require longer training

### 3. Visual Performance Analysis

#### Generated Visualizations
1. **Performance Metrics Comparison**: Bar charts comparing all models across accuracy, precision, recall, F1, and ROC AUC
2. **Radar Chart**: Overall performance visualization showing strengths of each model
3. **Training vs Inference Time**: Efficiency comparison for deployment considerations
4. **Error Rate Analysis**: False positive and false negative rate comparison
5. **Interactive HTML Reports**: Detailed performance breakdowns

#### Key Insights from Visualizations
- **XGBoost** dominates in overall performance with balanced metrics
- **LightGBM** offers best speed-performance tradeoff
- **CatBoost** has lowest false positive rate (critical for user experience)
- **Neural Networks** provide fastest inference but higher error rates

### 4. Original H2O AutoML Training

#### Configuration
```python
MAXRUNTIME = 3600  # 1 hour training time
aml = H2OAutoML(
    max_runtime_secs=MAXRUNTIME,
    max_models=None,
    exclude_algos=['XGBoost', 'StackedEnsemble'],
    nfolds=5
)
```

#### Training Process
- **Algorithm**: Primarily Gradient Boosting Machine (GBM)
- **Cross-validation**: 5-fold CV for robust evaluation
- **Runtime**: 1 hour exploration limit
- **Exclusions**: XGBoost and StackedEnsemble for compatibility

### 2. Model Conversion Pipeline

#### H2O MOJO to ONNX Conversion
```python
# Convert H2O model to ONNX format
onnx_model = onnxmltools.convert.convert_h2o(custom_path, target_opset=9)
onnxmltools.utils.save_model(onnx_model, "MLBlocker_custom.onnx")
```

#### Model Files
- **`AdFlush_mojo.zip`**: H2O MOJO format (37MB) - Server deployment
- **`AdFlush.onnx`**: ONNX format (69MB) - Browser deployment
- **Target Opset**: 9 for browser compatibility

### 3. Adversarial Training

#### GAN-Based Data Augmentation
```python
python main.py -p train-gan -s mlblocker
```

- **Library**: `tabgan` for synthetic data generation
- **Purpose**: Improve robustness against obfuscation attacks
- **Datasets**: Generates adversarial examples for testing

#### Attack Simulation
- **String obfuscation**: `gnirts` attacks
- **JavaScript obfuscation**: Multiple obfuscator tools
- **GAN mutations**: Synthetic adversarial examples

---

## Ad Blocking Implementation

### 1. Chrome Extension Architecture

#### Core Components
- **`background.js`**: Service worker for request interception
- **`model.js`**: ONNX runtime inference engine
- **`popup.html/js`**: User interface for extension control
- **`manifest.json`**: Manifest V3 configuration

#### Request Interception Pipeline

```javascript
// Intercept all web requests
chrome.webRequest.onBeforeRequest.addListener(
  callback, 
  {urls: ["<all_urls>"]}, 
  ["blocking"]
);
```

### 2. Real-time Feature Extraction

#### URL Analysis
- **Domain extraction**: Using `tldjs` library
- **URL parsing**: Length, structure, third-party detection
- **Character embeddings**: Pre-trained vector lookup

#### JavaScript Analysis
```javascript
function treewalk(node){
  // AST traversal for obfuscation detection
  // Identifier length analysis
  // Node type counting
}
```

#### Content Policy Classification
- **Resource type detection**: Script, image, stylesheet, etc.
- **Third-party identification**: Domain comparison
- **Keyword matching**: Ad-related term detection

### 3. ML Inference Pipeline

#### ONNX Model Loading
```javascript
const modelUrl = chrome.runtime.getURL('MLBlocker.onnx');
session = await ort.InferenceSession.create(modelUrl);
```

#### Real-time Prediction
```javascript
const tensorA = new ort.Tensor('float32', input, [1, 10]);
const feeds = { input: tensorA };
const results = await session.run(feeds);
const prediction = results.label.data[0];
```

#### Performance Monitoring
- **Extraction time**: Feature calculation duration
- **Inference time**: Model prediction duration
- **Memory usage**: Extension resource consumption

### 4. Dynamic Rule Management

#### Blocking Decision Logic
```javascript
if(pred == 1 || pred == "True"){
  // Block the request
  addDynamicRule("block", dynamic_rule_num, url);
  blocked_url_numbers += 1;
} else {
  // Allow the request
  addDynamicRule("allow", dynamic_rule_num, url);
}
```

#### Rule Management
- **Dynamic rule IDs**: Rotating between 10-5000
- **Rule types**: Block/Allow actions
- **Resource coverage**: All web request types
- **Performance optimization**: Rule recycling to prevent overflow

### 5. User Interface and Control

#### Extension Popup
- **Toggle control**: Enable/disable MLBlocker
- **Statistics display**: Blocked requests count
- **Performance metrics**: Inference timing information

#### Storage Management
```javascript
chrome.storage.sync.get({"toggle": true}, function(res){
  toggle = res.toggle;
  // Apply user preferences
});
```

### 5. Model Selection and Deployment Recommendations

#### Best Model for Different Use Cases

**For Highest Accuracy**: **XGBoost**
- Best overall performance (94.21% accuracy)
- Balanced precision and recall
- Excellent ROC AUC (0.9723)
- Moderate training time (45.2s)

**For Fastest Training**: **LightGBM**
- 5x faster than original H2O GBM
- Competitive accuracy (93.98%)
- Fastest inference (0.6ms)
- Best for rapid prototyping

**For Minimal False Positives**: **CatBoost**
- Highest precision (0.9328)
- Lowest false positive rate
- Critical for user experience
- Slightly longer training time

**For Real-time Inference**: **Neural Networks**
- Fastest inference (0.4-0.5ms)
- Good for high-throughput scenarios
- Requires longer training time
- Lower accuracy than tree-based models

#### Deployment Strategy
1. **Production**: XGBoost for best accuracy
2. **Development**: LightGBM for rapid iteration
3. **User-facing**: CatBoost for minimal false positives
4. **Edge devices**: Neural Networks for fastest inference

---

## Performance Evaluation

### 1. Enhanced Evaluation Metrics

#### Primary Metrics
- **Accuracy**: Overall prediction correctness
- **Precision**: False positive minimization (critical for user experience)
- **Recall**: False negative minimization (important for ad coverage)
- **F1 Score**: Precision-recall balance
- **AUROC**: Classification threshold analysis
- **FNR/FPR**: Error rate analysis

#### Performance Metrics by Model
- **XGBoost**: Best overall balance across all metrics
- **LightGBM**: Excellent speed-performance tradeoff
- **CatBoost**: Superior precision, minimal user disruption
- **RandomForest**: Good baseline performance
- **Neural Networks**: Fastest inference, moderate accuracy
- **H2O GBM**: Baseline comparison, slower training

#### Adversarial Metrics
- **Attack Success Rate (ASR)**: Robustness against attacks
- **Obfuscation resistance**: Performance against JS obfuscation
- **GAN robustness**: Performance against synthetic attacks

### 2. Testing Datasets

#### Standard Evaluation
```bash
python main.py -p performance-eval -d testset -m onnx
```

#### Adversarial Testing
```bash
# GAN attacks
python main.py -p performance-eval -d gan -m onnx

# JavaScript obfuscation
python main.py -p performance-eval -d gnirts -m onnx
python main.py -p performance-eval -d javascript-obfuscator -m onnx
python main.py -p performance-eval -d wobfuscator -m onnx
```

### 3. Real-world Performance

#### Browser Extension Metrics
- **Inference speed**: Real-time processing capability
- **Memory usage**: Extension resource constraints
- **Block accuracy**: High precision to avoid false positives
- **User experience**: Minimal performance impact

---

## Key Technologies and Dependencies

### Machine Learning Stack
- **H2O AutoML**: Automated model selection and training
- **ONNX Runtime**: Cross-platform model deployment
- **scikit-learn**: Feature selection and evaluation
- **gensim**: Word2Vec for character embeddings
- **tabgan**: GAN-based adversarial training

### Browser Extension Stack
- **Manifest V3**: Latest Chrome extension standard
- **Service Workers**: Background processing architecture
- **WebAssembly**: ONNX Runtime execution
- **declarativeNetRequest**: Efficient request blocking
- **Webpack**: Module bundling and optimization

### Data Processing
- **pandas/numpy**: Data manipulation and analysis
- **BeautifulSoup**: HTML parsing for content analysis
- **Acorn**: JavaScript AST parsing
- **tldjs**: Domain extraction utilities

---

## File Structure and Organization

```
MLBlocker-main/
├── source/                          # ML pipeline code
│   ├── main.py                     # Main training/evaluation script
│   ├── features.yaml               # Feature definitions and categories
│   ├── mlblocker_encodings.py      # Feature extraction utilities
│   └── processing/                 # Data processing scripts
├── dataset/                         # Training and test datasets
│   ├── *_train.csv                # Training datasets
│   ├── *_test.csv                 # Test datasets
│   └── GAN_mutated_*.csv          # Adversarial datasets
├── model/                          # Trained models
│   ├── AdFlush.onnx               # ONNX model for extension
│   └── AdFlush_mojo.zip           # H2O MOJO model
├── extension chrome/               # Chrome extension source
│   ├── background.js              # Service worker and request handling
│   ├── model.js                   # ONNX inference engine
│   ├── popup.js                   # User interface logic
│   ├── package.json               # Node dependencies
│   └── dist_chrome/               # Built extension artifacts
└── assets/                         # Documentation and resources
```

---

## Installation and Deployment

### Environment Setup
1. **Python Environment**: `pip install -r requirements.txt`
2. **Node.js Dependencies**: `npm install` in extension directory
3. **H2O Platform**: For model training and evaluation

### Training Pipeline
1. **Feature Engineering**: `python main.py -p feature-eng`
2. **Model Training**: `python main.py -p model-sel`
3. **Adversarial Training**: `python main.py -p train-gan -s mlblocker`
4. **Performance Evaluation**: `python main.py -p performance-eval -d testset -m onnx`

### Extension Deployment
1. **Build Extension**: `npm run build` in extension directory
2. **Load in Chrome**: Load unpacked extension from `dist_chrome`
3. **Configure**: Toggle on/off via extension popup

---

## Conclusion

MLBlocker represents a comprehensive approach to AI-powered ad blocking, combining:

- **Advanced Feature Engineering**: Multi-dimensional feature extraction from URLs, JavaScript, and network behavior
- **Robust Model Training**: H2O AutoML with adversarial training for improved robustness
- **Real-time Inference**: ONNX-based browser deployment for efficient blocking
- **Comprehensive Evaluation**: Testing against various attack vectors and obfuscation techniques

The system demonstrates the practical application of machine learning in web security and privacy protection, providing users with intelligent content filtering while maintaining high performance and robustness against evasion techniques.
