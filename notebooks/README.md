# 📓 Notebooks Guide

Complete Jupyter notebook implementations for sarcasm and offensive language detection in Spanish social media.

---

## 📋 Quick Navigation

| 📂 Category | 🔗 Notebooks | 📊 Purpose |
|-----------|-------------|-----------|
| **Data Extraction** | data_ext, data_ext_test | Load & preprocess dataset |
| **BERT Models** | BERT, BERT_enhanced, BERT_enhanced_shuffle, BERT_whole | Generate embeddings & train |
| **Analysis** | Grafs_accuracy_nsarc, Grafs_accuracy_nast | Visualize results |

---

## 🔄 Data Extraction (Prerequisites)

### 1️⃣ **data_ext.ipynb** - Main Dataset Extraction
```
⏱️ Runtime: ~1-2 hours  |  📈 Processes: 68,585+ entries
```

**🎯 Purpose**: Extract and preprocess the Meneame dataset

**📥 What it does**:
- Loads main dataset from JSON (`./data/Meneame/Corpus/polls-F1234_M5.json`)
- Parses annotated comments with sarcasm counts and nastiness ratings
- Matches annotations with parsed text from corpus
- Converts data to Excel format for downstream processing
- Outputs: `./data/datos.xlsx` (13,717 comments with labels)

**📤 Key outputs**:
- CSV/Excel file with columns: `id`, `text`, `nastiness`, `nsarc`, `parsed`
- **nsarc**: Number of annotators rating as sarcastic
- **nastiness**: Average offensive content score (0.0-1.0)

**✨ Status**: Essential - Run this first

---

### 2️⃣ **data_ext_test.ipynb** - Test Set Extraction
```
⏱️ Runtime: ~5-10 minutes
```

**🎯 Purpose**: Extract test set from a different batch

**📥 What it does**:
- Loads test batch from JSON (`./data/Meneame/Corpus/batch-FASE4.json`)
- Extracts `id`, `text`, and `context` fields
- Saves to Excel for validation/testing
- Outputs: `./data/datos_test.xlsx`

**💡 Use case**: Validation on held-out test data from different annotation batch

**✨ Status**: Optional - Useful for independent validation

---

## 🤖 BERT Embedding Models

### 3️⃣ **BERT.ipynb** - Initial Exploration
```
📊 Status: Exploratory  |  🎓 Learning: BERT workflow
```

**🎯 Purpose**: Initial exploration and testing of BERT for individual comments

**📥 What it does**:
- Loads pretrained Spanish BERT (`dccuchile/bert-base-spanish-wwm-cased`)
- Tests tokenization and embedding extraction on sample texts
- Demonstrates how to get 768-dimensional sentence embeddings
- Builds a simple 2-layer MLP classifier on top
- Tests inference on single comments

**🎓 Best for**: Understanding BERT workflow and debugging

**💡 Key learning**:
- BERT tokenization produces 512 context vectors per comment
- Mean pooling creates sentence-level embeddings
- Simple architectures sufficient for classification

**✨ Status**: Exploratory, not used for final results

---

### 4️⃣ **BERT_enhanced.ipynb** - Primary Model ⭐ **STAR THIS**
```
⏱️ Runtime: ~45-60 minutes  |  📈 Best accuracy: 85.4%  |  🎪 Task: Sarcasm (nsarc)
```

**🎯 Purpose**: Complete BERT pipeline for sarcasm detection (nsarc)

**📥 What it does**:
1. Loads dataset and performs 80/20 train/test split
2. Tokenizes all comments to fixed length (256 tokens)
3. Generates BERT embeddings for entire training set (~10K comments)
   - Processes in batches to manage memory
   - Extracts last hidden state and applies mean pooling
   - Output: 768-dimensional embedding per comment
4. Trains MLP classifiers with hyperparameter grid search:
   - Hidden layers: 1, 2, 3
   - Neurons per layer: 1, 5, 10, 15, 20, 25, 30
   - Learning rates: 0.001 to 1.0
5. Evaluates on validation set (10% of training)
6. Reports accuracy, F1-score, confusion matrices

**📊 Key results**:
- **Best accuracy: 85.4%** (1 layer, 30 neurons, lr=0.001)
- Single layer networks outperform deeper architectures
- Diminishing returns beyond 25 neurons
- 6-37 seconds training time per model

**📤 Main outputs**:
- Trained MLP classifiers
- Accuracy curves vs neuron count
- Performance metrics

**✨ Status**: PRIMARY MODEL - Essential for results

---

### 5️⃣ **BERT_enhanced_shuffle.ipynb** - Nastiness Detection
```
⏱️ Runtime: ~45-60 minutes  |  📈 Best accuracy: 85.8%  |  🎪 Task: Nastiness
```

**🎯 Purpose**: Alternative pipeline with shuffled data for nastiness detection

**📥 What it does**:
- Same as BERT_enhanced but with important differences:
  - Shuffles dataset before splitting (`data.sample(frac=1)`)
  - Uses **nastiness labels** instead of sarcasm (nsarc)
  - Trains on prediction task: is comment nasty/offensive?
  - Applies SMOTE for data balancing
- Hyperparameter optimization same as enhanced version
- Generates accuracy vs architecture plots

**📊 Key results**:
- **Best accuracy: 85.8%** (3 layers, 25 neurons, lr=0.01)
- Deeper networks slightly better for nastiness than sarcasm
- Training time: 2-37 seconds per model

**💡 Use case**: Demonstrates both detection tasks work well with BERT

**✨ Status**: ALTERNATIVE MODEL - Good for comparison

---

### 6️⃣ **BERT_whole.ipynb** - Full Dataset Embeddings
```
⏱️ Runtime: ~1.5-2 hours  |  📦 Output: Tensor files  |  📊 Coverage: ALL 13,717 comments
```

**🎯 Purpose**: Generate embeddings for entire dataset (full 13,717 comments)

**📥 What it does**:
1. Loads all comments (no train/test split)
2. Tokenizes all 13,717 comments
3. Processes through BERT in batches
4. Generates complete embedding matrix:
   - Shape: [13717, 768]
5. Saves both embeddings and labels to PyTorch tensor files:
   - `mean_tensor.pt`: Sentence embeddings
   - `nsarc_tensor.pt`: Sarcasm labels
   - `nast_tensor.pt`: Nastiness labels

**💡 Purpose**: Useful for:
- Creating global embedding space for visualization
- Transfer learning to other Spanish NLP tasks
- Ensemble methods using full dataset

**📤 Key outputs**:
- PyTorch tensor files with full embeddings
- Can be reused without re-running BERT

**✨ Status**: SUPPORTING - Useful for advanced analysis

---

## 📊 Results & Visualization

### 7️⃣ **Grafs_accuracy_nsarc.ipynb** - Sarcasm Analysis
```
📊 Focus: Sarcasm detection (nsarc)  |  🎨 Output: PNG graphs
```

**🎯 Purpose**: Analyze and visualize sarcasm detection performance

**📥 What it does**:
1. Loads pretrained embeddings and sarcasm labels
2. Tests MLP architectures varying:
   - Number of hidden layers: 1, 2, 3
   - Neurons per layer: 1, 5, 10, 15, 20, 25, 30
3. Applies data balancing (RandomOverSampler)
4. Tests learning rate effects: 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0
5. Generates plots:
   - **Accuracy vs neurons** (by layer count)
   - **Training time vs neurons**
   - **Accuracy vs learning rate**

**📊 Key visualizations**:
- 1 layer networks plateau at 85%+ accuracy
- 2-3 layers show minimal improvement
- Optimal: 25-30 neurons, 1 layer
- Learning rates 0.001-0.01 best
- Training time: 5-37 seconds depending on architecture

**📤 Output format**: High-resolution PNG graphs

**✨ Status**: RESULTS ANALYSIS - Best practice demonstration

---

### 8️⃣ **Grafs_accuracy_nast.ipynb** - Nastiness Analysis
```
📊 Focus: Nastiness detection  |  🎨 Output: PNG graphs
```

**🎯 Purpose**: Analyze and visualize nastiness detection performance

**📥 What it does**:
- Same analysis as Grafs_accuracy_nsarc but for **nastiness labels**
- Generates equivalent set of plots for offensive language detection

**🔄 Key difference from sarcasm**:
- 3-layer networks perform slightly better (85.8% vs 85.4%)
- Requires different learning rates (0.01 better than 0.001)
- More stable with increased depth

**📤 Output format**: PNG graphs with comparative analysis

**✨ Status**: RESULTS ANALYSIS - Complete comparison

---

## 🚀 Workflow Recommendations

### Option 1️⃣: Minimal Pipeline (Quick)
```
1. data_ext.ipynb                 [~1-2 hours - Extract dataset]
2. BERT_enhanced.ipynb            [~45-60 min - Train sarcasm model]
3. Grafs_accuracy_nsarc.ipynb     [~5 min - Analyze results]
```
**⏱️ Total Time**: ~2.5-3 hours | **📊 Output**: Sarcasm detection results

---

### Option 2️⃣: Complete Analysis (Recommended)
```
1. data_ext.ipynb                    [Extract main dataset]
2. BERT_enhanced.ipynb               [Train sarcasm detection]
3. BERT_enhanced_shuffle.ipynb       [Train nastiness detection]
4. Grafs_accuracy_nsarc.ipynb        [Analyze sarcasm results]
5. Grafs_accuracy_nast.ipynb         [Analyze nastiness results]
```
**⏱️ Total Time**: ~3-4 hours | **📊 Output**: Full comparative analysis

---

### Option 3️⃣: Advanced Usage (Data Prep Only)
```
1. data_ext.ipynb
2. data_ext_test.ipynb
3. BERT_whole.ipynb               [Generate embeddings for full dataset]
```
Then use embeddings for custom models or transfer learning.
**💡 Use case**: Custom models, ensemble methods, visualization

---

## 🔧 Technical Details

### 📊 Data Flow
```
Raw JSON 
   ↓
data_ext.ipynb (Excel file)
   ↓
BERT embedding extraction
   ↓
MLP classification training
   ↓
Results visualization
```

### 🧠 Embedding Process
1. **Tokenization**: Convert text to token IDs (length 256, padded/truncated)
2. **BERT**: Generate 768-dim token embeddings using Spanish BERT
3. **Pooling**: Average across all tokens to get sentence embedding
4. **Normalization**: Optional - data already normalized by BERT

### 🤖 Classification Details
- **Classifier**: Scikit-learn MLPClassifier
- **Training**: Adam optimizer with L2 regularization
- **Validation**: 10% holdout cross-validation
- **Balancing**: SMOTE or RandomOverSampler to handle class imbalance

### 🎯 Hyperparameter Grid
```
Layers:     [1, 2, 3]
Neurons:    [1, 5, 10, 15, 20, 25, 30] per layer
Learning:   [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
Epochs:     1000 (with early stopping)
```

---

## 📦 File Dependencies

| 📓 Notebook | 📥 Requires | 📤 Outputs |
|----------|----------|---------|
| data_ext.ipynb | JSON dataset | `./data/datos.xlsx` |
| data_ext_test.ipynb | JSON test batch | `./data/datos_test.xlsx` |
| BERT.ipynb | - | Demos only |
| BERT_enhanced.ipynb | `./data/datos.xlsx` | Trained models, metrics |
| BERT_enhanced_shuffle.ipynb | `./data/datos.xlsx` | Trained models, metrics |
| BERT_whole.ipynb | `./data/datos.xlsx` | `*.pt` embedding files |
| Grafs_accuracy_nsarc.ipynb | Embedding files | PNG graphs |
| Grafs_accuracy_nast.ipynb | Embedding files | PNG graphs |

---

## ⚠️ Important Notes

### 🔐 Dataset Location
- All notebooks use relative path `./data/`
- Adjust to your Meneame dataset location
- Download from Meneame repository separately

### 💻 Computational Requirements
- **GPU**: Not required but recommended for BERT embedding (3-5x speedup)
- **RAM**: 8GB minimum, 16GB recommended (for batch processing)
- **Runtime**: 
  - Full pipeline: ~2-3 hours
  - BERT embedding alone: ~1-2 hours
  - MLP training: 5-60 seconds per model

### ⚡ Known Limitations
- 🇪🇸 Spanish-specific: Model trained only on Spanish text
- 📱 Domain-specific: Best performance on social media comments
- 📝 Comment length: Optimized for typical comment length (256 tokens)
- 🎲 Batch effects: Results may vary slightly between runs (randomness in initialization)

---

## 🐛 Troubleshooting

### ❌ Out of Memory Error during BERT
```
Solution:
- Reduce batch size in BERT processing loops (line with batch_size=32)
- Process fewer comments at a time
- Use GPU if available
```

### ❌ Very Low Accuracy
```
Solution:
- Check data loading - ensure Excel file has correct columns
- Verify SMOTE is applied (handles imbalanced data)
- Check learning rate - too high or too low causes poor convergence
```

### ❌ Slow BERT Processing
```
Solution:
- Consider using GPU (install CUDA+cuDNN)
- Reduce sequence length if longer tokens aren't needed
- Process in smaller batches with more iterations
```

---

## 📚 Citation

If you use these notebooks, please cite:
```
Unai Zuazo (2024). "Sarcasm and offensive language detection in social media using machine learning techniques" Undergraduate Dissertation in Mathematics. Universidad del País Vasco UPV/EHU
```

---

## 👤 Author

**Unai Zuazo**

📧 Email: [unaizuazoangulo@gmail.com](mailto:unaizuazoangulo@gmail.com)  
🔗 LinkedIn: [linkedin.com/in/unai-zuazo/](https://www.linkedin.com/in/unai-zuazo/)

---

<div align="center">

**Happy analyzing! 🚀**

*Last Updated: April 2026*

</div>
