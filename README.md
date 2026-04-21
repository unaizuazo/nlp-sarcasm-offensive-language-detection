# 🎯 NLP Sarcasm and Offensive Language Detection in Spanish Social Media

> Detecting sarcasm and offensive language in Spanish social media using BERT embeddings and machine learning  
> *Undergraduate Dissertation (TFG) in Mathematics*

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.0%2B-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/)

---

## 📋 Table of Contents
- [Overview](#-project-overview)
- [Key Results](#-key-results)
- [Quick Start](#-quick-start)
- [Methodology](#-methodology)
- [Project Structure](#-project-structure)
- [Technologies](#-technologies)
- [Contact](#-contact)

---

## 🎓 Project Overview

This project develops a multi-class classification system to detect **sarcasm** and **offensive language (nastiness)** in Spanish social media comments from the **Meneame** social news platform. The approach combines pretrained Spanish BERT embeddings with optimized MLP classifiers, achieving **~85% accuracy** on both tasks.

### 🎯 Key Contributions

✅ **Dataset**: 13,717 annotated Spanish comments with multi-label sarcasm and nastiness ratings  
✅ **Architecture**: BERT-based embeddings (768-dim) → optimized MLP classifiers  
✅ **Results**: Best accuracy 85.4% (sarcasm) and 85.8% (nastiness)  
✅ **Analysis**: Comprehensive hyperparameter optimization and network depth evaluation  

---

## 📊 Key Results

| 🎪 Task | 📈 Best Accuracy | ⚙️ Optimal Config | 🎓 Learning Rate |
|---------|------------------|------------------|------------------|
| Sarcasm (nsarc) | **85.4%** | 1 layer, 30 neurons | 0.001 |
| Nastiness | **85.8%** | 3 layers, 25 neurons | 0.01 |

### 💡 Key Findings

- 🔍 Single-layer networks sufficient (no benefit from depth)
- ⚖️ Moderate neuron counts (20-30) optimal for accuracy/efficiency tradeoff
- 🎯 Learning rate critical: very high (>0.5) or very low (<0.001) rates degrade performance
- ⚗️ SMOTE balancing essential for handling class imbalance

---

## 🚀 Quick Start

### 📦 Requirements

```bash
pip install torch transformers scikit-learn imbalanced-learn pandas numpy matplotlib seaborn
```

### ▶️ Run the Pipeline

```bash
1️⃣  Extract dataset:        notebooks/data_ext.ipynb
2️⃣  Train model:           notebooks/BERT_enhanced.ipynb (or BERT_enhanced_shuffle.ipynb)
3️⃣  View results:          notebooks/Grafs_accuracy_nsarc.ipynb & Grafs_accuracy_nast.ipynb
```

📖 See [`notebooks/README.md`](notebooks/README.md) for detailed notebook descriptions.

---

## 🔬 Methodology

### 1️⃣ Data Representation
- **Model**: `dccuchile/bert-base-spanish-wwm-cased` (Spanish BERT pretrained on Wikipedia)
- **Embedding**: 768-dimensional sentence representations via mean pooling
- **Preprocessing**: Spanish text tokenization, max length 256 tokens

### 2️⃣ Classification
- **Architecture**: Multi-layer perceptron with hyperparameter grid search
- **Configuration**: 1-3 hidden layers, 1-30 neurons per layer
- **Regularization**: Dropout + data balancing (SMOTE/RandomOverSampler)

### 3️⃣ Evaluation
- **Train/Test**: 80/20 split with random shuffling
- **Metrics**: Accuracy, precision, recall, F1-score
- **Validation**: 10% holdout from training set

---

## 📁 Project Structure

```
nlp-sarcasm-offensive-language-detection/
│
├── 📄 README.md                          # This file
├── 📚 notebooks/                         # Jupyter notebooks with full pipeline
│   ├── 📖 README.md                      # Detailed notebook guide
│   ├── 🔄 data_ext.ipynb                # Dataset extraction & preprocessing
│   ├── 🔄 data_ext_test.ipynb           # Test set extraction
│   ├── 🤖 BERT.ipynb                    # Initial BERT exploration
│   ├── 🚀 BERT_enhanced.ipynb           # Full BERT pipeline (sarcasm)
│   ├── 🚀 BERT_enhanced_shuffle.ipynb   # Full pipeline (nastiness)
│   ├── 🎯 BERT_whole.ipynb              # Embeddings for entire dataset
│   ├── 📊 Grafs_accuracy_nsarc.ipynb    # Sarcasm detection analysis
│   └── 📊 Grafs_accuracy_nast.ipynb     # Nastiness detection analysis
├── 📋 docs/                              # Additional documentation
├── 🎨 graphs/                            # Result visualizations
│
└── 📦 data/                              # Data directory (relative paths ./data/)
    └── (dataset files - not included)
```

---

## 💻 Technologies

| Technology | Purpose | Badge |
|-----------|---------|-------|
| **Python** | Programming language | ![Python](https://img.shields.io/badge/3.9%2B-blue?logo=python) |
| **PyTorch** | Deep learning framework | ![PyTorch](https://img.shields.io/badge/1.9%2B-red?logo=pytorch) |
| **Transformers** | BERT models & utilities | ![HuggingFace](https://img.shields.io/badge/4.0%2B-yellow?logo=huggingface) |
| **Scikit-learn** | ML classifiers & metrics | ![Scikit](https://img.shields.io/badge/0.24%2B-orange) |
| **Pandas/NumPy** | Data manipulation | ![Data](https://img.shields.io/badge/Data-Processing-blue) |
| **Matplotlib/Seaborn** | Data visualization | ![Viz](https://img.shields.io/badge/Data-Visualization-green) |

### 🔗 Additional Libraries
- **Imbalanced-learn**: SMOTE and oversampling for class balancing
- **Jupyter**: Interactive notebook environment

---

## 📚 Dataset Details

### 🎯 Meneame Corpus
- **Size**: 13,717 Spanish comments with human annotations
- **Source**: Meneame social news aggregator
- **Language**: Spanish
- **Format**: Multi-label annotations

### 🏷️ Annotation Schema
- **nsarc**: Count of annotators identifying sarcasm (0-5+)
- **nastiness**: Average offensive content rating (0.0-1.0)

### 📊 Dataset Split
- **Training**: 10,973 samples (80%)
- **Testing**: 2,744 samples (20%)

---

## ⚡ Computational Requirements

| Resource | Requirement | Notes |
|----------|------------|-------|
| **GPU** | Optional | 3-5x speedup for BERT embedding |
| **RAM** | 8GB minimum | 16GB recommended for batch processing |
| **Storage** | ~2GB | Dataset + model checkpoints |
| **Runtime** | 2-3 hours | Full pipeline |

---

## ⚠️ Important Notes

- 🔐 **Dataset not included**: Original Meneame corpus requires separate download
- 📍 **Path configuration**: Notebooks use relative paths (`./data/`) - adjust to your setup
- ⏱️ **Runtime**: Full BERT embedding extraction takes ~1-2 hours on 13K+ comments
- 🎮 **GPU recommended**: CPU processing feasible but significantly slower

---

## 🔮 Future Work

- 🎯 Fine-tuning BERT on target task (transfer learning)
- 🔗 Ensemble methods with multiple classifiers
- 🌍 Cross-lingual transfer from English sarcasm datasets
- 🧠 Advanced architectures (Transformers, attention mechanisms)
- 📌 Multi-task learning for joint prediction

---

## 📚 References

- **Meneame Dataset**: Social news aggregator corpus
- **BERT**: Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
- **Spanish BERT**: DCCUChile model - pretrained on Wikipedia and crawled Spanish text

---

## 📄 License

This project is part of an undergraduate thesis. Please refer to the original Meneame dataset terms of use for data usage restrictions.

---

## 👤 About the Author

**[YOUR NAME]**

📧 **Email**: [your.email@example.com](mailto:your.email@example.com)  
🔗 **LinkedIn**: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)  
💼 **Portfolio**: [your-portfolio-link.com](https://your-portfolio-link.com)  
🐙 **GitHub**: [github.com/yourprofile](https://github.com/yourprofile)

---

## 💬 Questions or Collaboration?

Found this project interesting? Want to collaborate or have questions?

- ✉️ **Email me** at [your.email@example.com](mailto:your.email@example.com)
- 💼 **Connect on LinkedIn** at [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
- ⭐ **Star this repository** if you found it helpful!

---

<div align="center">

**Made with ❤️ for NLP and Spanish Language Processing**

*Last Updated: April 2026*  
*Project Type: Undergraduate Dissertation (TFG) in Mathematics*

</div>
