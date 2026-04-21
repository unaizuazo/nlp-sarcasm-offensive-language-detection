# NLP Sarcasm and Offensive Language Detection in Spanish Social Media

Detection of sarcasm and offensive language in Spanish social media comments using BERT embeddings and machine learning. Undergraduate Dissertation (TFG) in Mathematics.

## Project Overview

This project develops a multi-class classification system to detect **sarcasm** and **offensive language (nastiness)** in Spanish social media comments from the **Meneame** social news platform. The approach combines pretrained Spanish BERT embeddings with optimized MLP classifiers, achieving ~85% accuracy on both tasks.

## Key Contributions

- **Dataset**: 13,717 annotated Spanish comments with multi-label sarcasm and nastiness ratings
- **Architecture**: BERT-based embeddings (768-dim) → MLP classifiers with hyperparameter optimization
- **Results**: Best accuracy 85.4% (sarcasm) and 85.8% (nastiness) with optimized network architectures
- **Analysis**: Comprehensive evaluation of network depth (1-3 layers) and neuron count effects on performance

## Quick Start

### Requirements
```bash
pip install torch transformers scikit-learn imbalanced-learn pandas numpy matplotlib seaborn
```

### Run the Pipeline
1. Extract dataset: `notebooks/data_ext.ipynb`
2. Train model: `notebooks/BERT_enhanced.ipynb` or `BERT_enhanced_shuffle.ipynb`
3. View results: `notebooks/Grafs_accuracy_nsarc.ipynb` and `Grafs_accuracy_nast.ipynb`

See [`notebooks/README.md`](notebooks/README.md) for detailed notebook descriptions.

## Methodology

### 1. Data Representation
- **Model**: `dccuchile/bert-base-spanish-wwm-cased` (Spanish BERT pretrained on Wikipedia)
- **Embedding**: 768-dimensional sentence representations via mean pooling
- **Preprocessing**: Spanish text tokenization, max length 256 tokens

### 2. Classification
- **Architecture**: Multi-layer perceptron with hyperparameter grid search
- **Configuration**: 1-3 hidden layers, 1-30 neurons per layer
- **Regularization**: Dropout + data balancing (SMOTE/RandomOverSampler)

### 3. Evaluation
- **Train/Test**: 80/20 split with random shuffling
- **Metrics**: Accuracy, precision, recall, F1-score
- **Validation**: 10% holdout from training set

## Results Summary

| Task | Best Accuracy | Optimal Config | Learning Rate |
|------|---------------|----------------|---|
| Sarcasm (nsarc) | 85.4% | 1 layer, 30 neurons | 0.001 |
| Nastiness | 85.8% | 3 layers, 25 neurons | 0.01 |

**Key Findings**:
- Single-layer networks sufficient for this task (no benefit from depth)
- Moderate neuron counts (20-30) optimal for accuracy/efficiency tradeoff
- Learning rate critical: very high (>0.5) or very low (<0.001) rates degrade performance
- SMOTE balancing essential for handling class imbalance

## Project Structure

```
nlp-sarcasm-offensive-language-detection/
├── README.md                           # Main project documentation
├── notebooks/                          # Jupyter notebooks with full pipeline
│   ├── README.md                       # Detailed notebook guide
│   ├── data_ext.ipynb                 # Dataset extraction and preprocessing
│   ├── data_ext_test.ipynb            # Test set extraction
│   ├── BERT.ipynb                     # Initial BERT exploration
│   ├── BERT_enhanced.ipynb            # Full BERT pipeline (sarcasm)
│   ├── BERT_enhanced_shuffle.ipynb    # Full pipeline (nastiness)
│   ├── BERT_whole.ipynb               # Embeddings for entire dataset
│   ├── Grafs_accuracy_nsarc.ipynb     # Sarcasm detection analysis
│   └── Grafs_accuracy_nast.ipynb      # Nastiness detection analysis
├── docs/                               # Additional documentation
├── graphs/                             # Result visualizations
└── data/                               # Data directory (paths use relative ./data/)
```

## Technologies

- **Python 3.9+**
- **PyTorch**: Deep learning framework and tensor operations
- **Transformers**: Hugging Face library for BERT models
- **Scikit-learn**: ML classifiers, metrics, cross-validation
- **Imbalanced-learn**: SMOTE and oversampling for class balancing
- **Pandas/NumPy**: Data manipulation and numerical computing
- **Matplotlib/Seaborn**: Data visualization and result plotting

## Dataset Details

**Meneame Corpus**
- 13,717 Spanish comments with human annotations
- Multi-label format:
  - **nsarc**: Count of annotators identifying sarcasm (0-5+)
  - **nastiness**: Average offensive content rating (0.0-1.0)
- Split: 10,973 training + 2,744 test samples (80/20)

## Important Notes

- **Dataset not included**: Original Meneame corpus requires separate download due to licensing
- **Path configuration**: Notebooks use relative paths (`./data/`) - adjust to your setup
- **Runtime**: Full BERT embedding extraction takes ~1-2 hours on 13K+ comments
- **GPU recommended**: CPU processing is feasible but significantly slower

## Future Work

- Fine-tuning BERT on target task (transfer learning)
- Ensemble methods with multiple classifiers
- Cross-lingual transfer from English sarcasm datasets
- Advanced architectures (Transformers, attention mechanisms)
- Multi-task learning for joint prediction

## References

- Meneame Dataset: Social news aggregator corpus
- BERT: Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
- Spanish BERT: DCCUChile model - pretrained on Wikipedia and crawled Spanish text

## License

This project is part of an undergraduate thesis. Please refer to the original Meneame dataset terms of use for data usage restrictions.

---

**Project Type**: Undergraduate Dissertation (TFG) in Mathematics
**Academic Year**: 2023-2024
**Last Updated**: April 2026