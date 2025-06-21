# Comparative Analysis of Deep Learning Models for Web Traffic Time-Series Forecasting

A comprehensive comparison of six deep learning architectures (RNN, LSTM, GRU, and their bidirectional variants) for predicting Wikipedia web traffic using time-series analysis.

## 📄 Project Paper

You can read the full project paper [here](https://drive.google.com/file/d/1DPxv_dR2tCsf7uiuSYZmElaLtxAvnyrA/view?usp=sharing).

## 📊 Research Overview

This study evaluates the performance of different recurrent neural network architectures for web traffic forecasting using real-world Wikipedia page view data. The research demonstrates that **Bidirectional LSTM (BiLSTM)** achieves superior performance with an R² score of **0.93**, making it the most effective model for capturing long-term temporal dependencies in web traffic patterns.

## 🎯 Key Findings

| Model | MAE | R² Score | MSE | MSLE |
|-------|-----|----------|-----|------|
| **BiLSTM** | **0.0163** | **0.9309** | **0.0018** | **0.0007** |
| RNN | 0.0122 | 0.7828 | 0.0023 | 0.0008 |
| GRU | 0.0402 | 0.7491 | 0.0041 | 0.0025 |
| BiGRU | 0.0410 | 0.7390 | 0.0043 | 0.0025 |
| BiRNN | 0.0260 | 0.7020 | 0.0046 | 0.0020 |
| LSTM | 0.0458 | 0.3291 | 0.0097 | 0.0043 |

## 🚀 Features

- **Comprehensive Model Comparison**: Six different RNN architectures tested under identical conditions
- **Real-world Dataset**: Wikipedia web traffic data from Kaggle (145K+ articles, July 2015 - Dec 2016)
- **Multiple Evaluation Metrics**: MAE, MSE, MSLE, and R² score for thorough performance assessment
- **Bidirectional Processing**: Enhanced temporal pattern recognition through forward and backward sequence processing
- **GPU Acceleration**: Optimized training using NVIDIA Tesla P100 GPU



## 🔬 Methodology

### Data Preprocessing
- **Missing Value Handling**: Forward fill (ffill) method
- **Normalization**: Min-Max scaling to [0,1] range
- **Sliding Window**: Fixed-length sequences for temporal pattern learning
- **Train-Test Split**: Chronological 80-20 split

### Model Architecture
All models follow a consistent architecture:
- 2 recurrent layers (hidden_size=32, dropout=0.2)
- Fully connected layers (64 units → output)
- ReLU activation with dropout regularization

### Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Mean Squared Error (MSE)
- **Epochs**: 500
- **Hardware**: NVIDIA Tesla P100 GPU

## 📈 Results Analysis

### Key Insights
1. **BiLSTM Superiority**: Achieved the highest R² score (0.9309) and lowest MSE/MSLE
2. **Bidirectional Advantage**: All bidirectional variants outperformed their unidirectional counterparts
3. **RNN Efficiency**: Simple RNN showed competitive MAE performance despite architectural simplicity
4. **LSTM Underperformance**: Unexpected poor performance possibly due to hyperparameter sensitivity

### Performance Visualization
The repository includes comprehensive visualizations:
- Model comparison charts
- Prediction vs. actual value plots
- Training loss curves
- Error distribution analysis

## 🔮 Future Work

- [ ] Hyperparameter optimization for individual models
- [ ] Attention mechanism integration
- [ ] Transformer-based architectures exploration
- [ ] Multivariate forecasting with exogenous features
- [ ] Real-time prediction system development

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@article{geddam2024web,
  title={Comparative Analysis of Deep Learning Models for Web Traffic Time-Series Forecasting},
  author={Geddam, Poorvik Shrinil},
  journal={VIT-AP University},
  year={2025}
}
```


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Kaggle for providing the Web Traffic Time Series Forecasting dataset.
- PyTorch team for the deep learning framework.

---

⭐ **Star this repository if you found it helpful!**
