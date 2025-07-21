# Cyberbullying Detection Using Explainable AI

## Overview

Cyberbullying is a growing problem on social media, affecting an estimated 37% of young people worldwide. Traditional detection methods struggle with emerging challenges such as multilingual slang, emojis, and disguised toxicity. This project proposes a hybrid approach that combines state-of-the-art machine learning models with Explainable AI (XAI) techniques to build a transparent, robust, and multilingual cyberbullying detection system.

## Objectives

- Develop a cyberbullying detection system with **at least 90% accuracy** on a diverse dataset containing over 167,000 examples
- Incorporate **Explainable AI methods** (SHAP, LIME, attention visualization) to provide transparency and trust
- Support **multilingual and emoji-aware** text processing to better handle real-world social media language
- Implement **session-based detection** to consider conversational context
- Enable **real-time moderation** with fast predictions and actionable insights

## Dataset

The training data combines multiple sources:

- Jigsaw Toxic Comment Classification Dataset (~160,000 samples)
- Emojis Dataset (~5,000 emoji-rich tweets)
- Synthetic examples based on CONAN (~10,000 samples)
- Indian Cyberbullying and other multilingual cyberbullying datasets

The dataset consists of both single-message and session-based conversational examples.

## Approach

- Use transformer-based models (e.g., XLM-RoBERTa) for strong multilingual text classification
- Complement with traditional machine learning models like Logistic Regression and SVM
- Apply Explainable AI techniques to highlight key toxic words and phrases that influence predictions
- Build a Streamlit-based user interface to interact with the detection system, visualize explanations, and support moderation workflows

## Features

- **Multilingual input support** - Detect cyberbullying across different languages
- **Real-time toxicity detection** - Instant analysis of text input
- **Conversation thread visualization** - Context-aware session analysis
- **Downloadable PDF reports** - Generate detailed moderation reports
- **Toxic comment flagging** - Automated content moderation
- **Interactive controls** - Adjustable toxicity thresholds and customizable themes

## Project Structure

```
cyberbullying-detection/
├── app.py                    # Main Streamlit application
├── pipeline/                 # Text processing and model inference
├── models/                   # Pretrained and trained model files
├── components/               # Modular UI components
├── explainability/           # SHAP, LIME, and attention visualization
├── data/                     # Raw and processed datasets
├── requirements.txt          # Python dependencies
└── README.md                # Project documentation
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/cyberbullying-detection.git
   cd cyberbullying-detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Single Text Analysis:**
   - Enter text in the input field
   - View toxicity predictions and confidence scores
   - Explore explainable AI visualizations

2. **Batch Processing:**
   - Upload CSV files for bulk analysis
   - Download processed results with explanations

3. **Session Analysis:**
   - Analyze conversation threads for context-aware detection
   - Track toxicity patterns across message exchanges

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|----------|
| XLM-RoBERTa | 92.3% | 91.8% | 90.2% | 91.0% |
| Logistic Regression | 87.5% | 86.2% | 85.9% | 86.0% |
| SVM | 85.8% | 84.7% | 83.2% | 83.9% |

## Explainability Features

- **SHAP Analysis:** Token-level importance scores for model predictions
- **LIME Explanations:** Local interpretable model-agnostic explanations
- **Attention Visualization:** Transformer attention weight visualization
- **Feature Importance:** Traditional ML model feature analysis

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## References

- Mahmud et al., 2023 — Cyberbullying prevalence study
- Philipo et al., 2024 — Transformer model performance
- Yi and Zubiaga, 2022 — Session-based detection research
- Maity et al., 2024 — Sentiment and explainability methods
- El Koshiry et al., 2024 — Advances in explainable models

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

If you encounter any issues or have questions, please:

1. Check the [Issues](https://github.com/your-username/cyberbullying-detection/issues) page
2. Create a new issue with detailed information
3. Contact the development team

---

**Thank you for exploring this cyberbullying detection system. We welcome feedback and contributions to improve its safety and effectiveness.**