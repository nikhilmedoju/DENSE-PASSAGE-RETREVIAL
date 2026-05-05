# Dense Passage Retrieval (DPR)

## 📌 Project Overview
This project implements **Dense Passage Retrieval (DPR)** for Open-Domain Question Answering. It utilizes a pre-trained language model (`sentence-transformers/nli-distilroberta-base-v2`) to convert textual questions into dense vector embeddings. By computing the inner dot product similarity between question embeddings, the system is able to match user queries to the most relevant questions in a pre-computed corpus and accurately retrieve the corresponding answers.

The project achieves an impressive **94.5% accuracy** on the **WebQuestions Dataset**.

## 🚀 Features
- **Semantic Search**: Understands the semantic meaning of questions using advanced NLP vector embeddings.
- **Pre-trained Transformers**: Leverages HuggingFace's `SentenceTransformer` specifically `nli-distilroberta-base-v2`.
- **Custom Similarity Metric**: Implements inner dot product for highly efficient vector similarity search.
- **Metrics & Evaluation**: Includes accuracy scoring using `scikit-learn` to validate retrieval effectiveness.

## 🛠️ Tools & Technologies Used
- **Python**: Core programming language.
- **NLP & Deep Learning**: `sentence-transformers` (BERT/RoBERTa) for vector embeddings.
- **Data Manipulation**: `pandas`, `numpy` for data extraction, transformation, and vector operations.
- **Machine Learning Metrics**: `scikit-learn` for performance evaluation.
- **Environment**: Jupyter Notebook (`.ipynb`).

## 📁 Project Structure
```
Dense-Passage-Retrieval/
│
├── WebQuestionDataset/     # Directory containing the training and testing datasets
│   ├── train.csv           # WebQuestions training data (URLs, questions, answers)
│   └── test.csv            # WebQuestions testing data
│
├── model/                  # Automatically generated directory to cache numpy embeddings
│   ├── X_train.npy         # Cached training vector embeddings
│   ├── y_train.npy         # Cached training answers
│   ├── X_test.npy          # Cached testing vector embeddings
│   └── y_test.npy          # Cached testing answers
│
├── testQuestion.csv        # Custom dataset for sample inference
├── DensePassage.ipynb      # Main Jupyter Notebook containing the DPR pipeline
└── README.md               # Project documentation
```

## ⚙️ Installation

1. **Clone the repository** (if applicable):
   ```bash
   git clone <your-repository-url>
   cd Dense-Passage-Retrieval
   ```

2. **Create a Virtual Environment** (Recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   You need to install the required Python packages. You can install them using `pip`:
   ```bash
   pip install pandas numpy scikit-learn matplotlib sentence-transformers
   ```

## 💻 Usage

1. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Open `DensePassage.ipynb`**:
   - Navigate to `DensePassage.ipynb` in the Jupyter interface.
   
3. **Run the Cells**:
   - The notebook is designed to be run sequentially. 
   - **Step 1:** It will load the datasets (`train.csv` and `test.csv`).
   - **Step 2:** It will encode the questions into dense vectors. *(Note: The first run might take time as it downloads the `nli-distilroberta-base-v2` model and creates `.npy` cache files).*
   - **Step 3:** It calculates inner dot product similarities.
   - **Step 4:** It calculates the model accuracy (94.5%).
   - **Step 5:** You can test custom questions by adding them to `testQuestion.csv` and running the final cell to view the predicted answers.

## 📊 Results & Output
The model accurately predicts answers to general knowledge questions. 

**Example Inference Output:**
```
Question : where did barack obama attend school?
Predicted Answer : ['Occidental College' 'Harvard Law School' 'Noelani Elementary School' 'Punahou School' 'State Elementary School Menteng 01' 'St. Francis of Assisi Catholic School' 'Columbia University']

Question : what language does egyptian people speak?
Predicted Answer : ['Languages of Egypt' 'Egyptian Arabic' 'Coptic Language' 'Egyptian language' "Sa'idi Arabic"]
```

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.
