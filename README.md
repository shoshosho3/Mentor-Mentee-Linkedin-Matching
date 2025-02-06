# Find My Mentor
## Overview
**Find My Mentor** is a machine learning-based system designed for mentor-mentee matchmaking using NLP techniques, machine learning models, and profile embeddings. The system matches mentees with mentors based on their profiles, preferences, and professional background. It leverages large-scale data, advanced document embeddings, and classification models for optimal matching.
## Project Details
### Features:
- **Mentor-Mentee Matching**: Uses pre-trained BERT embeddings to compute similarity between mentor and mentee profiles.
- **Multi-Model Pipeline**: Several machine learning models, including Logistic Regression, Random Forest, Support Vector Machine (SVM), Gradient Boosted Trees (GBT), Decision Tree, and Multilayer Perceptron (MLP), are tested for optimal matching.
- **Web Scraping**: Scrapes real-world mentor data from **[The Mentoring Club](https://www.mentoring-club.com)**.
- **Profile Embedding**: Generates dense vector embeddings for LinkedIn-like profile descriptions.
- **Custom Metrics**: Tailor-made metrics like cosine similarity are used to evaluate profile compatibility.
- **Embedded Data Pipeline**: Cross-join-based calculations for mentor-mentee pairs to provide probabilistic matching.

## Technical Overview
### Key Libraries
- **PySpark**: For distributed data management and computations.
- **Spark NLP**: For natural language processing and embeddings.
- **BERT Sentence Embeddings**: For transforming textual profiles into numerical representations.
- **Scikit-learn/ML Pipelines**: For end-to-end machine learning workflows.
- **Requests/BeautifulSoup**: For scraping mentor data from external sites.

### Data Pipeline and Preprocessing:
1. **Mentor/Mentee Profile Ingestion**:
    - Profile data is scraped (via website) or loaded from provided datasets.
    - Profiles include information such as `About`, `Education`, `Position`, `Experience`, `Certifications`, and `Social Links`.

2. **Natural Language Feature Extraction**:
    - Profile texts are tokenized and transformed into embeddings using **BERT Sentence Embeddings** from Spark NLP.

3. **Mentor Archetype Calculation**:
    - A mentor archetype is derived by averaging mentor embeddings to create a baseline representative embedding.

4. **Similarity Calculation**:
    - Cosine similarity metrics are calculated between mentee and mentor embeddings to measure compatibility.

5. **Data Matching**:
    - Cartesian join is performed to generate mentor-mentee combinations.
    - Feature vectors include attributes like `followers`, `posts`, `certifications`, and `experience lengths` of both parties.

6. **Classification Models**:
    - Machine learning algorithms, e.g., GBT and Logistic Regression, are trained on labeled data to classify suitable mentors.

### Machine Learning Models:
- **Logistic Regression**:
    - Simple yet effective binary classification model for predicting mentor-mentee compatibility.

- **Random Forest**:
    - Utilized for robust decision-making using ensemble learning.

- **Support Vector Machine (SVM)**:
    - Ideal for separating compatible and incompatible matches using hyperplanes.

- **Gradient Boosted Trees (GBT)**:
    - Optimized boosting model for higher prediction accuracy.

- **Multi-Layer Perceptron (MLP)**:
    - Deep learning-based approach for enhanced pattern recognition.

Models were evaluated using test datasets, with predictions ranked by the probability of suitability.
## Applications
### Mentor Recommendation System:
The project provides mentees with a ranked list of potential mentors from a large pool, helping them find the best match based on:
- Shared interests.
- Expertise alignment.
- Professional experience.
- Profile similarity.

## Results and Outputs
- **Final Matching**: The best-matching mentors are listed with details, including:
    - Mentor ID
    - Match probability (confidence of the prediction)
    - Linked profiles (URLs, if provided).

## Usage Instructions
### Running the Project:
1. Ensure the necessary dependencies are installed:
    - `PySpark`, `Spark NLP`, and ML libraries (e.g., `pandas`, `scikit-learn`).

2. **Generate Mentor Data**:
    - Run `scraping_notebook.ipynb` to fetch mentor profiles from websites.

3. **Preprocess Data**:
    - Execute `Project.ipynb` for embedding generation, preprocessing, and training/test set creation.

4. **Find Mentors for Mentees**:
    - Use `Find My Mentor.ipynb` to enter mentee profiles and get mentor recommendations.

### Required Files:
- `linkedin_people_train_data` (Parquet): Contains mentee and mentor profile details.
- `mentors.parquet`: Mentor profiles obtained via scraping or manual input.

## Dependencies
Install the following Python packages:
``` bash
pip install pyspark sparknlp pandas requests beautifulsoup4 scikit-learn
```
## Future Scope
- **Dynamic Model Updates**: Incorporate real-time feedback from matched pairs to improve the recommendation algorithm.
- **Enhanced Scraping**: Include multiple mentorship platforms for broader data diversity.
- **Exploratory Features**: Add more features like industry alignment, location preferences, etc.

## Folder Structure:
``` plaintext
|-- scraping_notebook.ipynb (Web scraping mentor data)
|-- Project.ipynb (Data preprocessing, model training pipeline)
|-- Find My Mentor.ipynb (Main application for mentor recommendations)
|-- mentors.parquet (Processed mentor profiles dataset)
|-- README.md
```
