# Emotion-Infused Collaborative Filtering for Personalized Airbnb Recommendations

This project builds an emotion aware Airbnb recommendation and price prediction system that combines traditional listing features with deep semantic signals extracted from guest reviews. By generating transformer based embeddings and using FAISS for vector similarity search, the system recommends listings that align with the emotional intent expressed in a user’s natural language query. A companion XGBoost model predicts fair nightly prices using both structural and emotion derived features, allowing users to see whether a recommended listing is underpriced, fairly priced or overpriced. The final solution is deployed through a Streamlit interface that supports intuitive, emotion guided exploration of Airbnb listings.

## Official Data Source

Inside Airbnb San Francisco Data:
```
https://insideairbnb.com/san-francisco/
```

## Install dependencies:

```bash
pip install -r requirements.txt

```

## Dependencies
The environment requires:

      Python 3.8 or higher
      numpy
      matplotlib
      seaborn

NLP and Text Processing:

      nltk
      re
      sentence-transformers
      transformers
      scikit-learn

Sentiment and Emotion Analysis: 

      vaderSentiment
      j-hartmann/emotion-english-distilroberta-base (via Hugging Face transformers)

Machine Learning Models:

      xgboost
      lightgbm (optional)
      scikit-learn

Vector Search:

      faiss-cpu

File Handling:

      pyarrow
      fastparquet

Streamlit Application:

      streamlit

Google Colab and Drive Integration:

      google-colab
      gdown

## EDA_and_Reviews_Processing.ipynb

This notebook prepares the raw Airbnb data and establishes the emotional foundation of the entire project. It cleans listings and reviews, preprocesses review text, computes VADER sentiment scores, classifies emotions with DistilRoBERTa, extracts emotional tone from listing descriptions, and aggregates these signals for each property. These emotion enriched features allow the model to capture patterns in guest experience, from positive impressions to negative sentiment clusters that influence price. Run the notebook from start to finish to generate the sentiment ready parquet files used in later stages.

## Reviews_Embedding_Generator.ipynb

This notebook converts the language of guest reviews into semantic meaning. Using a SentenceTransformer model, it creates embedding for each listing and builds a FAISS index for fast similarity search. Run the notebook in sequence to generate the embeddings and index files.

## Reviews_Listing_Processing.ipynb

All emotional and sentiment features are merged with the original Airbnb datasets. The result is a complete listing profile that includes both structured attributes and emotional signals extracted from text. Run the notebook sequentially to produce listings_reviews_final csv.

## Model_Comparison.ipynb

This notebook compares two price prediction models

• a baseline model using standard listing features

• an emotion aware model that incorporates sentiment, dominant emotions, and emotional tone of descriptions

Emotion enriched features consistently improve RMSE and R squared scores because they capture how guests actually feel about the properties. Run all cells to view performance metrics and visual comparisons.

## Streamlit Application (app.py)

The final step delivers a clean Airbnb style interface powered by emotional intelligence. The app loads all models and embeddings, accepts user queries, applies filters, and displays listings with predicted prices and value ratings influenced by emotional features.

Launch the app by running
 
  ```
  streamlit run app.py
```
## License
This project is open-source and available under the MIT License.
# Emotion-Infused-Collaborative-Filtering-for-Personalized-Airbnb-Recommendations
# Emotion-Infused-Collaborative-Filtering-for-Personalized-Airbnb-Recommendations
# Emotion-Infused-Collaborative-Filtering-for-Personalized-Airbnb-Recommendations
# Emotion-Infused-Collaborative-Filtering-used-for-Personalized-Airbnb-Recommendations
