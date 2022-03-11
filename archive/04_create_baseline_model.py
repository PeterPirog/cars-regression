from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import json

# from pyearth import Earth

if __name__ == '__main__':
    filepath = '/home/ppirog/projects/cars-regression/text_dataset.csv'
    sep = ';'
    encoding = 'utf-8'
    df = pd.read_csv(filepath, sep=sep, encoding=encoding, on_bad_lines='skip', low_memory=False)

    # Use train_test_split to split training data into training and validation sets
    train_sentences, val_sentences, train_labels, val_labels = train_test_split(df["Text"].to_numpy(),
                                                                                df["Amount"].to_numpy(),
                                                                                test_size=0.05,
                                                                                # dedicate 10% of samples to validation set
                                                                                random_state=42)  # random state for reproducibility

    print(train_sentences)

    from pandas.core.dtypes.cast import maybe_infer_to_datetimelike
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB, GaussianNB
    from sklearn.linear_model import ElasticNet
    from sklearn.pipeline import Pipeline

    # Create tokenization and modelling pipeline
    model0 = Pipeline([
        ("tfidf", TfidfVectorizer()),  # convert words to numbers using tfidf
        ("clf", ElasticNet())  # model the text
    ])

    # Fit the pipeline to the training data
    model0.fit(train_sentences, train_labels)

    score = model0.score(val_sentences, val_labels)
    joblib.dump(model0, 'model0.pkl')

    print(score)
    model0_vocabulary = model0['tfidf'].vocabulary_
    print(model0_vocabulary)

    with open('model0_vocabulary.json', 'w') as fp:
        json.dump(model0_vocabulary, fp)
