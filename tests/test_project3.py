from modules import normalizeText, predict
import pandas as pd
import numpy as np


#This test function takes a string of text and returns a 
#normalized version of the text with all words in lower case, no stop words, and no special characters or HTML tags.
def test_normalize_text():
    text = "This is an example text. It has HTML tags, accented characters, contractions, special characters, and stopwords. Let's see if the normalize_text function can handle it!"

    df = normalizeText.normalize_text(text)
    #breakpoint()
    assert normalizeText.normalize_text(text) == "example text html tag accent character contraction special character stopword let see normalizetext function handle"


#This test function takes a Pandas DataFrame containing normalized
#text and returns a DataFrame with an additional column indicating the cluster ID.
def test_cluster():

    df = pd.DataFrame({
        'Normalized Text': ['this is some normalized text', 'more normalized text here']
    })

    result = predict.cluster(df)
    #breakpoint()
    assert 'Cluster ID' in result.columns

    assert result['Cluster ID'].dtype == np.dtype('int32')


#This test function takes a Pandas DataFrame containing normalized text and 
#returns a DataFrame with an additional column indicating the topic ID
def test_topic():

    df = pd.DataFrame({
        'Normalized Text': ['this is some normalized text', 'more normalized text here']
    })

    result = predict.topic(df)
    #breakpoint()
    assert 'Topic_ids' in result.columns

    assert len(result['Topic_ids']) == len(df)


#This test function takes a string of text and returns a list of the most important words in the text 
# based on their frequency and relevance.
def test_keywords_text():

    text = 'This is some example text. It contains some words that are not important, like "is", "some", and "that". The important words are "example", "text", "contains", and "words".'

    result = predict.keywords_text(text)
    #breakpoint()
    assert isinstance(result, list)
    assert isinstance(result[0], str)

    assert len(result) <= 4


# This test function takes a string of text and returns a summary of the text
# consisting of the most important words.
def test_summary():

    text1 = 'This is some example text. It contains some words that are not important, like "is", "some", and "that". The important words are "example", "text", "contains", and "words".'
    text2 = 'This is some more example text. It contains some more words that are not important, like "is", "some", and "that". The important words are "example", "text", "contains", and "words".'
    text3 = 'This is some even more example text. It contains some even more words that are not important, like "is", "some", and "that". The important words are "example", "text", "contains", and "words".'

    test_cases = [(text1, 'example text contains words'), 
                  (text2, 'example text contains words'), 
                  (text3, 'example text contains even')]

    for text, expected_summary in test_cases:
        result = predict.summary(text)
        #breakpoint()
        assert isinstance(result, str)

        assert len(result) <= 4 * 5 