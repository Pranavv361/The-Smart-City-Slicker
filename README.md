## cs5293sp23-project3 (Text - Analytics Project 3)
## Author - Pranav Vichare
## The Smart City Slicker
**About:**  Project 3's purpose is to  to investigate themes and similarities for smart cities with the use of cluster analysis, topic modeling, and summarization. Assume you are a stakeholder in a rising Smart City and want to know more about themes and concepts about existing smart cities. You also want to know where does your smart city place among others. The following steps should be taken.
1. Download and clean pdf documents.
2. Create and explore clustering models.
3. Perform topic modeling to derive meanings.
4. Extract a summary and keywords for each smart city document.

To run the code, Use the commands below in terminal or Visual Studio Code
```python
pipenv --python 3.9  #use your version of python which is installed on your system. This code is also used to create a virtual environment
pipenv install scikit-learn==1.0.2 #to install scikit-learn library in virtual environment
pipenv install nltk #to install nltk library in virtual environment
pipenv install pandas #to install pandas library in virtual environment
pipenv install pytest #to install pytest library in virtual environment
pipenv install numpy #to install numpy library in virtual environment
pipenv install bs4 #to install bs4 library in virtual environment
pipenv install spacy #to install spacy library in virtual environment
pipenv install pypdf2 #to install pypdf2 library in virtual environment
#to install spacy en_core_web_sm model in virtual environment
pipenv install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.4.1/en_core_web_sm-3.4.1-py3-none-any.whl
pipenv lock #to create piplock file
```

```python
import argparse
import sys
import os
import pandas as pd
import PyPDF2
import nltk
nltk.download('stopwords', quiet = True)
import spacy
import unicodedata
import re
from nltk.corpus import wordnet
import collections
from nltk.tokenize.toktok import ToktokTokenizer
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from joblib import load
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import numpy as np
```
To run the code, Import all the libraries listed above in the python code.

To run the code from the terminal use below command:
```
pipenv run python project3.py --document city.pdf --summarize --keywords"
```

### Code Explanation  
```python
#project2.py and PreprocessAndModelling.py has 4 functions as shown below.
def load_data()
def main()
def normalize_data()
def modeltrain()
```
**load_data()** reads data from a JSON file and returns it as a pandas DataFrame.

When the script is executed, **the main()** method is invoked. It takes an args parameter, which is parsed by argparse. The function calls **load_data()** to load data from a JSON file, and then **modeltrain()** to train and test a LinearSVC model on the loaded data.

The function **normalize_data()** is used to normalize a list of ingredients. It strips out specific verbs from the ingredients, lemmatizes the remaining words, removes non-alphabetic characters and spaces, and replaces spaces with underscores. The normalized ingredients are returned as a comma-separated string.

**modeltrain()** trains and tests a LinearSVC model on the given data. It normalizes the ingredients and removes duplicates, divides the input data into features (ingredients) and target (cuisine), encodes the target variable, uses TfidfVectorizer to convert the text data into numerical form, builds a pipeline with a preprocessor and estimator, performs cross-validation, fits the model on training data, predicts on testing data, and saves the model to a joblib file. If the model is already there, the function loads it from the file. The function accepts three arguments: data, pandas, and The input data, ingred - a list of ingredients for which the cuisine must be predicted, and top_n - a number indicating how many of the top closest matches to return. The function produces a tuple containing the names and scores of the top N closest matches, as well as a list of dictionaries containing the closest matches.

These values are stored in the stats files.
*The output video of the execution with the images is stored in **Output videos and images** folder*

### Code Output
The code output is stored in stored in file with name : **smart_predict.tsv**.
![image](https://github.com/Pranavv361/cs5293sp23-project3/blob/main/docs%20-%20Execution%20video%20and%20images/project3.py%20Execution.png)

### Test Output
The test cases are created in single file **test_project3.py**. The purpose of the test_project3.py is to check the functions with sample input and make sure we get correct output. The attached image below shows the output for test cases of all the functions.
To run the test_project2.py using pytest library use the following code.
```
pipenv run python -m pytest
```
![image](https://github.com/Pranavv361/cs5293sp23-project3/blob/main/docs%20-%20Execution%20video%20and%20images/test_project3.py%20Execution.png)

The **load_data()** function is tested using the **test_load_data()** method. It makes a test directory, generates a JSON file with test data named **test.json** in **docs** folder, then loads it with the **load_data()** function. The function then checks to see if the output is a pandas DataFrame, if the column names are correct, and if the data matches the expected values.

The function **test_normalize_data()** verifies the **normalize_data()** function defined in the PreprocessAndModelling module. It sends a list of ingredient names to the **normalize_data()** method and checks for a string of normalized ingredient names.

The **modeltrain()** function created in the PreprocessAndModelling module is tested using the **test_modeltrain()** method. It uses the **load_data()** method to load the yummly.json file, and then gives the loaded data, a list of ingredients, and a number to the **modeltrain()** function. After that, the function checks to see if the output lists contain the expected lengths and data types.

### Assumptions:
1. The data file should always be .pdf file.

### Bugs:   
1. Some words in keywords/summary and topics are incomplete because of normalizing techniques used.
