
import numpy as np 
import pandas as pd
df = pd.read_csv('/kaggle/input/suicidal-tweet-detection-dataset/Suicide_Ideation_Dataset(Twitter-based).csv')
df.head()

# analysing the data
import seaborn as sns
sns.catplot(df, x = 'Suicide', kind = 'count')

# data cleaning
df.dropna(inplace = True)

# preprocessing the text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

def preprocessor(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = word_tokenize(text.lower())
    text = [PorterStemmer().stem(word) for word in text if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    return text
df['Tweet'] = df['Tweet'].apply(preprocessor)


# text to vector
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

df['Suicide'] = encoder.fit_transform(df['Suicide'])
df.head()


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5220)
X = cv.fit_transform(df['Tweet']).toarray()
y = df.iloc[:, -1].values

# implementing the model- 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=0)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print(f'Test Accuracy Score : {accuracy_score(y_test, y_pred):.2f}')

# tweet detection system
categories = ['Not Suicidal', 'Potentially Suicidal.']
text = input('enter your tweet')
text = cv.transform([preprocessor(text)])
print(categories[model.predict(text)[0]])
