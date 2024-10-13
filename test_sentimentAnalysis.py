#%%

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import pandas as pd

nltk.download('vader_lexicon')

sentences = ["I am happy", 
             "I am sad", 
             "I am geek",
            "I got covid", 
            "I love geek"]

analyzer = SentimentIntensityAnalyzer()

result = []
for s in sentences:
    result.append(analyzer.polarity_scores(s))

i = 0
df = pd.DataFrame()
for i in range(len(sentences)):
    x = pd.DataFrame.from_dict(result[i], orient='index').T
    df = pd.concat([df,x], ignore_index=True)
df.index = sentences

print(df)

fig, ax = plt.subplots(figsize=(7,6))
ax.bar(df.index,df['compound'])
plt.rcParams.update({'font.size': 16})
ax.axhline(0, color='grey', linewidth=0.8)
ax.set_ylabel('Compound sentiment')
plt.xticks(rotation=40)


# %%
