from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("submission.csv")
sentiment = SentimentIntensityAnalyzer()
sentiment_scores = []
# df= pd.DataFrame(data)
# print(data)
for x in data["selected_text"]:
    # print(x)
#     text = row['selected_text']
    sentiment_result = sentiment.polarity_scores(x)
    # print(sentiment_result)
    sentiment_scores.append({"text":x,**sentiment_result})
#     # print(sentiment_result)
sentiment_df = pd.DataFrame(sentiment_scores)
print(sentiment_df)
pos = sentiment_df["pos"].mean()
neg = sentiment_df["neg"].mean()
neu = sentiment_df["neu"].mean()

print(pos, neg, neu)
y = (pos,neg, neu)
x = ("pos","neg","neu")
plt.bar(x,y)
plt.xlabel = "Sentiment"
plt.ylabel= "Value"
plt.title = "Chart"

plt.show()
