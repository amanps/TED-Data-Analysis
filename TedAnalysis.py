
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import ast
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from IPython.display import display
from pandas.plotting import scatter_matrix
from datetime import datetime

stopWords = set(stopwords.words('english'))
ted = pd.read_csv("./data/ted_main.csv")
transcript = pd.read_csv('./data/transcripts.csv')


# In[2]:


ted.head()


# In[3]:


ted.describe()


# In[4]:


#Checking data types
ted.info()


# In[5]:


ted["views"].mean()


# In[6]:


#Correlation matrix
ted.corr()


# In[7]:


#Boxplot for comments distribution
ted.boxplot(column = "comments")
ted["comments"].quantile(0.9)


# In[8]:


#Boxplot for views distribution
ted.boxplot(column = "views")
ted["views"].quantile(0.9)
ted[ted["views"] > 3051912].shape


# In[9]:


ted.boxplot(column = "duration")
# ted["duration"].quantile(0.9)
# ted["duration"].max()
ted[ted["duration"] == 5256].title
ted[ted["duration"] == 5256].main_speaker


# In[10]:


# Total count for each rating. Most talks pick up the inspiring rating.

import operator
ratings = {}
for index, rating_str in ted["ratings"].iteritems():
    ratings_list = ast.literal_eval(rating_str)
    for rating in ratings_list:
        ratings[rating["name"]] = ratings.get(rating["name"], 0) + rating["count"]
for i in sorted(ratings.items(), key=operator.itemgetter(1)):
    print(i)


# In[11]:


# Adding columns with counts for all ratings for a talk.

ratings_df = ted.copy()
ratings_list = ratings.keys()
for rating in ratings_list:
    ratings_df[rating] = 0


# In[12]:


# Speakers with more than 1 talk.

import operator
name_dict = {}
for name in ted["main_speaker"].iteritems():
    name_dict[name[1]] = name_dict.get(name[1], 0) + 1
count = 0
for k in sorted(name_dict.items(), key = operator.itemgetter(1)):
    if k[1] > 1:
        print ("%s - %s" % (k[0], k[1]))


# In[13]:


# Set of all tags

tag_set = set()
for tag_str in ted["tags"]:
    tag_list = ast.literal_eval(tag_str)
    for tag in tag_list:
        tag_set.add(tag)
#print(tag_set)


# In[14]:


# Print tags per talk of speaker.

def print_tags_for_speaker(speaker):
    rosling = ted[ted["main_speaker"] == speaker]
    tag_set = []
    for tag_str in rosling["tags"]:
        tag_list = ast.literal_eval(tag_str)
        tag_set.append(tag_list)
    print(tag_set)

#print_tags_for_speaker("Hans Rosling")
# print_tags_for_speaker("Juan Enriquez")


# In[15]:


# Most viewed talks.

most_viewed = ted[["title", "main_speaker", "views"]].sort_values("views", ascending=False)
most_viewed.head()


# In[16]:


# Most commented on talks

most_commented = ted[["title", "main_speaker", "views", "comments"]].sort_values("comments", ascending=False)
most_commented.head()


# In[17]:


# There doesn't seem to be any correlation between views and comments. The top viewed TED talk is not the top commented 
# inspite of having 10 times more views than the top commented one "Militant Atheism".

display(ted.plot(x = "views", y = "comments", kind = "scatter"))
display(ted[(ted["comments"] < 400) & (ted["views"] < 3050000)].plot(x = "views", y = "comments", kind = "scatter"))
display(ted[(ted["views"] < 500000) & (ted["comments"] > 600)].head())
display(ted[(ted["views"] > 3000000) & (ted["comments"] < 50)].head())


# In[18]:


# We observe that views and languages are slightly positively correlated 0.3, TED talks with more than 10 million 
# have atleast 28 languages

ted.plot(x = "views", y = "languages", kind = "scatter")
ted[ted["views"] > 10000000].languages.sort_values().head(1)


# In[19]:


# Each rating with the associated score for each talk normalized over the number of views. 

def populate_ratings():
    for index, rating_str in ratings_df["ratings"].iteritems():
        max_rating = -1
        ratings_list = ast.literal_eval(rating_str)
        for rating in ratings_list:
            ratings_df.loc[index, rating["name"]] = rating["count"] / ted.iloc[index]["views"]
populate_ratings()


# In[20]:


display_list = ["title", "main_speaker"] + list(ratings_list)
ratings_df[display_list].head()


# In[21]:


for rating in ratings_list:
    display(ratings_df.sort_values(by = rating, ascending = False)[["title", rating, "views"]].head(5))
    
# This gives us a more accurate description of whether the talk was funny/inspiring etc. This is per user how many 
# people found it funny as opposed to overall coz it may be biased for a talk with more views.


# In[22]:


ratings_df[list(ratings_list)].corr()


# In[23]:


# ratings_df[list(ratings_list)].corr()
# display(ratings_df[["Jaw-dropping", "Unconvincing", "Fascinating", "Confusing", "OK", "Longwinded", "Beautiful"]].corr())
# display(ratings_df[["Persuasive", "Unconvincing", "Informative", "Confusing", "OK", "Funny"]].corr())
positive = ratings_df[["Jaw-dropping", "Unconvincing", "Fascinating", "Confusing", "OK", "Longwinded", "Beautiful"]]
negative = ratings_df[["Persuasive", "Unconvincing", "Informative", "Confusing", "OK", "Funny"]]
display(scatter_matrix(positive, alpha=1, figsize=(14, 14), diagonal='kde'))
display(scatter_matrix(negative, alpha=1, figsize=(14, 14), diagonal='kde'))

# We were expecting positive correlation between some ratings which we were able to verify. 
# Didn't seem to find any negative correlation which was surprising.


# In[24]:


# Trends of tags across years
ted["year"] = -1
def populate_years():
    for index, epoch in ted["published_date"].iteritems():
        ted.loc[index, "year"] = datetime.fromtimestamp(epoch).year

populate_years()
ted.head()


# In[25]:


tag_dict_list = []

for year in ted.year.unique():
    tag_dict = {}
    for index, talk in ted[ted["year"] == year].iterrows():
        tags_list = ast.literal_eval(talk["tags"])
        for tag in tags_list:
            tag_dict[tag] = tag_dict.get(tag, 0) + 1
    tag_dict_list.append(tag_dict)

year_tag_df = pd.DataFrame(tag_dict_list).fillna(0)
year_tag_df = year_tag_df.set_index(ted.year.unique())
year_tag_df


# In[26]:


# Filtering out tags

tags_of_interest = []

for i in year_tag_df.columns:
    # Removing tags that contain 'TED'
    if 'TED' not in i:
        tags_of_interest.append(i)

#print(tags_of_interest)


# In[27]:


# Top 10 tags of 2016-17
top_tags_1617 = year_tag_df.loc[[2016,2017],tags_of_interest].sum().sort_values(ascending = False)[:10].index

ax = year_tag_df[top_tags_1617].plot(figsize = (15,10), title = 'Trend of the top 10 tags of 2016-17 across all years' )
ax.set_xlabel("Years")
ax.set_ylabel("Number of Talks")
plt.show()


# In[ ]:


# Top 10 tags of all time
top_tags_alltime = year_tag_df.loc[:,tags_of_interest].sum().sort_values(ascending = False)[:10].index

ax = year_tag_df[top_tags_alltime].plot(figsize = (20,10), title = 'Trend of the top 10 tags of all time across all years' )
ax.set_xlabel("Years")
ax.set_ylabel("Number of Talks")
plt.show()


# In[ ]:


def make_dtm(lst):
    tokenizer = nltk.tokenize.RegexpTokenizer(r'[a-zA-Z]+') #Only english alphabets
    res = []
    for script in lst:
        d = {}
        d['audience_laughter'] = script.lower().count('(laughter)')
        d['audience_applause'] = script.lower().count('(applause)')
        tokens =  tokenizer.tokenize(script.lower())
        for word in tokens:
            if word in stopWords:
                continue
            d[word] =  d.get(word,0) + 1
        res.append(d)
    return pd.DataFrame(res).fillna(0)

#Creating the document term matrix
dtm = make_dtm(transcript.transcript)
#print(dtm.shape)
dtm_end = dtm.shape[1]


# In[ ]:


def wordcount(script):
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+') #Only english alphabets
    tokens =  tokenizer.tokenize(script.lower())
    return len(tokens)

#Features to calculate word per minutes
ratings_df['duration_min'] = round(ratings_df.duration / 60)
transcript['words'] = transcript['transcript'].apply(wordcount)
dtm['word_count'] = transcript.words
dtm['url_match'] = transcript.url


#Creating the combined dataframe
full_df = pd.merge(dtm,ratings_df, left_on = 'url_match', right_on = 'url')


# In[ ]:


#Word per minute calculations
full_df['wpm'] = full_df['word_count'] / full_df['duration_min']
full_df['wpm_category'] = ['Optimal' if (i >= 140 and i <= 175) else 'Fast' if i > 175 else 'Slow' for i in full_df.wpm]
display(full_df[full_df['word_count'] >= 100][['wpm','wpm_category']].describe())
display(full_df.wpm_category.value_counts())


# In[ ]:


#Cheaking the 0.2 wpm talk and other outliers

transcript.iloc[transcript.words.sort_values()[:15].index,[0,2]]


# In[ ]:


#Checking correlation of wpm with ratings
full_df[['wpm']+list(ratings_list)].corr()

#There is no strong correlation.


# In[ ]:


#Top occuring words excluding stop words
ax = dtm.iloc[:,:dtm_end].sum().sort_values(ascending = False)[:14].plot(kind = 'bar', title = 'Top Occurring Words')
ax.set_xlabel("Emoticons")
ax.set_ylabel("Number of Tweets")
plt.show()


# In[ ]:


#Word per minute analysis
agg_list = {}
for rating in ratings_list:
    agg_list[rating] = 'mean'

display(full_df.groupby(['wpm_category']).agg(agg_list))


# In[ ]:


#Audience Engagement
def getTopRating(lst):
    res = []
    count = []
    for x in lst:
        res.append(pd.DataFrame(eval(x)).sort_values(by = 'count', ascending = False).name[0])
        count.append(pd.DataFrame(eval(x)).sort_values(by = 'count', ascending = False).iloc[0][0])
    return (res,count)
full_df['top_rating'] , full_df['top_rating_votes'] = getTopRating(full_df.ratings_y)

#Laugther
display(full_df.sort_values(by = 'audience_laughter', ascending = False)[['top_rating','title_y']].head(15))

#Applause
display(full_df.sort_values(by = 'audience_applause', ascending = False)[['top_rating','title_y']].head(15))


# In[ ]:


#Word cloud
wordcloud = WordCloud(background_color="white",max_font_size=100).generate(' '.join(transcript.transcript))
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

