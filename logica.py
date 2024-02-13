import math
import pandas as pd
import datetime
import math
import numpy as np
import pandas_datareader.data as web
import pickle
from pandas import Series, DataFrame
from pandas.plotting import scatter_matrix
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Ridge, Lasso, MultiTaskLasso, BayesianRidge, LassoLars, OrthogonalMatchingPursuit, ARDRegression, LogisticRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import l1_min_c

from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
import matplotlib.pyplot as mlpt


import tweepy
import csv
import random

from pycm import ConfusionMatrix

######DISCLAIMER! THIS VIEW IS HORRIFIC, WATCH WITH YOUR OWN RISK OF BLEEDING EYES AND TURNING EYES BACK!, YOU'VE BEEN WARNED!#######

def task1(df):

    try:
        #Get all closing values
        close_px = df['Adj Close']

        #Create moving avergae values
        mavg = close_px.rolling(window=10).mean()

        #Calculate rets
        rets = close_px / close_px.shift(1) - 1

        #Provide data to Flask app
        return close_px.index.format(formatter=lambda x: x.strftime('%Y-%m-%d')), close_px.to_list(), mavg.to_list(), rets.to_list()

    #If any error, provide back to flask app, although it does not work properly.
    except TypeError as e:
        return e
    except NameError as e:
        return e
    except Exception as e:
        return e
    except RemoteDataError as e:
        return e

def task2(data1,comp):
    
    consumer_key    = '3jmA1BqasLHfItBXj3KnAIGFB'
    consumer_secret = 'imyEeVTctFZuK62QHmL1I0AUAMudg5HKJDfkx0oR7oFbFinbvA'

    access_token  = '265857263-pF1DRxgIcxUbxEEFtLwLODPzD3aMl6d4zOKlMnme'
    access_token_secret = 'uUFoOOGeNJfOYD3atlcmPtaxxniXxQzAU4ESJLopA1lbC'
    
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth,wait_on_rate_limit=True)
    
    url=comp.lower()
    url=url.strip('\t')
    print(url)
    
    if url.endswith('limited'):
        url = url[:-7]
    
    print(url)
    
    q="#"+url
    
    print(q)
    # #### Fetching tweets for Tesla in extended mode (means entire tweet will come and not just few words + link)
    
    # In[4]:
    
    
    fetch_tweets=tweepy.Cursor(api.search, q ,count=100, lang ="en",since="2020-02-01", tweet_mode="extended").items()
    data=pd.DataFrame(data=[[tweet_info.created_at.date(),tweet_info.full_text]for tweet_info in fetch_tweets],columns=['Date','Tweets'])
    
    
    # #### Removing special character from each tweets
    
    # In[5]:
    
    
    data.to_csv("Tweets.csv")
    cdata=pd.DataFrame(columns=['Date','Tweets'])
    total=100
    index=0
    for index,row in data.iterrows():
        stre=row["Tweets"]
        my_new_string = re.sub('[^ a-zA-Z0-9]', '', stre)
        cdata.sort_index()
        cdata.set_value(index,'Date',row["Date"])
        cdata.set_value(index,'Tweets',my_new_string)
        index=index+1
    #print(cdata.dtypes)
    
    
    # #### Displaying the data with date and tweets, you can notice there are multiple tweets for each day. So we will club them together later.
    
    # In[6]:
    
    
    print(cdata)
    
    
    # #### Creating a dataframe where we will combine the tweets date wise and store into
    
    # In[7]:
    
    
    ccdata=pd.DataFrame(columns=['Date','Tweets'])
    
    
    # In[8]:
    
    
    indx=0
    get_tweet=""
    for i in range(0,len(cdata)-1):
        get_date=cdata.Date.iloc[i]
        next_date=cdata.Date.iloc[i+1]
        if(str(get_date)==str(next_date)):
            get_tweet=get_tweet+cdata.Tweets.iloc[i]+" "
        if(str(get_date)!=str(next_date)):
            ccdata.set_value(indx,'Date',get_date)
            ccdata.set_value(indx,'Tweets',get_tweet)
            indx=indx+1
            get_tweet=" "
    
    
    # #### All the tweets has been clubbed as per their date.
    
    # In[9]:
    
    
    print(ccdata)
    
    
    # #### Now to know the "closing price" of each day we will import STOCK PRICE DATA for UNITED AIRLINES from "yahoo.finance". We will consider "Close" price only.
    
    # In[10]:
    
    
    read_stock_p=data1
    read_stock_p=read_stock_p.reset_index()
    print(read_stock_p)
    
    
    # #### Adding a "Price" column in our dataframe and fetching the stock price as per the date in our dataframe.
    
    # In[11]:
    
    
    print(int(read_stock_p.Close[4]))
    ccdata['Prices']=""
    
    
    # In[12]:
    
    
    read_stock_p['new_date'] = [d.date() for d in read_stock_p['Date']]
    read_stock_p['new_time'] = [d.time() for d in read_stock_p['Date']]
    
    #read_stock_p['new_date1'] = read_stock_p['new_date'].dt.strftime('%d/%m/%Y')
    
    print(read_stock_p)
    
    indx=0
    for i in range (0,len(ccdata)):
        for j in range (0,len(read_stock_p)):
            get_tweet_date=ccdata.Date.iloc[i]
            get_stock_date=read_stock_p.new_date.iloc[j]
            print(get_stock_date," ",get_tweet_date)
            if(str(get_stock_date)==str(get_tweet_date)):
                ccdata.set_value(i,'Prices',int(read_stock_p.Close[j]))
                break
   
    
    # #### Prices are fetched but some entires are blank as close price might not be available for that day due to some reason (like holiday, etc.)
    
    # In[13]:
    
    
    print(ccdata)
    leng=len(ccdata.index)
    print(leng)
    # #### So we take the mean for the close price and put it in the blank value
    
    # In[14]:
    
    
    mean=0
    summ=0
    count=0
    for i in range(0,len(ccdata)):
        if(ccdata.Prices.iloc[i]!=""):
            summ=summ+int(ccdata.Prices.iloc[i])
            count=count+1
    mean=summ/count
    for i in range(0,len(ccdata)):
        if(ccdata.Prices.iloc[i]==""):
            ccdata.Prices.iloc[i]=int(mean)
    
    
    # #### Now all the entries have some value
    
    # In[15]:
    
    
    print(ccdata)
    
    
    # #### Making "prices" column as integer so mathematical operations could be performed easily.
    
    # In[16]:
    
    
    ccdata['Prices'] = ccdata['Prices'].apply(np.int64)
    
    
    # #### Adding 4 new columns in our dataframe so that sentiment analysis could be performed.. Comp is "Compound" it will tell whether the statement is overall negative or positive. If it has negative value then it is negative, if it has positive value then it is positive. If it has value 0, then it is neutral.
    
    # In[17]:
    
    
    ccdata["Comp"] = ''
    ccdata["Negative"] = ''
    ccdata["Neutral"] = ''
    ccdata["Positive"] = ''
    ccdata["Classlabel"] = ''
    ccdata
    
    
    # #### Downloading this package was essential to perform sentiment analysis.
    
    # In[18]:
    
    
    import nltk
    nltk.download('vader_lexicon')
    
    
    # #### This part of the code is responsible for assigning the polarity for each statement. That is how much positive, negative, neutral you statement is. And also assign the compound value that is overall sentiment of the statement.
    
    # In[19]:
    
    
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import unicodedata
    df = []
    sentiment_i_a = SentimentIntensityAnalyzer()
    for indexx, row in ccdata.T.iteritems():
        try:
            sentence_i = unicodedata.normalize('NFKD', ccdata.loc[indexx, 'Tweets'])
            sentence_sentiment = sentiment_i_a.polarity_scores(sentence_i)
            ccdata.set_value(indexx, 'Comp', sentence_sentiment['compound'])
            ccdata.set_value(indexx, 'Negative', sentence_sentiment['neg'])
            ccdata.set_value(indexx, 'Neutral', sentence_sentiment['neu'])
            ccdata.set_value(indexx, 'Positive', sentence_sentiment['pos'])
            df.append(sentence_sentiment['compound'])
            #ccdata.set_value(indexx, 'Classlabel',  pd.cut(sentence_sentiment['compound'], bins=5, labels=[1, 2, 3, 4, 5]))
        except TypeError:
            print (stocks_dataf.loc[indexx, 'Tweets'])
            print (indexx)
    
    y_actu = []
    y_pred = []
    print(df)
    
    if (len(df)%2!=0):
        df.append(0)
    print(df)    
    
    df=(pd.cut(df, bins=[-1,-0.5,-0.01,0.01,0.5,1] , labels=['Strong Buy','Weak Buy','Hold','Weak Sell','Strong Sell']))
    
    
    # In[20]:
    
    
    def list_splitter(list_to_split, ratio):
        elements = len(list_to_split)
        middle = int(elements * ratio)
        return [list_to_split[:middle], list_to_split[middle:]]
    
    
    [y_actu,y_pred]=list_splitter(df,0.5)
    
    print(y_actu)
    print(y_pred)
    
    y_actu=np.asarray(y_actu)
    y_pred=np.asarray(y_pred)
    
    
    try:
        cm = ConfusionMatrix(actual_vector=y_actu, predict_vector=y_pred)
        cm.print_matrix()
    except:
        pass
    #cm.print_normalized_matrix()
    
    
    # In[21]:
    
    
    print(ccdata)
    
    
    # #### Calculating the percentage of postive and negative tweets, and plotting the PIE chart for the same.
    
    # In[22]:
    
    
    posi=0
    nega=0
    for i in range (0,len(ccdata)):
        get_val=ccdata.Comp[i]
        if(float(get_val)<(0)):
            nega=nega+1
        if(float(get_val>(0))):
            posi=posi+1
    posper=(posi/(len(ccdata)))*100
    negper=(nega/(len(ccdata)))*100
    print("% of positive tweets= ",posper)
    print("% of negative tweets= ",negper)
    arr=np.asarray([posper,negper], dtype=int)
    mlpt.pie(arr,labels=['positive','negative'])
    mlpt.plot()
    
    
    # #### Making a new dataframe with necessary columns for providing machine learning.
    
    # In[23]:
    
    
    df_=ccdata[['Date','Prices','Comp','Negative','Neutral','Positive']].copy()
    
    
    # In[24]:
    
    
    df_
    
    
    # #### Dividing the dataset into train and test.
    
    # In[25]:
    
    
    train_start_index = '0'
    train_end_index =leng-4 
    test_start_index = '0'
    test_end_index = leng-1
    train = df_.ix[train_start_index : train_end_index]
    test = df_.ix[test_start_index:test_end_index]
    
    
    # #### Making a 2D array that will store the Negative and Positive sentiment for Training dataset.
    
    # In[26]:
    
    
    sentiment_score_list = []
    for date, row in train.T.iteritems():
        sentiment_score = np.asarray([df_.loc[date, 'Negative'],df_.loc[date, 'Positive']])
        sentiment_score_list.append(sentiment_score)
    numpy_df_train = np.asarray(sentiment_score_list)
    
    
    # In[27]:
    
    
    print(numpy_df_train)
    
    
    # #### Making a 2D array that will store the Negative and Positive sentiment for Testing dataset.
    
    # In[28]:
    
    
    sentiment_score_list = []
    for date, row in test.T.iteritems():
        sentiment_score = np.asarray([df_.loc[date, 'Negative'],df_.loc[date, 'Positive']])
        sentiment_score_list.append(sentiment_score)
    numpy_df_test = np.asarray(sentiment_score_list)
    
    
    # In[29]:
    
    
    print(numpy_df_test)
    
    
    # #### Making 2 dataframe for Training and Testing "Prices". You can also make 1-D array for the same.
    
    # In[30]:
    
    
    y_train = pd.DataFrame(train['Prices'])
    #y_train=[91,91,91,92,91,92,91]
    y_test = pd.DataFrame(test['Prices'])
    print(y_train)
    
    
    # #### Fitting the sentiments(this acts as in independent value) and prices(this acts as a dependent value (like class-lables in iris dataset))
    
    # In[31]:
    
    
    from treeinterpreter import treeinterpreter as ti
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import classification_report,confusion_matrix
    
    rf = RandomForestRegressor()
    rf.fit(numpy_df_train, y_train)
    
    
    # #### Making Predictions
    
    # In[32]:
    
    
    prediction, bias, contributions = ti.predict(rf, numpy_df_test)
    
    
    # In[33]:
    
    
    print(prediction)
    
    
    # #### Importing matplotlib library for plotting graph
    # In[34]:
    
    
    import matplotlib.pyplot as plt
    
    
    # #### Defining index position for the test data. Making dataframe for the predicted value.
    
    # In[35]:
    
    
    idx=np.arange(int(test_start_index),int(test_end_index)+1)
    print(idx)
    data=prediction[0:]
    print(data)
    columns=['Prices']
    print(columns)
    predictions_df_ = pd.DataFrame(data, index = idx, columns=['Prices'])
    
    
    # In[36]:
    
    
    print(predictions_df_)
    
    
    # #### Plotting the graph for the Predicted_price VS Actual Price
    
    # In[37]:
    
    
   
    
    
    dfreg=pd.DataFrame(columns=['Date','Actual'])
    
    dfreg['Date']=df_['Date']
    dfreg['Actual']=df_['Prices']
    dfreg['RandomForest']=predictions_df_['Prices']
    
    
    reg = LinearRegression()
    reg.fit(numpy_df_train, y_train)
    dfreg['LinearRegression']=reg.predict(numpy_df_test)
    
    rf1 = DecisionTreeRegressor()
    rf1.fit(numpy_df_train, y_train)
    dfreg['DecisionTree']=rf1.predict(numpy_df_test)
    
    from sklearn.neural_network import MLPClassifier
    mlpc = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', #'relu', the rectified linear unit function
                         solver='lbfgs', alpha=0.005, learning_rate_init = 0.001, shuffle=False)
    """Hidden_Layer_Sizes: tuple, length = n_layers - 2, default (100,)
    The ith element represents the number of Neutralrons in the ith
    hidden layer."""
    mlpc.fit(numpy_df_train,y_train)   
    dfreg['MLPClassifier'] = mlpc.predict(numpy_df_test)
    
    
    clfknn = KNeighborsRegressor(n_neighbors=2)
    clfknn.fit(numpy_df_train, y_train)
    dfreg['KNN'] = clfknn.predict(numpy_df_test)
    
    clflas = Lasso()
    clflas.fit(numpy_df_train, y_train)
    dfreg['Forecast_las'] = clflas.predict(numpy_df_test)
    
    clfbyr = BayesianRidge()
    clfbyr.fit(numpy_df_train, y_train)
    dfreg['Forecast_byr'] = clfbyr.predict(numpy_df_test)
    
    clflar = LassoLars(alpha=.1)
    clflar.fit(numpy_df_train, y_train)
    dfreg['Forecast_lar'] = clflar.predict(numpy_df_test)
    
    clfomp = OrthogonalMatchingPursuit(n_nonzero_coefs=2)
    clfomp.fit(numpy_df_train, y_train)
    dfreg['Forecast_omp'] = clfomp.predict(numpy_df_test)
    
    clfard = ARDRegression(compute_score=True)
    clfard.fit(numpy_df_train, y_train)
    dfreg['Forecast_ard'] = clfard.predict(numpy_df_test)
    
    clfsgd = SGDRegressor(random_state=0, max_iter=1000, tol=1e-3)
    clfsgd.fit(numpy_df_train, y_train)
    dfreg['Forecast_sgd'] = clfsgd.predict(numpy_df_test)
    
    
    print(dfreg)
    
    print(rf.score(numpy_df_train,y_train))
    print(reg.score(numpy_df_train,y_train))
    #print(rf1.score(numpy_df_train,y_train))
    print(clfknn.score(numpy_df_train,y_train))
    print(clflas.score(numpy_df_train,y_train))
    print(clfbyr.score(numpy_df_train,y_train))
    print(clflar.score(numpy_df_train,y_train))
    print(clfomp.score(numpy_df_train,y_train))
    print(clfard.score(numpy_df_train,y_train))
    print(clfsgd.score(numpy_df_train,y_train))
    
    print("hello")
    
    
    
    
    dfreg=dfreg.set_index('Date')

    return dfreg.index.format(formatter=lambda x: x.strftime('%Y-%m-%d')), dfreg['Actual'].to_list(), dfreg['RandomForest'].to_list(), dfreg['LinearRegression'].to_list(), dfreg['DecisionTree'].to_list(), dfreg['KNN'].to_list(), dfreg['Forecast_las'].to_list(), dfreg['Forecast_byr'].to_list(), dfreg['Forecast_lar'].to_list(), dfreg['Forecast_omp'].to_list(), dfreg['Forecast_ard'].to_list(), dfreg['Forecast_sgd'].to_list()








