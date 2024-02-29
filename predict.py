import json 
import pandas as pd
import numpy as np
import re
import string
import nltk
import json
from pandas import json_normalize

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier


def f(x):
    s = {'comment':['350 000 - 400 000 гросс', '450 на руки', 'удаленка, 300к гросс (может +10%) + 2 оклада премия годовая в среднем', 'смотрим до 500 net окладом + perf review каждые 6 мес. (1,5-2 оклада после ревью премию/ 1,5 для senior, 2 для lead)'],
        'comment_int': [375, 450, 300, 500]}
    try:
        return s['comment_int'][s['comment'].index(x)]
    except:
        return 0
        

def stemm_text(text):
    stemmer = nltk.SnowballStemmer("russian")
    text = ' '.join(stemmer.stem(word) for word in text.split(' '))
    return text


def clean_text(text):
    text = str(text).lower()
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    return text


def get_data():
    with open('data/case_2_data_for_members.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    data_members = pd.DataFrame(list(data))
    data_members_vacancy = pd.DataFrame(list(data_members['vacancy']))
    data_members_vacancy.head(5)

    return data_members, data_members_vacancy

def get_reference_data():
    with open('data/case_2_reference_without_resume_sorted.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    return data


def predict():
    #preprocessing
    data_members, data_members_vacancy = get_data()

    df = pd.DataFrame(data_members['failed_resumes'][0])
    df['vacancy'] = data_members['vacancy'][0]['uuid']
    for i in range(1, 29):
        df2 = pd.DataFrame(data_members['failed_resumes'][i])
        df2['vacancy'] = data_members['vacancy'][i]['uuid']
        df = pd.concat([df, df2])
    df['prinali'] = np.zeros(len(df))

    df_confirmed_resumes = pd.DataFrame(data_members['confirmed_resumes'][0])
    df_confirmed_resumes['vacancy'] = data_members['vacancy'][0]['uuid']
    for i in range(1, 29):
        df2 = pd.DataFrame(data_members['confirmed_resumes'][i])
        df2['vacancy'] = data_members['vacancy'][i]['uuid']
        df_confirmed_resumes = pd.concat([df_confirmed_resumes, df2])
    df_confirmed_resumes['prinali'] = np.ones(len(df_confirmed_resumes))

    df = pd.concat([df_confirmed_resumes, df])
    data_members_vacancy.columns = ['vacancy', 'name', 'keywords', 'description', 'comment']
    df = pd.merge(df, data_members_vacancy, on='vacancy')

    df = df.drop(columns = ['uuid', 'first_name', 'last_name', 'vacancy', 'keywords'])

    df['experienceItem_'] = np.zeros(len(df))
    for i in range(len(df)):
        d = []
        try:
            for j in range(len(df['experienceItem'][i])):
                start = pd.to_datetime(df['experienceItem'][i][j]['starts'])
                end =  pd.to_datetime(df['experienceItem'][i][j]['ends'])
                if start == None or end == None:
                    duration = None
                else:
                    duration = end - start
                    d.append(duration.days / 364)

        except:
            print(i, "fail")
        if d:
            df['experienceItem_'][i] = int(np.sum(d))
        else:
            df['experienceItem_'][i] = None

    df = df.fillna(0)
        
    df['comment'] = df['comment'].apply(f)
        
    df['birth_date'] = df['birth_date'].replace(0, value='1990-01-01')
    df['birth_date'] = pd.to_datetime(df['birth_date'], format='%Y-%m-%d')
    target_date = pd.to_datetime('2024-02-28', format='%Y-%m-%d')
    df['age'] = (target_date - df['birth_date']).dt.days / 364

    df = df.drop(columns='birth_date')

    label_encoder = LabelEncoder()
    df['country'] = label_encoder.fit_transform(df['country'])

    df['Русский яз'] = np.zeros(len(df))
    df['Английский яз'] = np.zeros(len(df))
    for i in range(len(df)):
        if df['languageItem'][i] != 0 and df['languageItem'][i] != None:
            if "Английский" in df['languageItem'][i]:
                df['Английский яз'][i] = 1
            if "Русский" in df['languageItem'][i]:
                df['Русский яз'][i] = 1


    #embedding
    df['experienceItem_text'] = np.zeros(len(df))
    for i in range(len(df)):
        d = []
        try:
            for j in range(len(df['experienceItem'][i])):
                if df['experienceItem'][i][j]['description'] != None:
                    d.append(df['experienceItem'][i][j]['description'])
        except:
            print(i, "fail")
        if d:
            df['experienceItem_text'][i] = " ".join(d)

    for i in range(len(df)):
        if df['about'][i] != 0 and df['about'][i] != None:
            df['experienceItem_text'][i] += df['about'][i]

    df['experienceItem_text'] = df['experienceItem_text'].apply(clean_text)
    df['experienceItem_text'] = df['experienceItem_text'].apply(stemm_text)

    drops = ["country", "city", "about", "key_skills", "experienceItem", "educationItem", "name", "description", "languageItem", "languageItems", 'experienceItem_text']
    df.rename(columns={'prinali': 'target'}, inplace=True)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['experienceItem_text'])
    Y = df['target']

    emb = pd.DataFrame(X.toarray())
    df_with_embeddings = pd.concat([df.drop(columns=drops), emb], axis=1)
    df_with_embeddings = df_with_embeddings.applymap(lambda x: 0 if pd.isna(x) else x)

    X_ = df_with_embeddings.drop(columns='target')
    Y_ = df_with_embeddings['target']
   
    X_train, X_val, y_train, y_val = train_test_split(X_.values, Y_.values, test_size = 0.2, random_state = 42)

    model = LogisticRegression(random_state = 42)
    model.fit(X_train, y_train)

    model = XGBClassifier(booster='gbtree', max_depth=15, n_jobs=4, n_estimators=500)
    model.fit(X_train, y_train)

    #testing
    data = get_reference_data()
    data_resume = pd.DataFrame(data['resumes'])
    data_resume.head(3)

    data_vacancy = pd.DataFrame([data['vacancy']])
    data_vacancy.head(3)

    data_resume['lol'] = np.zeros(len(data_resume))
    data_vacancy['lol'] = np.zeros(len(data_vacancy))

    data_vacancy.columns = ['vacancy', 'name', 'keywords', 'description', 'comment', "lol"]
    data_resume = pd.merge(data_resume, data_vacancy, on='lol')
    data_resume = data_resume.drop(columns = ['first_name', 'vacancy', 'last_name', 'keywords', 'lol'])

    data_resume['experienceItem_'] = np.zeros(len(data_resume))
    for i in range(len(data_resume)):
        d = []
        try:
            for j in range(len(data_resume['experienceItem'][i])):
                start = pd.to_datetime(data_resume['experienceItem'][i][j]['starts'])
                end =  pd.to_datetime(data_resume['experienceItem'][i][j]['ends'])
                if start == None or end == None:
                    duration = None
                else:
                    duration = end - start
                    d.append(duration.days / 364)

        except:
            print(i, "fail")
        if d:
            data_resume['experienceItem_'][i] = int(np.sum(d))
        else:
            data_resume['experienceItem_'][i] = None

    data_resume = data_resume.fillna(0)
    data_resume['comment'] = data_resume['comment'].apply(f)

    data_resume['birth_date'] = data_resume['birth_date'].replace(0, value='1990-01-01')
    data_resume['birth_date'] = pd.to_datetime(data_resume['birth_date'], format='%Y-%m-%d')
    target_date = pd.to_datetime('2024-02-28', format='%Y-%m-%d')
    data_resume['age'] = (target_date - data_resume['birth_date']).dt.days / 364

    data_resume['experienceItem_text'] = np.zeros(len(data_resume))
    for i in range(len(data_resume)):
        d = []
        try:
            for j in range(len(data_resume['experienceItem'][i])):
                if data_resume['experienceItem'][i][j]['description'] != None:
                    d.append(data_resume['experienceItem'][i][j]['description'])
        except:
            print(i, "fail")
        if d:
            data_resume['experienceItem_text'][i] = " ".join(d)

    data_resume = data_resume.drop(columns='country')
    data_resume = data_resume.drop(columns='city')

    data_resume['Русский яз'] = np.zeros(len(data_resume))
    data_resume['Английский яз'] = np.zeros(len(data_resume))
    for i in range(len(data_resume)):
        if data_resume['languageItem'][i] != 0 and data_resume['languageItem'][i] != None:
            if "Английский" in data_resume['languageItem'][i]:
                data_resume['Английский яз'][i] = 1
            if "Русский" in data_resume['languageItem'][i]:
                data_resume['Русский яз'][i] = 1

    for i in range(len(data_resume)):
        if data_resume['about'][i] != 0 and data_resume['about'][i] != None:
            data_resume['experienceItem_text'][i] += data_resume['about'][i]

    data_resume['experienceItem_text'] = data_resume['experienceItem_text'].apply(clean_text)
    data_resume['experienceItem_text'] = data_resume['experienceItem_text'].apply(stemm_text)

    X = vectorizer.transform(data_resume['experienceItem_text'])

    drops = ['uuid', 
             'birth_date',
            'about',
            'key_skills',
            'experienceItem',
            'educationItem',
            'name',
            'description',
            'languageItem',
            'languageItems',
            'experienceItem_text']

    uuids = data_resume['uuid'].to_list()
    ans_df = data_resume.drop(columns=drops)

    x_emb = pd.DataFrame(X.toarray())
    X_embeddings = pd.concat([data_resume.drop(columns=drops), x_emb], axis=1)
    X_embeddings = X_embeddings.applymap(lambda x: 0 if pd.isna(x) else x)

    # predict
    model = LogisticRegression(random_state = 42)
    model.fit(X_.values, Y_)

    ans_df['uuid'] = uuids
    ans_df['ans'] = model.predict_proba(X_embeddings.values)[:, 1]
    ans_df = ans_df[['uuid', 'ans']]

    sorted_ans_df = ans_df.sort_values('ans', ascending=False)
    sorted_ans_df.to_csv('sorted_ans_df.csv', index=False)