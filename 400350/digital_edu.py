import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
df = pd.read_csv('train.csv')
df_t = pd.read_csv('test.csv')
df.drop(['id', 'bdate', 'graduation', 'city', 'education_form', 'occupation_name', 'last_seen', 'occupation_type'], axis=1, inplace=True)
df_t.drop(['bdate', 'graduation', 'city', 'education_form', 'occupation_name', 'last_seen', 'occupation_type'], axis=1, inplace=True)
def career(year):
    try:
        year = int(year)
        return year
    except:
        return 0
df['career_start'] = df['career_start'].apply(career)
df['career_end'] = df['career_end'].apply(career)
df_t['career_start'] = df_t['career_start'].apply(career)
df_t['career_end'] = df_t['career_end'].apply(career)

def education(status):
    if status == 'Undergraduate applicant':
        return 1
    elif status == "Student (Specialist)":
        return 2
    elif status == "Alumnus (Specialist)":
        return 3
    elif status == "Student (Bachelor's)":
        return 4
    elif status == "Alumnus (Bachelor's)":
        return 5
    elif status == "Student (Master's)":
        return 6
    elif status == "Alumnus (Master's)":
        return 7
    elif status == 'PhD':
        return 8
    elif status == 'Candidate of Sciences':
        return 9
    return status
df['education_status'] = df['education_status'].apply(education)
df_t['education_status'] = df_t['education_status'].apply(education)

def lang(langs):
    langs = langs.split(';')
    return langs
def get_eng(lang):
    if 'English' in lang:
        return 1
    return 0
def get_rus(lang):
    if 'Русский' in lang:
        return 1
    return 0
def langs_amount(langs):
    return len(langs)
df['langs'] = df['langs'].apply(lang)
df['english'] = df['langs'].apply(get_eng)
df['russian'] = df['langs'].apply(get_rus)
df['langs amount'] = df['langs'].apply(langs_amount)
df.drop('langs', axis=1, inplace=True)
df_t['langs'] = df_t['langs'].apply(lang)
df_t['english'] = df_t['langs'].apply(get_eng)
df_t['russian'] = df_t['langs'].apply(get_rus)
df_t['langs amount'] = df_t['langs'].apply(langs_amount)
df_t.drop('langs', axis=1, inplace=True)

def zero_main(main):
    if main == 'False':
        return 0
    return main
    
df['life_main'] = df['life_main'].apply(zero_main)
df['people_main'] = df['people_main'].apply(zero_main)
df['life_main'] = df['life_main'].apply(int)
df['people_main'] = df['people_main'].apply(int)
df_t['life_main'] = df_t['life_main'].apply(zero_main)
df_t['people_main'] = df_t['people_main'].apply(zero_main)
df_t['life_main'] = df_t['life_main'].apply(int)
df_t['people_main'] = df_t['people_main'].apply(int)


ID = df_t['id']
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.25)
x_train = df.drop('result', axis = 1)
y_train = df['result']
x_test = df_t.drop('id', axis = 1)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
classifier = KNeighborsClassifier(n_neighbors = 7) 
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
result = pd.DataFrame({'id' : ID, 'result' : y_pred})
result.to_csv('result.csv', index= False)
# percent = accuracy_score(y_test, y_pred) * 100
print(result)