#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Ładowanie bibliotek
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
from sklearn import preprocessing


# In[2]:


# Wczytaj dane DATA.csv
filename = 'DATA.csv'
data_location = 'art_condition/'
article_data = pd.read_csv(data_location + filename, sep=',')


# In[3]:


# Sprawdzam wczytane dane 
article_data.head()


# In[4]:


# Usuwam zbędne kolumny 'PRODUCT_KEY', 'ARTICLE_CODE' oraz 'ARTICLE_NAME', które są indeksami w badanym zbiorze.
del article_data['PRODUCT_KEY']
del article_data['ARTICLE_CODE']
del article_data['ARTICLE_NAME']


# In[5]:


# Ponownie sprawdzam dane.
article_data.head()


# In[6]:


# Sprawdzam strukturę danych. 
article_data.info()


# In[7]:


# Część zmiennym jest typu object, więc zmieniamy je na zmienną kategoryczną. W tym momencie zamieniam wszsystkie zmienne,
# dopiero w późniejszych krokach będę zastanawiać się nad ich eliminacją.  

for col in ["EU_BRAND_NAME", "SPGS_CLASS_NAME","SPGS_CLASS_CODE", "SPGS_DEPARTMENT_CODE","SPGS_DEPARTMENT_NAME",
            "SPGS_DIVISION_NAME","ARTICLE_TYPE","ARTICLE_TYPE_CODE","ECO_EASY_FLG", "PRICING_ARTICLE_CATEGORY_NAME",
            "SELLING_UOM_CODE", "ABOVE_THAN_AVG_country"]:
    article_data[col] = article_data[col].astype('category')


# In[8]:


# Ponownie sprawdzam strukturę danych. 
article_data.info()


# In[9]:


article_data.shape


# In[10]:


article_data.head()


# In[11]:


print(article_data.columns.str.cat(sep=", "))


# In[12]:


# Dla lepszej czytelności zmieniam pozycję zmiennej objaśnianej 'ARTICLE_SALES_CONDITION' na pierwsze miejsce.
print(article_data.columns.values)
article_data = article_data[['ARTICLE_SALES_CONDITION','EU_BRAND_CODE', 'EU_BRAND_NAME', 'SPGS_CLASS_CODE', 'SPGS_CLASS_NAME',
 'SPGS_DEPARTMENT_CODE', 'SPGS_DEPARTMENT_NAME' ,'SPGS_DIVISION_CODE',
 'SPGS_DIVISION_NAME', 'ARTICLE_TYPE', 'ARTICLE_TYPE_CODE' ,'ECO_EASY_FLG',
 'PRICING_ARTICLE_CATEGORY_CODE', 'PRICING_ARTICLE_CATEGORY_NAME',
 'ACTUAL_PURCHASE_PRICE_EUR_AMT', 'CATALOGUE_SALES_PRICE_EUR_AMT',
 'VAT_PERCENTAGE', 'SELLING_UOM_CODE', 'SPECIFIC_CUSTOMER_SALES',
 'SPECIFIC_CUSTOMER_CLASS_SALES_MEAN', 'COUNTRY_SALES',
 'COUNTRY_SALES_CLASS_MEAN', 'CLASS_ART_COUNT', 'ABOVE_THAN_AVG_CUSTOMER',
 'ABOVE_THAN_AVG_country']]


# In[13]:


article_data["ARTICLE_SALES_CONDITION"].value_counts()


# In[14]:


# Sprawdzam występowanie wartości nan
article_data.isnull().values.any()


# In[15]:


# Sprawdzam ilość wartości
article_data.isnull().sum().sum()


# In[16]:


# Usuwam wiersze z wartościami nan. Zbiór jest duży, więc można poświęcić 102 obserwacji.
article_data = article_data.dropna()
# Resetuję index.
article_data.reset_index(drop=True)


# In[17]:


# Sprawdza podstawowe statystyki zbioru. 
article_data.describe()


# In[18]:


# Zmienna "CATALOGUE_SALES_PRICE_EUR_AMT" posiada skrajne wartości. 
# Jako właściciel danych zakładam, że ceny sprzedaży równe 0.01 są błędne. Wyeliminuje również ceny sprzedaży powyżej 10,000
# ponieważ są to skrajne i sporadyczne przypadki, które negatywnie mogą wpłynąć na model.

# Warości > 10000 zamieniam na nan.
mask = article_data["CATALOGUE_SALES_PRICE_EUR_AMT"] > 10000
article_data.loc[mask,"CATALOGUE_SALES_PRICE_EUR_AMT"] = np.nan


# In[19]:


# Warości = 0.01 zamieniam na nan.
mask = article_data["CATALOGUE_SALES_PRICE_EUR_AMT"] ==  0.01
article_data.loc[mask,"CATALOGUE_SALES_PRICE_EUR_AMT"] = np.nan


# In[20]:


# Ponownie usuwam wiersze z wartościami nan. Zbiór jest duży, więc można poświęcić 102 obserwacji.
article_data = article_data.dropna()
# Resetuję index.
article_data.reset_index(drop=True)


# In[21]:


# Ponownie sprawdza podstawowe statystyki zbioru. 
article_data.info()


# In[22]:


# Standaryzuję zmienne liczbowe. W tym celu wydzielam je ze zbioru. 
scaler = preprocessing.StandardScaler()

# Tworzę nowy zbiór/kopię pod standaryzację.
article_data_std = article_data

# Dokonuję standaryzacji wybranych zmiennych.
article_data_std[['ACTUAL_PURCHASE_PRICE_EUR_AMT','CATALOGUE_SALES_PRICE_EUR_AMT',
                  'VAT_PERCENTAGE','SPECIFIC_CUSTOMER_CLASS_SALES_MEAN','COUNTRY_SALES_CLASS_MEAN', 
                  'CLASS_ART_COUNT']] = scaler.fit_transform(article_data_std[['ACTUAL_PURCHASE_PRICE_EUR_AMT',
                'CATALOGUE_SALES_PRICE_EUR_AMT','VAT_PERCENTAGE','SPECIFIC_CUSTOMER_CLASS_SALES_MEAN','COUNTRY_SALES_CLASS_MEAN', 'CLASS_ART_COUNT']])


# In[23]:


# Sprawdzam jak wyglądają dane po standaryzacji.
article_data_std.head()


# In[24]:


# Ze zbioru usuwam kolejne zmienne oraz powielające się pary zmiennych kategorycznych (np."EU_BRAND_CODE" = "EU_BRAND_NAME")

article_data_cleaned = article_data_std[['ARTICLE_SALES_CONDITION','EU_BRAND_NAME', 'SPGS_DIVISION_NAME','SPGS_DEPARTMENT_NAME',
                                    'SPGS_CLASS_NAME', 'ARTICLE_TYPE','ECO_EASY_FLG', 'SELLING_UOM_CODE', 'ACTUAL_PURCHASE_PRICE_EUR_AMT',
                                    'CATALOGUE_SALES_PRICE_EUR_AMT','SPECIFIC_CUSTOMER_CLASS_SALES_MEAN',
                                    'COUNTRY_SALES_CLASS_MEAN','CLASS_ART_COUNT']]


# In[25]:


# Zamieniam zmienne kategorycnze na zmienne liczbowe za pomocą LabelEncoder

article_data_labled = article_data_cleaned
# Zamieniam zmienną opisową:


article_data_labled['ARTICLE_SALES_CONDITION'] = article_data_labled['ARTICLE_SALES_CONDITION'].replace(['POOR','OK','GOOD'],['0','1','2'])
article_data_labled['ARTICLE_SALES_CONDITION'] = article_data_labled['ARTICLE_SALES_CONDITION'].astype(str).astype(int)
article_data_labled["ARTICLE_SALES_CONDITION"].value_counts()


# In[26]:


# Zamieniam kategoryczne zmienne opisujące:
article_data_labled['EU_BRAND_NAME']= article_data_labled['EU_BRAND_NAME'].cat.codes
article_data_labled['SPGS_DIVISION_NAME'] = article_data_labled['SPGS_DIVISION_NAME'].cat.codes
article_data_labled['SPGS_DEPARTMENT_NAME']= article_data_labled['SPGS_DEPARTMENT_NAME'].cat.codes
article_data_labled['SPGS_CLASS_NAME'] = article_data_labled['SPGS_CLASS_NAME'].cat.codes
article_data_labled['ARTICLE_TYPE'] = article_data_labled['ARTICLE_TYPE'].cat.codes
article_data_labled['ECO_EASY_FLG'] = article_data_labled['ECO_EASY_FLG'].cat.codes
article_data_labled['SELLING_UOM_CODE'] = article_data_labled['SELLING_UOM_CODE'].cat.codes


# In[27]:


article_data_labled.info()


# In[28]:


# # Sprawdzam jak wyglądają dane po zamianie zmiennych kategorycznych.
article_data_labled.head()


# In[38]:


#tworzymy macierze zmiennych objaśniających (predyktorów) i wektor kolumnowy zmiennej objaśnianej
#y = article_data_labled.iloc[:, 0]
y = article_data_labled['ARTICLE_SALES_CONDITION']
y.head()


# In[31]:


X = article_data_labled.iloc[:, 1:]
X.head()


# In[36]:


# Obliczm wspóczynnik korelacji liniowej Pearsona
corr_P = article_data_labled.corr("pearson")
corr_P.shape


# In[37]:


# Tworzę macierz trójkątną i wyświetlamy wspóczynnik korelacji większy od 0.5
corr_P_tri = corr_P.where(np.triu(np.ones(corr_P.shape, dtype=np.bool), k=1)).stack().sort_values()
corr_P_tri


# In[38]:


# Wyświetlamy wspóczynnik korelacji większy od 0.5. 
# Z oczywistych względów największy współczynnik korelacji posiada para "ACTUAL_PURCHASE_PRICE_EUR_AMT"/"CATALOGUE_SALES_PRICE_EUR_AMT"
corr_P_tri[abs(corr_P_tri)>0.5]


# In[39]:


# Obliczm wspóczynnik korelacji liniowej Spearmana
corr_S = article_data_labled.corr("spearman")
corr_S.shape


# In[40]:


corr_S_tri = corr_S.where(np.triu(np.ones(corr_S.shape, dtype=np.bool), k=1)).stack().sort_values()
corr_S_tri


# In[41]:


# Wyświetlamy wspóczynnik korelacji większy od 0.5
corr_S_tri[abs(corr_S_tri)>0.5]


# In[36]:


# Wizualizacja korelacji liniowej Spearmana przy pomocy seaborn pairplot.
sns.pairplot(article_data_labled)
plt.show()


# In[50]:


import sklearn.linear_model
mnk = sklearn.linear_model.LinearRegression()


# In[51]:


mnk.fit(X,y)


# In[52]:


mnk.intercept_


# In[53]:


mnk.coef_


# In[54]:


X.columns


# In[56]:


x_nowy = X.mean().values.reshape(1,-1)+0.001
x_nowy


# In[57]:


X.mean().values.reshape(1,-1)


# In[58]:


mnk.predict(x_nowy)


# In[46]:


X_std = (X-X.mean(axis=0))/X.std(axis=0)
X_std.describe()


# In[61]:


y_std = (y - y.mean())/y.std()


# In[62]:


y_std.describe()


# In[63]:


mnk_std = sklearn.linear_model.LinearRegression()
mnk_std.fit(X_std, y_std)


# In[64]:


pd.Series(np.abs(mnk_std.coef_), index=X.columns.to_list()).round(4).sort_values(ascending=False)


# In[65]:


# Ocena jakości modelu
# porównanie wartości dopasowanych, obliczonych za pomocą modelu z wartościami oryginalnymi
y_pred = mnk.predict(X)
y_pred[0:10]


# In[66]:


y[0:10]


# In[67]:


#współczynnik determinacji R2
mnk.score(X,y)


# In[68]:


sklearn.metrics.r2_score(y, y_pred)


# In[69]:


# MSE
sklearn.metrics.mean_squared_error(y, y_pred)


# In[70]:


# MAE
sklearn.metrics.mean_absolute_error(y, y_pred)


# In[71]:


# MedAE
sklearn.metrics.median_absolute_error(y, y_pred)


# In[73]:


# Dzielę zbiór na próbę uczącą i testową.
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=12345)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[76]:


# Tworzę funkcję, która dopasowuje model regresji liniowej do danej próby
# oraz oblicza miary błędów dopasowania.
def fit_regression(model, X_train, X_test, y_train, y_test):
    
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    r2 = sklearn.metrics.r2_score
    mse = sklearn.metrics.mean_squared_error
    mae = sklearn.metrics.mean_absolute_error
    medae = sklearn.metrics.median_absolute_error
    
    return {
        "r_score_u": r2(y_train, y_train_pred),
        "r_score_t": r2(y_test, y_test_pred),
        "MSE_u": mse(y_train, y_train_pred),
        "MSE_t": mse(y_test, y_test_pred),
        "MAE_u": mae(y_train, y_train_pred),
        "MAE_t": mae(y_test, y_test_pred),
        "MEDAE_u": medae(y_ucz, y_train_pred),
        "MEDAE_t": medae(y_test, y_test_pred)
        
    }


# In[77]:


# Przedstawiam działanie powyższej funkcji oraz jej wyniki.
models = ["Reg. liniowa"]
res = [fit_regression(sklearn.linear_model.LinearRegression(), X_train, X_test, y_train, y_test)]
df_results_errors = pd.DataFrame(res, index=models)


# In[78]:


df_results_errors


# In[82]:


plt.figure(figsize=(14,7))
plt.plot(df_results_errors.MSE_t, label='MSE_t')
plt.plot(df_results_errors.MSE_u, label='MSE_u')
plt.xticks(rotation=45)
plt.legend()
plt.show()


# In[88]:


# Tworzę model drzewa decyzyjnego.
from sklearn import tree
model_tree = sklearn.tree.DecisionTreeClassifier()


# In[89]:


# Uczę model na zbiorze. (Ze względu na specyfikę zbioru, uczę na całym zbiorze)
model_tree.fit(X, y)


# In[90]:



model_tree.score(X, y)


# In[ ]:


# Testuję model na zbiorze testowym.
y_model_tree_prediction = model_tree.predict(X_test)
print(y_predicted)


# In[102]:


# Sprawdzam trafność modelu na podstawie przeprowadzonej predykcji.
print(sklearn.metrics.accuracy_score(y_test,y_model_tree_prediction)*100)
# Trafność modelu to 75%


# In[95]:


# Tworzę las losowy
import sklearn.ensemble
model_random_forest = sklearn.ensemble.RandomForestClassifier(random_state=123)


# In[96]:


# Uczę model na zbiorze. (Ze względu na specyfikę zbioru, uczę na całym zbiorze)
model_random_forest.fit(X, y)


# In[97]:


model_random_forest.score(X, y)


# In[103]:


y_model_random_forest_prediction = model_random_forest.predict(X_test)
print(model_random_forest)


# In[104]:


# Sprawdzam trafność modelu na podstawie przeprowadzonej predykcji.
print(sklearn.metrics.accuracy_score(y_test,y_model_random_forest_prediction)*100)
# Trafność modelu to 95%


# In[ ]:




