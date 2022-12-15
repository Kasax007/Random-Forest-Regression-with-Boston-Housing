import numpy as np                                              #a library for working with numerical data, particularly multi-dimensional arrays
import pandas as pd                                             #a library for working with data in tabular form (similar to a spreadsheet)
import matplotlib.pyplot as plt                                 #a library for generating visualizations from data
import seaborn as sns                                           #a library built on top of matplotlib that provides additional functionality for generating visualizations
from sklearn.metrics import mean_squared_error                  #a library for machine learning in Python, which provides functions for training, testing, and evaluating machine learning models
from sklearn.model_selection import cross_val_score
from collections import Counter
from IPython.core.display import display, HTML                  #a library that provides tools for working with the IPython kernel, which is a package for running Python code interactively in a web-based environment
import streamlit as st                                          #a library for building interactive web applications in Python
sns.set_style('darkgrid')                                       #The code then sets the style for visualizations generated using the seaborn library.


with st.spinner('Preparing Visuals and calculating Regression Model'):  #The with st.spinner line wraps the rest of the code in a block that will display a spinner while the code is running, indicating that it is in progress.
    def regression():                                                   #defines a function called regression
        data_url = "http://lib.stat.cmu.edu/datasets/boston"            #loads a data set from a URL
        raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None) #processes the data into a format that cuts of the starting description 
        data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])  #processes the data into a format that is suitable for analysis
        dataexample = data[0:11]                                            #defines the dataexample to be the first few lines of the normal data
        target = raw_df.values[1::2, 2]                                     #defines which values forms the target aka. the MEDV values
        targetexample = target[0:11]                                        #defines the targetexample to be the first few lines of the normal target
        global boston_dataset                                               #defines the global value boston_dataset
        boston_dataset = raw_df                                             #defines the variable to be the raw dataframe
        global dataset                                                      #defines the global value dataset
        dataset = pd.DataFrame(data, columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"])       #creates the dataset and labels the columns
        dataset['MEDV'] = target                                            #defines what the MEDV is supposed to be 
        global datasetexample                                               #defines the global variable datasetexample
        datasetexample = pd.DataFrame(dataexample, columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"])     #creates the example dataset and labels the columns
        datasetexample['MEDV'] = targetexample                              #defines what the MEDV is supposed to be 

        X = dataset.iloc[:, 0:13].values                                    #does the same for a x variable, resembeling the first 13 values 
        y = dataset.iloc[:, 13].values.reshape(-1,1)                        #does the same for a x variable, resembeling the last value --> MEDV
        from sklearn.model_selection import train_test_split                #imports the train test split module 
        #global ratio
        #ratio = st.slider('Train-Test-Split Ratio', 0.01, 1.0, step=0.01)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 25)       #defines the train test split with values and ratios 
        print("Shape of X_train: ", X_train.shape)                          #prints the shape of the table for each variable 
        print("Shape of X_test: ", X_test.shape)
        print("Shape of y_train: ",y_train.shape)
        print("Shape of y_test",y_test.shape)

        # Feature Scaling
        from sklearn.preprocessing import StandardScaler                    #This code performs a preprocessing step called standardization, which is a common technique in machine learning
        sc_X = StandardScaler()                                             #Standardization involves scaling the data so that it has a mean of zero and a standard deviation of one
        sc_y = StandardScaler()                                             #This is useful because many machine learning algorithms perform better when the data is in this form
        X_scaled = sc_X.fit_transform(X_train)                              #The fit_transform method is called on these objects to fit the standardization parameters to the data 
        y_scaled = sc_y.fit_transform(y_train.reshape(-1,1))                #(i.e., to compute the mean and standard deviation of the data) and to apply the standardization to the data

        # Fitting the Random Forest Regression to the dataset
        from sklearn.ensemble import RandomForestRegressor                  #Imports the Random Forest Regressor 
        #Selected best fitting arguments as listed in https://towardsdatascience.com/predicting-housing-prices-using-a-scikit-learns-random-forest-model-e736b59d56c5
        global regressor_rf                                                 #defines the variable regressor_rf 
        regressor_rf = RandomForestRegressor(n_estimators = 640, random_state = 0, min_samples_leaf=1, max_features=0.5, bootstrap=False) #defines the variable to be the Random Forest Regressor 
        regressor_rf.fit(X_train, y_train.ravel())                                                                                        #Fits the train values to the regressor 

        from sklearn.metrics import r2_score                                #imports the module r2 score 
    
        # Predicting Cross Validation Score
        global cv_rf                                                        #defines a couple of global values 
        global rmse_rf
        global r2_score_rf_train
        global r2_score_rf_test
        cv_rf = cross_val_score(estimator = regressor_rf, X = X_scaled, y = y_train.ravel(), cv = 10)   #calculates the cross validation score with the regressor and the x and y values 

        # Predicting R2 Score the Train set results
        y_pred_rf_train = regressor_rf.predict(X_train)
        r2_score_rf_train = r2_score(y_train, y_pred_rf_train)

        # Predicting R2 Score the Test set results
        y_pred_rf_test = regressor_rf.predict(X_test)
        r2_score_rf_test = r2_score(y_test, y_pred_rf_test)

        # Predicting RMSE the Test set results
        rmse_rf = (np.sqrt(mean_squared_error(y_test, y_pred_rf_test)))

        print('CV: ', cv_rf.mean())                                         #prints the different scores into the output 
        print('R2_score (train): ', r2_score_rf_train)
        print('R2_score (test): ', r2_score_rf_test)
        print("RMSE: ", rmse_rf)

    @st.experimental_memo                                                   #chaces the upcoming function so it doesnt have to be recalculated 
    def crossvalidation():                                                  #defines a function called corssvalidation 
        corr = dataset.corr()                                               #defines a variable called corr that makes use of the dataset 
        #Plot figsize
        fig, ax = plt.subplots(figsize=(10, 10))                            #Plots the correlation of the different values with plots and a heatmap 
        #Generate Heat Map, allow annotations and place floats in map
        sns.heatmap(corr, cmap='RdBu', annot=True, fmt=".2f")
        #Apply xticks
        plt.xticks(range(len(corr.columns)), corr.columns)
        #Apply yticks
        plt.yticks(range(len(corr.columns)), corr.columns)
        #show plot
        #plt.show()
        #sns.pairplot(dataset)
        #plt.show()

    @st.experimental_memo                                                   #chaces the upcoming function so it doesnt have to be recalculated 
    def streamlit():                                                        #defines a funtion called streamlit 
        # Streamlit part
        st.title('Random forest regression model for housing prices')       #Writes a lot into streamlit 
        st.text('This is a web app to allow exploration of Datasets')
        st.text('''The Boston house-price data of Harrison, D. and Rubinfeld, D.L.'Hedonic\nprices and the demand for clean air', J. Environ. Economics & Management,\nvol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics\n...', Wiley, 1980.   N.B. Various transformations are used in the table on\npages 244-261 of the latter.''')
        st.text(' Variables in order:')
        st.text('CRIM     per capita crime rate by town')
        st.text('ZN       proportion of residential land zoned for lots over 25,000 sq.ft.')
        st.text('INDUS    proportion of non-retail business acres per town')
        st.text('CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)')
        st.text('NOX      nitric oxides concentration (parts per 10 million)')
        st.text('RM       average number of rooms per dwelling')
        st.text('AGE      proportion of owner-occupied units built prior to 1940')
        st.text('DIS      weighted distances to five Boston employment centres')
        st.text('RAD      index of accessibility to radial highways')
        st.text('TAX      full-value property-tax rate per $10,000')
        st.text('PTRATIO  pupil-teacher ratio by town')
        st.text('B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town')
        st.text('LSTAT    percent lower status of the population')
        st.text('MEDV     Median value of owner-occupied homes in $1000s')
        st.header('Random forest example data')
        st.table(datasetexample)                                            #shows the datasetexample values in the form of a datatable 

        st.header('Heatmap of Correlation')
        st.pyplot(plt)                                                      #displays the heatmap of the correlation 

        st.header('Correlation Plots')

        st.pyplot(sns.pairplot(dataset))                                    #displays the pairplots 

        st.header('Evaluation of the Random Forest Regression Model')       #displays the evaluation scores
        st.write('CV: ', cv_rf.mean())
        st.write('R2_score (train): ', r2_score_rf_train)
        st.write('R2_score (test): ', r2_score_rf_test)
        st.write("RMSE: ", rmse_rf)

    if st.button('Clear Streamlit Cache'):                                  #adds a button that clears the cache and forces a rerun to recalculate values or fix errors
        st.experimental_memo.clear()

    regression()                                                            #runs the regression function

    if st.button('Run Script'):                                             #adds a button that upon beeing pressed runs the corssvalidation and streamlit function
        crossvalidation()
        streamlit()
    
    st.header('Defining Maximum and Minimum Values')                        #displays the maximum and minimum values of each of the 14 variables 
    #st.table(dataset.style.highlight_max(axis=0).highlight_min(axis=0,color='red'))
    maxValue = dataset.max()
    minValue = dataset.min()
    extremes = maxValue, minValue
    mx = pd.DataFrame(maxValue)
    mn = pd.DataFrame(minValue)
    #st.table(extremes)
    ex = mx.merge(mn, left_index=True, right_index=True)
    st.table(ex)



    st.header('Value Selection')
    crimmin = mn[0].values[0]
    crimmax = mx[0].values[0]
    crim = st.slider('per capita crime rate by town', int(crimmin)+0.01-0.01, int(crimmax)+0.01-0.01, (int(crimmax)+int(crimmin)+0.01-0.01)/2, step=0.01)

    znmin = mn[0].values[1]
    znmax = mx[0].values[1]
    zn = st.slider('proportion of residential land zoned for lots over 25,000 sq.ft.', int(znmin)+0.01-0.01, int(znmax)+0.01-0.01, (int(znmax)+int(znmin)+0.01-0.01)/2, step=0.01)

    indusmin = mn[0].values[2]
    indusmax = mx[0].values[2]
    indus = st.slider('proportion of non-retail business acres per town', int(indusmin)+0.01-0.01, int(indusmax)+0.01-0.01, (int(indusmax)+int(indusmin)+0.01-0.01)/2, step=0.01)
    
    chasmin = mn[0].values[3]
    chasmax = mx[0].values[3]
    chas = st.slider('Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)', int(chasmin)+0.01-0.01, int(chasmax)+0.01-0.01, (int(chasmax)+int(chasmin)+0.01-0.01)/2, step=0.01)

    noxmin = mn[0].values[4]
    noxmax = mx[0].values[4]
    nox = st.slider('nitric oxides concentration (parts per 10 million)', int(noxmin)+0.01-0.01, int(noxmax)+1+0.01-0.01, (int(noxmax)+0.01-0.01+1.0)/2, step=0.01)

    rmmin = mn[0].values[5]
    rmmax = mx[0].values[5]
    rm = st.slider('average number of rooms per dwelling', int(rmmin)+0.01-0.01, int(rmmax)+0.01-0.01, (int(rmmax)+int(rmmin)+0.01-0.01)/2, step=0.01)

    agemin = mn[0].values[6]
    agemax = mx[0].values[6]
    age = st.slider('proportion of owner-occupied units built prior to 1940', int(agemin)+0.01-0.01, int(agemax)+0.01-0.01, (int(agemax)+int(agemin)+0.01-0.01)/2, step=0.01)

    dismin = mn[0].values[7]
    dismax = mx[0].values[7]
    dis = st.slider('weighted distances to five Boston employment centres', int(dismin)+0.01-0.01, int(dismax)+0.01-0.01, (int(dismax)+int(dismin)+0.01-0.01)/2, step=0.01)

    radmin = mn[0].values[8]
    radmax = mx[0].values[8]
    rad = st.slider('index of accessibility to radial highways', int(radmin)+0.01-0.01, int(radmax)+0.01-0.01, (int(radmax)+int(radmin)+0.01-0.01)/2, step=0.01)

    taxmin = mn[0].values[9]
    taxmax = mx[0].values[9]
    tax = st.slider('full-value property-tax rate per $10,000', int(taxmin)+0.01-0.01, int(taxmax)+0.01-0.01, (int(taxmax)+int(taxmin)+0.01-0.01)/2, step=0.01)

    ptratiomin = mn[0].values[10]
    ptratiomax = mx[0].values[10]
    ptratio = st.slider('pupil-teacher ratio by town', int(ptratiomin)+0.01-0.01, int(ptratiomax)+0.01-0.01, (int(ptratiomax)+int(ptratiomin)+0.01-0.01)/2, step=0.01)

    bmin = mn[0].values[11]
    bmax = mx[0].values[11]
    b = st.slider('1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town', int(bmin)+0.01-0.01, int(bmax)+0.01-0.01, (int(bmax)+int(bmin)+0.01-0.01)/2, step=0.01)

    lstatmin = mn[0].values[12]
    lstatmax = mx[0].values[12]
    lstat = st.slider('percent lower status of the population', int(lstatmin)+0.01-0.01, int(lstatmax)+0.01-0.01, (int(lstatmax)+int(lstatmin)+0.01-0.01)/2, step=0.01)

    calcdata = {"CRIM": [crim], "ZN": [zn], "INDUS": [indus], "CHAS": [chas], "NOX": [nox], "RM": [rm], "AGE": [age], "DIS": [dis], "RAD": [rad], "TAX": [tax], "PTRATIO": [ptratio], "B": [b], "LSTAT": [lstat]}
    calcset = pd.DataFrame(data = calcdata)
    prediction = regressor_rf.predict(calcset)
    price = prediction[0]*1000
    st.write('With the given input parameters the random forest algorythm has determined that the home costs: ', price, '$.')


