
# coding: utf-8

# In[10]:


import pandas as pandas
import numpy as numpy
import os
import matplotlib.pyplot as matplot
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import itertools
from IPython.display import Image  
from sklearn import tree
from os import system
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.grid_search import GridSearchCV
numpy.random.seed(1234)
RandomState = numpy.random.seed(1234)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')
import scipy.stats as stats
from scipy.stats import chi2_contingency

# In[8]:


class Perform_EDA():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        
    def EDA_Corr(df):
        """This gives output as Covariance matrix and feature wise uniquess i.e how much its statistically
        independent. This is done with default range of corr between +0.5 to -0.6"""
        corr = df.corr()
        index = corr.columns
        Output = []
        for i in range(0,len(index)):
            i = index[i]
            Pos = corr.index[(corr[i] >= 0.5)].tolist()
            No = corr.index[(corr[i] < 0.5) & (corr[i] > -0.6)].tolist()
            Neg = corr.index[(corr[i] <= -0.5)].tolist()
            leng_u = len(No)
            leng_pos = len(Pos)
            leng_neg = len(Neg)
            Out = [i, leng_u, leng_pos, leng_neg, Pos, Neg, No]
            Output.append(Out)
        fig, ax = matplot.subplots(figsize=(20,10))  
        sns.heatmap(corr,annot=True,vmin=-1,vmax=1,cmap='Blues', linewidths=0, ax = ax)
        Output1 = pandas.DataFrame(Output, columns= ['Feature','Uniqueness','Positive rel', 'inverse rel', 'Pos', 'Neg', 'No'])
        return Output1

    def EDA(df):
        """This function creates a dataframe transpose with 5 point summary, Kurtosis, Skewness, IQR, Range
        Total count of missing values, % missing values against the total records"""
        EDA = pandas.DataFrame((df.describe()).T)
        EDA["Kurtosis"] = df.kurtosis()
        EDA["Skewness"] = df.skew()
        EDA["Range"] = EDA['max'] -  EDA['min']
        EDA["IQR"] = EDA['75%'] -  EDA['25%']
        EDA["Missing Values"] = df.shape[0] - EDA["count"]
        print("Total Missing Values = ", EDA['Missing Values'].sum(), "Data Points, Contributing to ", 
              (round(((EDA['Missing Values'].sum())/len(df)),2))*100,"%")
        print("Columns with values as 0\n\n",pandas.Series((EDA.loc[EDA['min'] == 0]).index),'\n')
        indices = EDA[EDA['min'] == 0].index
        print("\nColumns with numnber of Zeros\n")
        for i in range(0,len(indices)):
            j = indices[i]
            print(j,"   =",(df[j].value_counts())[0])
        return EDA
    
    def NaN_treatment_median(df):
        EDA_Summary = Perform_EDA.EDA(df)
        Missing_Value_Columns = pandas.Series((EDA_Summary.loc[EDA_Summary['Missing Values'] != 0]).index)
        len(Missing_Value_Columns)
        for i in range(0,len(Missing_Value_Columns)):
            df[Missing_Value_Columns[i]].fillna(df[Missing_Value_Columns[i]].median(), inplace = True)
        return df
    
    def NaN_treatment_mean(df):
        EDA_Summary = Perform_EDA.EDA(df)
        Missing_Value_Columns = pandas.Series((EDA_Summary.loc[EDA_Summary['Missing Values'] != 0]).index)
        len(Missing_Value_Columns)
        for i in range(0,len(Missing_Value_Columns)):
            df[Missing_Value_Columns[i]].fillna(df[Missing_Value_Columns[i]].mean(), inplace = True)
        return df
    
    def Zero_Values_Treatment_Median(df):
        EDA_Summary = Perform_EDA.EDA(df)
        Zero_Value_Columns = pandas.Series((EDA_Summary.loc[EDA_Summary['min'] == 0]).index)
        for i in range(0,len(Zero_Value_Columns)):
            df[Zero_Value_Columns[i]].replace(0,(df[Zero_Value_Columns[i]]).median(), inplace = True)
        return df
    
    def Zero_Values_Treatment_Mean(df):
        EDA_Summary = Perform_EDA.EDA(df)
        Zero_Value_Columns = pandas.Series((EDA_Summary.loc[EDA_Summary['min'] == 0]).index)
        for i in range(0,len(Zero_Value_Columns)):
            df[Zero_Value_Columns[i]].replace(0,(df[Zero_Value_Columns[i]]).mean(), inplace = True)
        return df

    def Missing_Values(df):
        for i in range(0,len(df.columns)):
            i = df.columns[i]
            print(i,'\n\n',df[i].value_counts(),'\n\n',(df[i].value_counts()).sum(),'\n\n')
        return Missing_Values
    
    def EDA_target(df,Y):
        DF = pandas.DataFrame(Y.value_counts())
        DF['Contribution'] = round(((DF['class'])/len(df)*100),2)
        Missing_values = len(df) - (Y.value_counts()).sum()
        print("Total Missing Values = ", Missing_values, "Data Points, Contributing to ",
              round(((Missing_values/len(df))*100),2),"%")
        return DF
    
    def univariate_plots(Source):
        print("Columns that are int32,int64 = ",Source.select_dtypes(include=['int32','int64']).columns)
        print("Columns that are flaot32,float64 = ",Source.select_dtypes(include=['float64']).columns)
        print("Columns that are objects = ",Source.select_dtypes(include=['object']).columns)
        a = pandas.Series(Source.select_dtypes(include=['int32','int64']).columns)
        leng = len(a)
        for j in range(0,len(a)):
            f, axes = matplot.subplots(1, 2, figsize=(10, 10))
            sns.boxplot(Source[a[j]], ax = axes[0])
            sns.distplot(Source[a[j]], ax = axes[1])
            matplot.subplots_adjust(top =  1.5, right = 10, left = 8, bottom = 1)

        a = pandas.Series(Source.select_dtypes(include=['float64']).columns)
        leng = len(a)
        for j in range(0,len(a)):
            matplot.Text('Figure for float64')
            f, axes = matplot.subplots(1, 2, figsize=(10, 10))
            sns.boxplot(Source[a[j]], ax = axes[0])
            sns.distplot(Source[a[j]], ax = axes[1])
            matplot.subplots_adjust(top =  1.5, right = 10, left = 8, bottom = 1)

        a = pandas.Series(Source.select_dtypes(include=['object']).columns)
        leng = len(a)
        for j in range(0,len(a)):
            matplot.subplots()
            sns.countplot(Source[a[j]])
            
    def Impute_Outliers(df,method = "median",threshold = 0.1):
            """Pls input the method as a string - mean or median with 'm' in lower case
            Default method = median
            Detault threshold = 0.1
            The function will give 3 outputs 
            1. df data imputed based on the value provided
            2. Outlier impact as a printed message with % of records impacted
            """
            df_Columns = df.columns
            Subset_Columns = pandas.Series(df.select_dtypes(include=['int32','int64','float64','float32']).columns)

            Subset = df[Subset_Columns]

            IQR = Subset.quantile(0.75) - Subset.quantile(0.25)

            Q3_values = Subset.quantile(0.75) + (1.5 * IQR)
            Q1_values = Subset.quantile(0.25) - (1.5 * IQR)

            Q1 = []
            for i in range(1,len(Subset_Columns)+1):
                c = "Q1"+str(i)
                Q1.append(c)

            Q3 = []
            for i in range(1,len(Subset_Columns)+1):
                c = "Q3"+str(i)
                Q3.append(c)

            df[Q3] = Subset > Q3_values[0:len(Subset_Columns)]
            df[Q1] = Subset < Q1_values[0:len(Subset_Columns)]

            Q1_Outliers = []
            Q1_j = []
            Q3_Outliers = []
            Q3_j = []
            for i in range(0,len(Q1)):
                i = Q1[i]
                No = df.shape[0] - df[i].value_counts()[0]
                Q1_Outliers.append(No)
                Q1_j.append(i)
            Q1_Col = pandas.DataFrame(Q1_j, columns=["Q1"])
            Q1_outliers = pandas.DataFrame(Q1_Outliers, columns=["Q1 Outliers"])
            Outliers_impact_Q1 = Q1_Col.join(Q1_outliers)

            for i in range(0,len(Q3)):
                i = Q3[i]
                No = df.shape[0] - df[i].value_counts()[0]
                Q3_Outliers.append(No)
                Q3_j.append(i)
            Q3_Col = pandas.DataFrame(Q3_j, columns=["Q3"])
            Q3_outliers = pandas.DataFrame(Q3_Outliers, columns=["Q3 Outliers"])
            Outliers_impact_Q3 = Q3_Col.join(Q3_outliers)

            Outliers_impact = Outliers_impact_Q1['Q1 Outliers']+Outliers_impact_Q3['Q3 Outliers']
            Outliers_impact = (pandas.DataFrame(Subset_Columns, columns=["Column Name"])).join(pandas.DataFrame(Outliers_impact, columns=["No of Outliers"]))
            print(Outliers_impact)


            aij = []
            for i in range(0,len(Q3)):
                i = Q3[i]
                bij = ((pandas.DataFrame(df[i])).index[(df[i] == True)].tolist())
                aij = aij + bij
            Q3_indices = (pandas.Series(aij)).value_counts()


            cij = []
            for i in range(0,len(Q1)):
                i = Q1[i]
                dij = ((pandas.DataFrame(df[i])).index[(df[i] == True)].tolist())
                cij = cij + dij
            Q1_indices = (pandas.Series(cij)).value_counts()

            print("No of records impacted by Outliers = ",round((Outliers_impact['No of Outliers'].sum() / len(df)),2)*100,"%")
            print("No of records in outliers beyond Q4 = ",round(((pandas.DataFrame(Q3_Outliers)[0]).sum() / len(df)),2)*100,"%")
            print("No of records in outliers beyond Q1 = ",round(((pandas.DataFrame(Q1_Outliers)[0]).sum() / len(df)),2)*100,"%")

            if (round((Outliers_impact['No of Outliers'].sum() / len(df)),2)) <= threshold :
                print((round((Outliers_impact['No of Outliers'].sum() / len(df)),2)*100)," ",threshold)
                Outliers_Q3_Q1 = pandas.DataFrame(Q3_values, columns = ['Q3_values']).join(pandas.DataFrame(Q1_values, columns=['Q1_values']))
                for i in range(0,len(Subset_Columns)):
                    Q3 = ((Outliers_Q3_Q1).T)[Subset_Columns[i]].loc['Q3_values']
                    Q1 = ((Outliers_Q3_Q1).T)[Subset_Columns[i]].loc['Q1_values']
                    df.loc[df[Subset_Columns[i]] > Q3, Subset_Columns[i]] = numpy.nan
                    df.loc[df[Subset_Columns[i]] < Q1, Subset_Columns[i]] = numpy.nan
                    if method == "median":
                        median1 = ((df.loc[(df[Subset_Columns[i]]<((((Outliers_Q3_Q1).T)[Subset_Columns[i]])['Q3_values'])) & 
                         (df[Subset_Columns[i]]>((((Outliers_Q3_Q1).T)[Subset_Columns[i]])['Q1_values']))])[Subset_Columns[i]]).median()
                    else:
                        median1 = ((df.loc[(df[Subset_Columns[i]]<((((Outliers_Q3_Q1).T)[Subset_Columns[i]])['Q3_values'])) & 
                         (df[Subset_Columns[i]]>((((Outliers_Q3_Q1).T)[Subset_Columns[i]])['Q1_values']))])[Subset_Columns[i]]).mean()
                df.replace(numpy.nan,median1,inplace= True)
                print("No of records imputed using the",method,"is",Outliers_impact['No of Outliers'].sum())
                df = df.iloc[:,0:len(df_Columns)]
                return df    
            else:
                print((round((Outliers_impact['No of Outliers'].sum() / len(df)),2))," ",threshold)
                print("Too many outliers, please alter the 'threshold' if outliers will have to be treated")
                df = df.iloc[:,0:len(df_Columns)]
                return df
        
class ChiSquare:
    def __init__(self, dataframe):
        self.df = dataframe
        self.p = None #P-Value
        self.chi2 = None #Chi Test Statistic
        self.dof = None
        
        self.dfObserved = None
        self.dfExpected = None
        
    def _print_chisquare_result(self, colX,alpha):
        result = ""
        if self.p<alpha:
            result="{0} exhibits multicollinearity. (Consider Discarding {0} from model)".format(colX)
        else:
            result="{0} is NOT related and can be a good predictor".format(colX)

        print(result)
        
    def TestIndependence(self,colX,colY,alpha=0.05):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)
        
        self.dfObserved = pandas.crosstab(Y,X) 
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof 
        
        self.dfExpected = pandas.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)
        #print(self.dfExpected)
        self._print_chisquare_result(colX,alpha)
        return self.dfExpected


class Evaluate_Model():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
    
    
    def plot_confusion_matrix(Y_test,Y_predict, target_names,title='Confusion matrix',cmap=None,normalize=True):
        cm = metrics.confusion_matrix(Y_test, Y_predict)
        accuracy = numpy.trace(cm) / float(numpy.sum(cm))
        misclass = 1 - accuracy

        if cmap is None:
            cmap = matplot.get_cmap('Blues')

        matplot.figure(figsize=(8, 6))
        matplot.imshow(cm, interpolation='nearest', cmap=cmap)
        matplot.title(title)
        matplot.colorbar()

        if target_names is not None:
            tick_marks = numpy.arange(len(target_names))
            matplot.xticks(tick_marks, target_names, rotation=45)
            matplot.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]


        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                matplot.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                matplot.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")


        matplot.tight_layout()
        matplot.ylabel('True label')
        matplot.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        matplot.show()
        print(metrics.classification_report(Y_test, Y_predict))
        model_performance = [metrics.accuracy_score(Y_test, Y_predict),metrics.recall_score(Y_test, Y_predict, average = 'macro'),
                             metrics.precision_score(Y_test, Y_predict, average = 'macro'),metrics.f1_score(Y_test, Y_predict, average = 'macro') ]
        accuracy_report = pandas.DataFrame(model_performance, columns=['Model_Performance'], 
                                       index=['Accuracy','Recall','Precision','f1_Score'])
        return accuracy_report
    
    
    def evaluate_PCA(df,y,n):
        """ df = Data Frame, y = target column name and n = number of columns Xs as int value 
        This function will use Eigen value decomposition method to determine PCs
        and returns a df with PCs and Cumulative frequecny with elbow chart
        
        This function uses StandardScaler() as default function"""
        X = df.iloc[:,0:n].values
        y = df[y].values
        X_Scaled = StandardScaler().fit_transform(X)
        cov_matrix = numpy.cov(X_Scaled.T)
        #print('Covariance Matrix \n%s', cov_matrix)

        e_vals, e_vecs = numpy.linalg.eig(cov_matrix)
        e_vals, e_vecs = numpy.linalg.eig(cov_matrix)
        #print('Eigenvectors \n%s' %e_vecs)
        #print('\nEigenvalues \n%s' %e_vals)

        tot = sum(e_vals)
        var_exp = [( i /tot ) * 100 for i in sorted(e_vals, reverse=True)]
        cum_var_exp = numpy.cumsum(var_exp)
        #print("Cumulative Variance Explained", cum_var_exp)

        matplot.figure(figsize=(10 , 5))
        matplot.bar(range(1, e_vals.size + 1), var_exp, alpha = 0.5, align = 'center', label = 'Individual explained variance')
        matplot.step(range(1, e_vals.size + 1), cum_var_exp, where='mid', label = 'Cumulative explained variance')
        matplot.ylabel('Explained Variance Ratio')
        matplot.xlabel('Principal Components')
        matplot.legend(loc = 'best')
        matplot.tight_layout()
        matplot.show()

        Pricipal_comp_composition = (pandas.DataFrame(cum_var_exp).reset_index())
        Pricipal_comp_composition.columns = ['Pricipal Components', '% info retained']
        return Pricipal_comp_composition
    

class Build_Model():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        
    def GS_DT(X_train, X_test, Y_train, Y_test,max_depth_start,max_depth_end,max_depth_jump,max_leaf_nodes_start,
          max_leaf_nodes_end,max_leaf_nodes_jump, cv_count, target_names):
        """ The function creates a decision tree by taking in all the parameters and follows below
        1. Performs GridSearch and cross validation to determine the best parameters
        2. Chooses the parameter that provides the best result
        3. Creates a Tree along with visualization as a json structure using .dot file format
        
        Output of the function
        1. depth : Grid Search CV chosen depth parameter
        2. leaf_node : Grid Search chosen leaf_nodes count
        3. GS parameters
        4. regularized model"""
        max_depth = numpy.arange(max_depth_start,max_depth_end,max_depth_jump)
        max_leaf_nodes = numpy.arange(max_leaf_nodes_start,max_leaf_nodes_end,max_leaf_nodes_jump)
        dt_Model = DecisionTreeClassifier(criterion = 'entropy', random_state=RandomState)
        parameters = {'max_depth': max_depth, 'max_leaf_nodes':max_leaf_nodes}
        dt_GS = GridSearchCV(dt_Model,parameters,cv=cv_count)
        #Y_train_1d = (Y_train.values).reshape(((Y_train.shape)[0]),)
        dt_GS.fit(X_train,Y_train)
        dt_GS.predict(X_test)
        depth = (pandas.Series(dt_GS.best_params_))[0]
        leaf_node = (pandas.Series(dt_GS.best_params_))[1]
        dt_model_reg = DecisionTreeClassifier(criterion = 'entropy', max_depth = depth, max_leaf_nodes = leaf_node, random_state= RandomState )
        print(dt_model_reg)
        dt_model_reg.fit(X_train, Y_train)
        Tree_File = open('Decision_Tree.dot','w')
        dot_data = tree.export_graphviz(dt_model_reg, out_file=Tree_File, feature_names = list(X_train), class_names = list(target_names))
        Tree_File.close()
        print (pandas.DataFrame(dt_model_reg.feature_importances_, columns = ["Imp"], index = X_train.columns))
        Y_predict = dt_model_reg.predict(X_test)
        dt_model_reg1 = Evaluate_Model.plot_confusion_matrix(Y_test,Y_predict, target_names,title='Confusion matrix',cmap=None,normalize=True)
        dt_model_reg1 = dt_model_reg1.rename(columns={"Model_Performance" : "Decision Tree Regularized" })
        dt_model_reg1
        return depth, leaf_node, dt_GS.get_params,dt_model_reg

    def Log_model(X_train, X_test, Y_train, Y_test):
        logmodel = LogisticRegression()
        Log_model = logmodel.fit(X_train,Y_train)
        Y_predict = logmodel.predict(X_test)
        Log_model_scores = Evaluate_Model.plot_confusion_matrix(Y_test,Y_predict, target_names,title='Confusion matrix',cmap=None,normalize=True)
        Log_model_scores = Log_model_scores.rename(columns={"Model_Performance" : "Logistic Regression" })
        Log_model_scores
        return Log_model_scores

    def Gau_NB_model(X_train, X_test, Y_train, Y_test):
        NBmodel = GaussianNB()
        Gau_NB_model = NBmodel.fit(X_train,Y_train)
        Y_predict = NBmodel.predict(X_test)
        Gau_NB_model_scores = Evaluate_Model.plot_confusion_matrix(Y_test,Y_predict, target_names,title='Confusion matrix',cmap=None,normalize=True)
        Gau_NB_model_scores = Gau_NB_model_scores.rename(columns={"Model_Performance" : "Gaussian NB" })
        Gau_NB_model_scores
        return Gau_NB_model_scores

    def GS_KNN(X_train,Y_train,X_test,Y_test, n_neighbors_start, n_neighbors_end, n_neighbors_jump,cv_count):
        X_train = preprocessing.scale(X_train)
        X_test = preprocessing.scale(X_test)
        k = numpy.arange(n_neighbors_start, n_neighbors_end, n_neighbors_jump)
        KNN = KNeighborsClassifier()
        parameters = {'n_neighbors': k}
        GS_KNN = GridSearchCV(KNN,parameters,cv=cv_count)
        Y_train_1d = (Y_train.values).reshape(((Y_train.shape)[0]),)
        GS_KNN.fit(X_train,Y_train_1d)
        GS_KNN.predict(X_test)
        n_neighbors = (pandas.Series(GS_KNN.best_params_))[0]
        KNN_model = KNeighborsClassifier(n_neighbors = n_neighbors)
        print(KNN_model)
        KNN_model.fit(X_train, Y_train)
        Y_predict = KNN_model.predict(X_test)
        KNN_model_Scores = Evaluate_Model.plot_confusion_matrix(Y_test,Y_predict, target_names,title='Confusion matrix',cmap=None,normalize=True)
        KNN_model_Scores = KNN_model_Scores.rename(columns={"Model_Performance" : "K Nearest Neighbors" })
        KNN_model_Scores
        return n_neighbors, GS_KNN.get_params,KNN_model_Scores

    def GS_RandomForest(X_train,Y_train,X_test,Y_test, n_estimator_start, n_estimator_end, n_estimator_jump,cv_count,target_names):
        """ The function creates a Random Forest tree by taking in all the parameters and follows below
        1. Performs GridSearch and cross validation to determine the best parameters
        2. Chooses the parameter that provides the best result
        3. Creates random forest trees
        
        Output of the function
        1. n_estimator : Grid Search CV chosen number of trees parameter
        2. Random forest parameters
        3. RF"""        
        n = numpy.arange(n_estimator_start, n_estimator_end, n_estimator_jump)
        RF = RandomForestClassifier(random_state=RandomState)
        parameters = {'n_estimators': n}
        GS_RandomForest = GridSearchCV(RF,parameters,cv=cv_count)
        #Y_train_1d = (Y_train.values).reshape(((Y_train.shape)[0]),)
        GS_RandomForest.fit(X_train,Y_train)
        GS_RandomForest.predict(X_test)
        n_estimator = (pandas.Series(GS_RandomForest.best_params_))[0]
        RF = RandomForestClassifier(n_estimators = n_estimator, random_state=RandomState)
        print(RF)
        RF.fit(X_train, Y_train)
        Y_predict = RF.predict(X_test)
        RF_Scores = Evaluate_Model.plot_confusion_matrix(Y_test,Y_predict, target_names,title='Confusion matrix',cmap=None,normalize=True)
        RF_Scores = RF_Scores.rename(columns={"Model_Performance" : "Random Forest" })
        RF_Scores
        return n_estimator, GS_RandomForest.get_params,RF
    
    def GS_SVC(X_train ,Y_train ,X_test ,Y_test ,target_names,cv_count = 10 ,C = None, C_start = None, C_end = None, C_jump = None ,Kernel = 'linear'):
        """ This function creates SVC using parameters passed with CV defaulted to 10 and Grid Search
        The best parameter is popped out and final model is created.
        The final output will be optimal model popped using grid search"""
        if C != None:
            parameters = {'C' : C}
        else:
            C = numpy.arange(C_start, C_end, C_jump)
            parameters = {'C' : C}    
        SVCl = SVC(parameters,kernel=Kernel,random_state=RandomState)
        GS_SVC = GridSearchCV(SVCl,parameters,cv=2)
        Y_train_1d = (Y_train.values).reshape(((Y_train.shape)[0]),)
        GS_SVC.fit(X_train,Y_train_1d)
        GS_SVC.predict(X_test)
        C = (pandas.Series(GS_SVC.best_params_))[0]
        SVCL = SVC(C = C,kernel = Kernel,random_state=RandomState)
        print(SVC)
        SVCL.fit(X_train, Y_train)
        Y_predict = SVCL.predict(X_test)
        SVC_Scores = Evaluate_Model.plot_confusion_matrix(Y_test,Y_predict, target_names,title='Confusion matrix',cmap=None,normalize=True)
        SVC_Scores = SVC_Scores.rename(columns={"Model_Performance" : "SVC" })
        SVC_Scores
        return SVCL

