import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report \
    ,f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import xgboost as xgb
from generateplots import *


columns = ['Voters_Age', 'Broad Ethnic Groupings', 'Mailing_Addresses_City', 'Mailing_Addresses_State',
           'Mailing_Addresses_Zip', 'Parties_Description', 'Voters_Gender', 'MaritalStatus_Description',
           'Vote_Frequency', 'County', 'PresenceOfChildrenCode', 'Home Owner/Renter', 'EstHomeValue',
           'MedianEducationYears', 'Net Worth', 'EstimatedIncome', 'Education', 'Household Composition',
           'Trend - BLM', 'Trend- Defund the police', 'Trend - Medicare for all',
           'Trend - $15 Minimum wage', 'COVID-19 Mask/No Mask', 'HUD NIMBY', 'Cancel Culture']


def processDfData(df):
    target = 'Parties_Description'

    df[target] = df[target].apply(
        lambda x: 0 if x == 'Likely Democratic' else 1 if x == 'Likely Republican' else 2)

    df = pd.concat([df, pd.get_dummies(df['Voters_Age'], prefix='Voters_Age', prefix_sep=':')], axis=1)
    df.drop('Voters_Age', axis=1, inplace=True)

    df = pd.concat([df, pd.get_dummies(df['Broad Ethnic Groupings'], prefix='Broad Ethnic Groupings',
                                       prefix_sep=':')], axis=1)
    df.drop('Broad Ethnic Groupings', axis=1, inplace=True)

    df = pd.concat([df, pd.get_dummies(df['Voters_Gender'], prefix='Voters_Gender', prefix_sep=':')],
                   axis=1)
    df.drop('Voters_Gender', axis=1, inplace=True)

    df = pd.concat(
        [df, pd.get_dummies(df['MaritalStatus_Description'], prefix='MaritalStatus_Description', prefix_sep=':')],
        axis=1)
    df.drop('MaritalStatus_Description', axis=1, inplace=True)

    df = pd.concat([df, pd.get_dummies(df['PresenceOfChildrenCode'], prefix='PresenceOfChildrenCode', prefix_sep=':')],
                   axis=1)
    df.drop('PresenceOfChildrenCode', axis=1, inplace=True)

    df = pd.concat([df, pd.get_dummies(df['Home Owner/Renter'], prefix='Home Owner/Renter',
                                       prefix_sep=':')], axis=1)
    df.drop('Home Owner/Renter', axis=1, inplace=True)

    df = pd.concat([df, pd.get_dummies(df['EstHomeValue'], prefix='EstHomeValue', prefix_sep=':')], axis=1)
    df.drop('EstHomeValue', axis=1, inplace=True)

    df = pd.concat([df, pd.get_dummies(df['EstimatedIncome'], prefix='EstimatedIncome', prefix_sep=':')], axis=1)
    df.drop('EstimatedIncome', axis=1, inplace=True)

    df = pd.concat([df, pd.get_dummies(df['Education'], prefix='Education', prefix_sep=':')], axis=1)
    df.drop('Education', axis=1, inplace=True)

    df = pd.concat([df, pd.get_dummies(df['Trend - BLM'], prefix='Trend - BLM', prefix_sep=':')], axis=1)
    df.drop('Trend - BLM', axis=1, inplace=True)

    df = pd.concat(
        [df, pd.get_dummies(df['Trend- Defund the police'], prefix='Trend- Defund the police', prefix_sep=':')], axis=1)
    df.drop('Trend- Defund the police', axis=1, inplace=True)
    df = pd.concat(
        [df, pd.get_dummies(df['Trend - Medicare for all'], prefix='Trend - Medicare for all', prefix_sep=':')], axis=1)
    df.drop('Trend - Medicare for all', axis=1, inplace=True)
    df = pd.concat(
        [df, pd.get_dummies(df['Trend - $15 Minimum wage'], prefix='Trend - $15 Minimum wage', prefix_sep=':')], axis=1)
    df.drop('Trend - $15 Minimum wage', axis=1, inplace=True)
    df = pd.concat([df, pd.get_dummies(df['COVID-19 Mask/No Mask'], prefix='COVID-19 Mask/No Mask', prefix_sep=':')],
                   axis=1)
    df.drop('COVID-19 Mask/No Mask', axis=1, inplace=True)
    df = pd.concat([df, pd.get_dummies(df['HUD NIMBY'], prefix='HUD NIMBY', prefix_sep=':')], axis=1)
    df.drop('HUD NIMBY', axis=1, inplace=True)
    df = pd.concat([df, pd.get_dummies(df['Cancel Culture'], prefix='Cancel Culture', prefix_sep=':')], axis=1)
    df.drop('Cancel Culture', axis=1, inplace=True)

    df.drop('Mailing_Addresses_City', axis=1, inplace=True)
    df.drop('Mailing_Addresses_State', axis=1, inplace=True)
    df.drop('Mailing_Addresses_Zip', axis=1, inplace=True)
    df.drop('Vote_Frequency', axis=1, inplace=True)
    df.drop('County', axis=1, inplace=True)
    df.drop('MedianEducationYears', axis=1, inplace=True)
    df.drop('Net Worth', axis=1, inplace=True)
    df.drop('Household Composition', axis=1, inplace=True)

    X = np.array(df.drop([target], 1))
    y = np.array(df[target])

    # Splitting data as train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    df.to_csv('df_new.csv', index=False, header=True)

    return X_train, X_test, y_train, y_test


def generate_model_report(modelName, y_test, predicted_value):
    cm = confusion_matrix(y_test, predicted_value)
    accuracy = accuracy_score(y_test, predicted_value)
    precision = precision_score(y_test, predicted_value, average="macro")
    recall = recall_score(y_test, predicted_value, average="macro")
    f1Score = f1_score(y_test, predicted_value, average="macro")
    print(f"Confusion Matrix for {modelName} :\n{cm}")
    generateConfusionMatrix(cm,accuracy,modelName)
    print(f"Classification report for {modelName} :\n{classification_report(y_test, predicted_value)}")
    print(f"The Accuracy for Election Data {modelName} Model is {format(accuracy)}")
    print(f"The Precision for Election Data {modelName} is {format(precision)}")
    print(f"The Recall for Election Data {modelName} Model is {format(recall)}")
    print(f"F1 Score for Election Data {modelName} Model is {f1Score}", )
    return accuracy, precision, recall ,f1Score


def decision_tree(X_train, X_test, y_train, y_test):
    dtmodel = DecisionTreeClassifier()
    dtmodel.fit(X_train, y_train)
    dt_model_predict = dtmodel.predict(X_test)
    return generate_model_report("Decision Tree", y_test, dt_model_predict)


def randomForest(X_train, X_test, y_train, y_test):
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    rf_model_predict = rf_model.predict(X_test)
    return generate_model_report("Random Forest", y_test, rf_model_predict)


def KNN(X_train, X_test, y_train, y_test):
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train, y_train)
    knn_model_predict = knn_model.predict(X_test)
    return  generate_model_report("KNN", y_test, knn_model_predict)


def svm(X_train, X_test, y_train, y_test):
    svc = SVC(C=1.0, kernel="linear")
    svc.fit(X_train, y_train)
    svc_predict = svc.predict(X_test)
    return generate_model_report("SVM", y_test, svc_predict)


def naive_byes(X_train, X_test, y_train, y_test):
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    naive_pre = nb_model.predict(X_test)
    return generate_model_report("Naive Bayes", y_test, naive_pre)


def xgboost_classifier(X_train, X_test, y_train, y_test):
    XGBClassifier = xgb.XGBClassifier()
    XGBClassifier.fit(X_train, y_train)
    xgb_predict = XGBClassifier.predict(X_test)
    return generate_model_report("XGBoost", y_test, xgb_predict)


def tuneParameterandGenerateModel(modelName, pipe):
    example_params = {
        'sampling__sampling_strategy': ['not minority']
    }
    gsc = GridSearchCV(
        estimator=pipe,
        param_grid=example_params,
        scoring='f1_macro',
        cv=5
    )

    grid_result = gsc.fit(X_train, y_train)
    predicted_value = grid_result.predict(X_test)
    return generate_model_report(modelName, y_test, predicted_value)


if __name__ == '__main__':
    df = pd.read_csv('Dallas_county_output.csv', skiprows=1, names=columns)
    X_train, X_test, y_train, y_test = processDfData(df)

    benchmark_overall_accuracy_score = []
    benchmark_overall_recall_score = []
    benchmark_overall_precision_score = []
    benchmark_overall_f1_score = []
    # Bench mark Values for all algorithms
    print("Benchmark Values for each algorithm")
    accuracy_DT, recall_DT, precision_DT, f1Score_DT = decision_tree(X_train, X_test, y_train, y_test)
    accuracy_RF, recall_RF, precision_RF, f1Score_RF = randomForest(X_train, X_test, y_train, y_test)
    accuracy_KNN, recall_KNN, precision_KNN, f1Score_KNN = KNN(X_train, X_test, y_train, y_test)
    accuracy_svm, recall_svm, precision_svm, f1Score_svm = svm(X_train, X_test, y_train, y_test)
    accuracy_NB, recall_NB, precision_NB, f1Score_nb = naive_byes(X_train, X_test, y_train, y_test)
    accuracy_xgb, recall_xgb, precision_xgb, f1Score_xgb = xgboost_classifier(X_train, X_test, y_train, y_test)

    benchmark_overall_accuracy_score.append(accuracy_DT)
    benchmark_overall_accuracy_score.append(accuracy_RF)
    benchmark_overall_accuracy_score.append(accuracy_KNN)
    benchmark_overall_accuracy_score.append(accuracy_svm)
    benchmark_overall_accuracy_score.append(accuracy_NB)
    benchmark_overall_accuracy_score.append(accuracy_xgb)

    generatePlots(benchmark_overall_accuracy_score,"Accuracy","Benchmark")

    benchmark_overall_recall_score.append(recall_DT)
    benchmark_overall_recall_score.append(recall_RF)
    benchmark_overall_recall_score.append(recall_KNN)
    benchmark_overall_recall_score.append(recall_svm)
    benchmark_overall_recall_score.append(recall_NB)
    benchmark_overall_recall_score.append(recall_xgb)

    generatePlots(benchmark_overall_recall_score,"Recall","Benchmark")

    benchmark_overall_precision_score.append(precision_DT)
    benchmark_overall_precision_score.append(precision_RF)
    benchmark_overall_precision_score.append(precision_KNN)
    benchmark_overall_precision_score.append(precision_svm)
    benchmark_overall_precision_score.append(precision_NB)
    benchmark_overall_precision_score.append(precision_xgb)

    generatePlots(benchmark_overall_precision_score,"Precision","Benchmark")

    benchmark_overall_f1_score.append(f1Score_DT)
    benchmark_overall_f1_score.append(f1Score_RF)
    benchmark_overall_f1_score.append(f1Score_KNN)
    benchmark_overall_f1_score.append(f1Score_svm)
    benchmark_overall_f1_score.append(f1Score_nb)
    benchmark_overall_f1_score.append(f1Score_nb)

    generatePlots(benchmark_overall_f1_score,"F1","Benchmark")

    # ax = sns.countplot(x=target, data=df)
    # print(df[target].value_counts())
    # print(df.shape[0])
    # print(100 * (269 / float(df.shape[0])))
    # print(100 * (139 / float(df.shape[0])))
    # print(100 * (91 / float(df.shape[0])))

    print("-----------------------------------------------------")

    print("Synthesize the input Dataset using SMOTE method")

    smote_overall_accuracy_score = []
    smote_overall_recall_score = []
    smote_overall_precision_score = []
    smote_overall_f1_score = []

    unique, count = np.unique(y_train, return_counts=True)
    Y_train_dict_value_count = {k: v for (k, v) in zip(unique, count)}
    print(f"Y_train_dict_value_count :{Y_train_dict_value_count}")
    sm = SMOTE(sampling_strategy="auto",
               random_state=None,
               k_neighbors=3,
               n_jobs=None, )
    x_train_res, y_train_res = sm.fit_sample(X_train, y_train)
    # print(f"x_train_res{x_train_res} y_train_res{y_train_res}")
    unique, count = np.unique(y_train_res, return_counts=True)
    y_train_smote_value_count = {k: v for (k, v) in zip(unique, count)}
    print(f"Y_train_dict_value_count value after smote :{y_train_smote_value_count}")

    accuracy_DT_sm, recall_DT_sm, precision_DT_sm, f1Score_DT_sm = decision_tree(x_train_res, X_test, y_train_res, y_test)
    accuracy_RF_sm, recall_RF_sm, precision_RF_sm, f1Score_RF_sm = randomForest(x_train_res, X_test, y_train_res, y_test)
    accuracy_knn_sm, recall_knn_sm, precision_knn_sm, f1Score_knn_sm = KNN(x_train_res, X_test, y_train_res, y_test)
    accuracy_svm_sm, recall_svm_sm, precision_svm_sm, f1Score_svm_sm = svm(x_train_res, X_test, y_train_res, y_test)
    accuracy_nb_sm, recall_nb_sm, precision_nb_sm, f1Score_nb_sm = naive_byes(x_train_res, X_test, y_train_res, y_test)
    accuracy_xgb_sm, recall_xgb_sm, precision_xgb_sm, f1Score_xgb_sm = xgboost_classifier(x_train_res, X_test, y_train_res, y_test)

    smote_overall_accuracy_score.append(accuracy_DT_sm)
    smote_overall_accuracy_score.append(accuracy_RF_sm)
    smote_overall_accuracy_score.append(accuracy_knn_sm)
    smote_overall_accuracy_score.append(accuracy_svm_sm)
    smote_overall_accuracy_score.append(accuracy_nb_sm)
    smote_overall_accuracy_score.append(accuracy_xgb_sm)

    generatePlots(smote_overall_accuracy_score, "Accuracy", "SMOTE")

    smote_overall_recall_score.append(recall_DT_sm)
    smote_overall_recall_score.append(recall_RF_sm)
    smote_overall_recall_score.append(recall_knn_sm)
    smote_overall_recall_score.append(recall_svm_sm)
    smote_overall_recall_score.append(recall_nb_sm)
    smote_overall_recall_score.append(recall_xgb_sm)

    generatePlots(smote_overall_recall_score, "Recall", "SMOTE")

    smote_overall_precision_score.append(precision_DT_sm)
    smote_overall_precision_score.append(precision_RF_sm)
    smote_overall_precision_score.append(precision_knn_sm)
    smote_overall_precision_score.append(precision_svm_sm)
    smote_overall_precision_score.append(precision_nb_sm)
    smote_overall_precision_score.append(precision_xgb_sm)

    generatePlots(smote_overall_precision_score, "Precision", "SMOTE")

    smote_overall_f1_score.append(f1Score_DT_sm)
    smote_overall_f1_score.append(f1Score_RF_sm)
    smote_overall_f1_score.append(f1Score_knn_sm)
    smote_overall_f1_score.append(f1Score_svm_sm)
    smote_overall_f1_score.append(f1Score_nb_sm)
    smote_overall_f1_score.append(f1Score_nb_sm)

    generatePlots(smote_overall_f1_score, "F1", "SMOTE")
    print("---------------------------------------------")
    print("Hyper parameter tuning to increase the accuracy")

    ht_overall_accuracy_score = []
    ht_overall_recall_score = []
    ht_overall_precision_score = []
    ht_overall_f1_score = []


    dt_pipe = Pipeline([
        ('sampling', SMOTE(random_state=12,)),
        ('classification', DecisionTreeClassifier())
    ])

    accuracy_DT_ht, recall_DT_ht, precision_DT_ht, f1Score_DT_ht =tuneParameterandGenerateModel("Decision Tree" ,dt_pipe)

    rf_pipe = Pipeline([
        ('sampling', SMOTE(random_state=12, )),
        ('classification', RandomForestClassifier())
    ])

    accuracy_RF_ht, recall_RF_ht, precision_RF_ht, f1Score_RF_ht = tuneParameterandGenerateModel("Random Forest", rf_pipe)

    knn_pipe = Pipeline([
        ('sampling', SMOTE(random_state=12, )),
        ('classification', KNeighborsClassifier())
    ])

    accuracy_KNN_ht, recall_KNN_ht, precision_KNN_ht, f1Score_KNN_ht = tuneParameterandGenerateModel("KNN" ,knn_pipe)

    svm_pipe = Pipeline([
        ('sampling', SMOTE(random_state=12, )),
        ('classification', SVC(C=1.0, kernel="linear"))
    ])

    accuracy_svm_ht, recall_svm_ht, precision_svm_ht, f1Score_svm_ht = tuneParameterandGenerateModel("SVM", svm_pipe)

    nb_pipe = Pipeline([
        ('sampling', SMOTE(random_state=12, )),
        ('classification', GaussianNB())
    ])

    accuracy_nb_ht, recall_nb_ht, precision_nb_ht, f1Score_nb_ht = tuneParameterandGenerateModel("Naive Byes", nb_pipe)

    xgboost_pipe = Pipeline([
        ('sampling', SMOTE(random_state=12, )),
        ('classification', xgb.XGBClassifier())
    ])

    accuracy_xgb_ht, recall_xgb_ht, precision_xgb_ht, f1Score_xgb_ht = tuneParameterandGenerateModel("XGBoost", xgboost_pipe)

    ht_overall_accuracy_score.append(accuracy_DT_ht)
    ht_overall_accuracy_score.append(accuracy_RF_ht)
    ht_overall_accuracy_score.append(accuracy_KNN_ht)
    ht_overall_accuracy_score.append(accuracy_svm_ht)
    ht_overall_accuracy_score.append(accuracy_nb_ht)
    ht_overall_accuracy_score.append(accuracy_xgb_ht)

    generatePlots(ht_overall_accuracy_score, "Accuracy", "Hyper parameter Tuning")

    ht_overall_recall_score.append(recall_DT_ht)
    ht_overall_recall_score.append(recall_RF_ht)
    ht_overall_recall_score.append(recall_KNN_ht)
    ht_overall_recall_score.append(recall_svm_ht)
    ht_overall_recall_score.append(recall_nb_ht)
    ht_overall_recall_score.append(recall_xgb_ht)

    generatePlots(ht_overall_recall_score, "Recall", "Hyper parameter Tuning")

    ht_overall_precision_score.append(precision_DT_ht)
    ht_overall_precision_score.append(precision_RF_ht)
    ht_overall_precision_score.append(precision_KNN_ht)
    ht_overall_precision_score.append(precision_svm_ht)
    ht_overall_precision_score.append(precision_nb_ht)
    ht_overall_precision_score.append(precision_xgb_ht)

    generatePlots(ht_overall_precision_score, "Precision", "Hyper parameter Tuning")

    ht_overall_f1_score.append(f1Score_DT_ht)
    ht_overall_f1_score.append(f1Score_RF_ht)
    ht_overall_f1_score.append(f1Score_KNN_ht)
    ht_overall_f1_score.append(f1Score_svm_ht)
    ht_overall_f1_score.append(f1Score_nb_ht)
    ht_overall_f1_score.append(f1Score_nb_ht)

    generatePlots(ht_overall_f1_score, "F1", "Hyper parameter Tuning")