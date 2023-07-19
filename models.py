import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, accuracy_score, make_scorer

def get_x(df):
    return df.drop(["Question", "Answer", "Gender", "AnswerGender"], axis=1)

def get_y(df):
    return df["Gender"]

def df_train_test_split(df):
    X = get_x(df)
    Y = get_y(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        Y, 
        test_size = 0.25, 
        stratify=Y, 
        random_state=42
    )

    return X_train, X_test, y_train, y_test

def model_metrics(y_test, y_pred):
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred))

def plot_features_importance(coefficients, X, num = 25):
    feature_importance = pd.DataFrame({'Feature': [x[::-1] for x in X.columns], 'Importance': np.abs(coefficients)})
    feature_importance = feature_importance.sort_values('Importance', ascending=True)
    feature_importance_n = feature_importance.nlargest(num, 'Importance')
    feature_importance_n.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6*(num/33)))

# def plot_features_importance_rand_forest(forest, X):
#     feature_names = [f"feature {i}" for i in range(X.shape[1])]
#     importances = forest.feature_importances_
#     std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
#     forest_importances = pd.Series(importances, index=feature_names)
# 
#     fig, ax = plt.subplots()
#     forest_importances.plot.bar(yerr=std, ax=ax)
#     ax.set_title("Feature importances using MDI")
#     ax.set_ylabel("Mean decrease in impurity")
#     fig.tight_layout()


def LogReg(X_train, X_test, y_train, y_test, max_iter = 100):
    log_reg = LogisticRegression(max_iter = max_iter)
 
    log_reg.fit(X_train, y_train)

    print(f"LogReg train accuracy: {log_reg.score(X_train, y_train):.3f}")
    print(f"LogReg test accuracy: {log_reg.score(X_test, y_test):.3f}")

    y_pred = log_reg.predict(X_test)
    model_metrics(y_test, y_pred)
    # plot_features_importance(log_reg.coef_[0], X_train)
    
    return log_reg

def LogRegCV(X_train, X_test, y_train, y_test, max_iter = 100):
    log_reg_cv = LogisticRegressionCV(cv=10, max_iter = max_iter, random_state=0)
 
    log_reg_cv.fit(X_train, y_train)

    print(f"LogRegCV train accuracy: {log_reg_cv.score(X_train, y_train):.3f}")
    print(f"LogRegCV test accuracy: {log_reg_cv.score(X_test, y_test):.3f}")

    y_pred = log_reg_cv.predict(X_test)
    model_metrics(y_test, y_pred)
    # plot_features_importance(log_reg_cv.coef_[0], X_train)
    
    return log_reg_cv
    
## TODO: add NB
## TODO: add NBCV

def RandForest(X_train, X_test, y_train, y_test, n_estimators = 1000, max_depth = 500, min_samples_leaf = 1):
    rand_forest = RandomForestClassifier(n_estimators=n_estimators,
         n_jobs=-1,
         min_samples_leaf = min_samples_leaf,
         oob_score=True,
         random_state = 42,
         max_depth=max_depth)

    rand_forest.fit(X_train, y_train)
    
    print(f"RandForest train accuracy: {rand_forest.score(X_train, y_train):.3f}")
    print(f"RandForest test accuracy: {rand_forest.score(X_test, y_test):.3f}")

    y_pred = rand_forest.predict(X_test)
    model_metrics(y_test, y_pred)
    # plot_features_importance(rand_forest.feature_importances_, X_train)
    return rand_forest
    
def RFGSCV(X_train, X_test, y_train, y_test, param_grid):
    rf = RandomForestClassifier(max_features='sqrt', random_state=42, n_jobs=-1)  

    scoring = {"AUC": "roc_auc", "Accuracy": make_scorer(accuracy_score)}
    CV_rf = GridSearchCV(estimator=rf, param_grid=param_grid, scoring=scoring, refit="AUC")  
    CV_rf.fit(X_train, y_train)

    print(CV_rf.best_params_)
    print(CV_rf.best_index_)
    print(CV_rf.best_score_)
    print(CV_rf.best_estimator_)

    return CV_rf


# Taken from - https://www.kaggle.com/code/grfiv4/displaying-the-results-of-a-grid-search
def GridSearch_table_plot(grid_clf, param_name,
                          num_results=15,
                          negative=True,
                          graph=True,
                          display_all_params=True):

    '''Display grid search results

    Arguments
    ---------

    grid_clf           the estimator resulting from a grid search
                       for example: grid_clf = GridSearchCV( ...

    param_name         a string with the name of the parameter being tested

    num_results        an integer indicating the number of results to display
                       Default: 15

    negative           boolean: should the sign of the score be reversed?
                       scoring = 'neg_log_loss', for instance
                       Default: True

    graph              boolean: should a graph be produced?
                       non-numeric parameters (True/False, None) don't graph well
                       Default: True

    display_all_params boolean: should we print out all of the parameters, not just the ones searched for?
                       Default: True

    Usage
    -----

    GridSearch_table_plot(grid_clf, "min_samples_leaf")

                          '''
    from matplotlib      import pyplot as plt
    from IPython.display import display
    import pandas as pd

    clf = grid_clf.best_estimator_
    clf_params = grid_clf.best_params_
    if negative:
        clf_score = -grid_clf.best_score_
    else:
        clf_score = grid_clf.best_score_
    clf_stdev = grid_clf.cv_results_['std_test_score'][grid_clf.best_index_]
    cv_results = grid_clf.cv_results_

    print("best parameters: {}".format(clf_params))
    print("best score:      {:0.5f} (+/-{:0.5f})".format(clf_score, clf_stdev))
    if display_all_params:
        import pprint
        pprint.pprint(clf.get_params())

    # pick out the best results
    # =========================
    scores_df = pd.DataFrame(cv_results).sort_values(by='rank_test_score')

    best_row = scores_df.iloc[0, :]
    if negative:
        best_mean = -best_row['mean_test_score']
    else:
        best_mean = best_row['mean_test_score']
    best_stdev = best_row['std_test_score']
    best_param = best_row['param_' + param_name]

    # display the top 'num_results' results
    # =====================================
    display(pd.DataFrame(cv_results) \
            .sort_values(by='rank_test_score').head(num_results))

    # plot the results
    # ================
    scores_df = scores_df.sort_values(by='param_' + param_name)

    if negative:
        means = -scores_df['mean_test_score']
    else:
        means = scores_df['mean_test_score']
    stds = scores_df['std_test_score']
    params = scores_df['param_' + param_name]

    # plot
    if graph:
        plt.figure(figsize=(8, 8))
        plt.errorbar(params, means, yerr=stds)

        plt.axhline(y=best_mean + best_stdev, color='red')
        plt.axhline(y=best_mean - best_stdev, color='red')
        plt.plot(best_param, best_mean, 'or')

        plt.title(param_name + " vs Score\nBest Score {:0.5f}".format(clf_score))
        plt.xlabel(param_name)
        plt.ylabel('Score')
        plt.show()