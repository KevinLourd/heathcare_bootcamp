from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.pipeline import Pipeline


def model(X_train, y_train, X_test):
    columnSelected = np.array([370, 282, 285, 242, 284, 101, 130, 303,  24,  22, 183,  23, 304,
                               131, 144,  26,  27,   2, 211, 102, 103, 212, 132, 210, 145, 359,
                               330, 292, 262, 226, 270, 146,  28, 334, 331, 329, 113, 357, 341,
                               485,  60,  61, 272, 360,  62, 283, 306, 291,  94, 111, 265, 170,
                               356, 345,  50, 333, 206, 353, 114, 337])
    X_train = X_train[columnSelected]
    clf = Pipeline([('imputer', Imputer(strategy='most_frequent')),
                ('scaler', StandardScaler()),
                ('select', SelectPercentile(f_classif, 90)),
                ('clf', AdaBoostClassifier(RandomForestClassifier(n_estimators=300, max_depth=3, n_jobs=-1), n_estimators=20))
                ])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)
    return y_pred, y_score