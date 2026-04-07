
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

def get_models():
    """
    Returns a dictionary of models to be trained.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Naive Bayes": MultinomialNB(),
        "SVM": SVC(kernel='linear', random_state=42), # Linear kernel is often good for text
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Extra Trees": ExtraTreesClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42)
    }

    # Try to import XGBoost
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(random_state=42, eval_metric='mlogloss')
    except ImportError:
        print("XGBoost not installed. Skipping XGBoost.")

    return models
