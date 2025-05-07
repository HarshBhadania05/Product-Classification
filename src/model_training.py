from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from src import config
from src.utils import save_model

def train_models(X_train, y_train):
    models = {}

    # Logistic Regression with GridSearchCV
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear']
    }
    grid_lr = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, n_jobs=-1)
    grid_lr.fit(X_train, y_train)
    best_lr = grid_lr.best_estimator_
    models['logistic_regression'] = best_lr
    save_model(best_lr, f"{config.MODEL_DIR}/logistic_regression.joblib")

    # Neural Network
    nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=config.RANDOM_STATE)
    nn.fit(X_train, y_train)
    models['neural_network'] = nn
    save_model(nn, f"{config.MODEL_DIR}/neural_network.joblib")

    return models