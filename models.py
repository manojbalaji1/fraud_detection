import xgboost as xgb
from utils.utilities import train_test_split
import pickle
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from utils import logger
from model_versioning.mlflow_client import CustomerMlflowClient


class XGBoost:

    def __init__(self, X, y, test_size, params, num_round, mlflow_tracking_server_uri, experiment_name,
                 early_stopping_rounds=10):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size)
        self.dtrain = xgb.DMatrix(data=self.X_train, label=self.y_train)
        self.dtest = xgb.DMatrix(data=self.X_test, label=self.y_test)
        self.evallist = [(self.dtest, 'eval'), (self.dtrain, 'train')]
        self.num_round = num_round
        self.params = params
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None
        self.y_pred = None
        self.dtrain = None
        self.dtest = None
        self.evallist = None
        self.y_pred_score = None
        self.mlflow_server_uri = mlflow_tracking_server_uri
        self.experiment_name = experiment_name
        self.mlflow_client = CustomerMlflowClient(tracking_server_uri=self.mlflow_server_uri,
                                                  experiment_name=self.experiment_name)
        self.metrics = dict()

    def train(self):
        self.model = xgb.train(self.params, self.dtrain, self.num_round, self.evallist,
                               early_stopping_rounds=self.early_stopping_rounds)

    def test(self):
        self.y_pred_score = self.model.predict(self.dtest)
        self.y_pred = self.y_pred_score >= 0.5

    def predict(self, data):
        X_predict = xgb.DMatrix(data=data)
        return self.model.predict(X_predict)

    def save_model(self, model_file_path):
        if model_file_path:
            logger.info("Please provide model_file_path")
        else:
            with open(model_file_path, "wb") as file:
                pickle.dump(self.model, file)
        self.mlflow_client.logger(self.params, self.metrics, model_file_path)

    def evaluate(self):
        self.test()
        self.metrics['roc_auc'] = roc_auc_score(y_true=self.y_test, y_score=self.y_pred_score)
        logger.info(self.metrics['roc_auc'])


class LightGBM:

    def __init__(self, X, y, test_size, params, mlflow_tracking_server_uri, experiment_name):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size)
        self.params = params
        self.model = None
        self.y_pred = None
        self.y_pred_score = None
        self.mlflow_server_uri = mlflow_tracking_server_uri
        self.experiment_name = experiment_name
        self.mlflow_client = CustomerMlflowClient(tracking_server_uri=self.mlflow_server_uri,
                                                  experiment_name=self.experiment_name)
        self.metrics = dict()

    def train(self):
        self.model = lgb.LGBMClassifier(**self.params).fit(self.X_train, self.y_train)

    def test(self):
        self.y_pred_score = self.model.predict_proba(self.X_test)[:, 1]
        self.y_pred = self.y_pred_score >= 0.5

    def predict(self, data):
        return self.model.predict_proba(data)[:, 1]

    def save_model(self, model_file_path):
        if not model_file_path:
            logger.info("Please provide model_file_path")
        else:
            with open(model_file_path, "wb") as file:
                pickle.dump(self.model, file)
        self.mlflow_client.logger(self.params, self.metrics, model_file_path)

    def evaluate(self):
        self.test()
        self.metrics['roc_auc'] = roc_auc_score(y_true=self.y_test, y_score=self.y_pred_score)
        logger.info(self.metrics['roc_auc'])


def load_model(model_file_path):
    if not model_file_path:
        logger.info("Please provide model_file_path")
    else:
        with open(model_file_path, "rb") as file:
            model = pickle.load(file)
    return model


def load_model_from_mlflow(mlflow_tracking_server_uri, experiment_name, dest_path):
    try:
        mlflow_client = CustomerMlflowClient(tracking_server_uri=mlflow_tracking_server_uri,
                                             experiment_name=experiment_name)
    except Exception as e:
        logger.error(str(e))
        raise e
    model_file_path = mlflow_client.get_latest_artifact(dest_path)

    if not model_file_path:
        logger.info("model_file_path not correct as downloaded model path")
    else:

        with open(model_file_path, "rb") as file:
            model = pickle.load(file)
    return model
