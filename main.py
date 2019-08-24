import sys
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
from utils.utilities import get_transaction_identity_merge_df, generate_reports
from config.settings import TRAIN_IDENTITY_FILE_PATH, TRAIN_IDENTITY_FILE_INDEX_VARIABLE, TRAIN_TRANSACTION_FILE_PATH, \
    TRAIN_TRANSACTION_FILE_INDEX_VARIABLE, TEST_IDENTITY_FILE_PATH, TEST_IDENTITY_FILE_INDEX_VARIABLE, \
    TEST_TRANSACTION_FILE_PATH, TEST_TRANSACTION_FILE_INDEX_VARIABLE, CATEGORICAL_VARIABLES, \
    CATEGORICAL_VARIABLES_FILL_NAN, TARGET_VARIABLE, LIGHTGMB_MODEL_PARAMS, MODEL_FILE_PATH, SELECTED_FEATURES, \
    LOG_NAME, LOG_FILE_NAME, LOG_FORMAT, MLFLOW_TRACKING_SERVER_URI, MLFLOW_EXPERIMENT_NAME, MLFLOW_MODEL_DOWNLOAD_PATH
from preprocess import PreProcessCategoricalVariables, PreProcessContinuousVariables
import argparse
from models import LightGBM, load_model_from_mlflow
from utils import logger


def preprocess(df):
    CONTINUOUS_VARIABLES = [col for col in df.columns if col not in CATEGORICAL_VARIABLES]
    # preprocess categorical variables
    categorical_variable_preprocessor = PreProcessCategoricalVariables(df, CATEGORICAL_VARIABLES)
    train_categorical_feature_df = categorical_variable_preprocessor.preprocess_and_encode(
        CATEGORICAL_VARIABLES_FILL_NAN)
    # preprocess continuous variables
    continuous_variable_preprocessor = PreProcessContinuousVariables(df, CONTINUOUS_VARIABLES)
    train_continuous_feature_df = continuous_variable_preprocessor.preprocess()
    train_df = get_transaction_identity_merge_df(train_continuous_feature_df, train_categorical_feature_df)
    # generate reports of variable distribution before preprocessing
    generate_reports(prefix='After_Preprocessing_', df=train_df,
                     continuous_variable_list=train_continuous_feature_df.columns,
                     categorical_variable_list=train_categorical_feature_df.columns)
    return train_df


def isolate_target_variable(df, target):
    df_target = df[TARGET_VARIABLE].copy()
    df_features = df.drop(TARGET_VARIABLE, axis=1)
    return df_features, df_target


def model_train(df, train_target, params):
    lgbm = LightGBM(X=df, y=train_target, test_size=0.25, params=params,
                    mlflow_tracking_server_uri=MLFLOW_TRACKING_SERVER_URI, experiment_name=MLFLOW_EXPERIMENT_NAME)
    logger.info("Training......")
    lgbm.train()
    logger.info("Evaluating....")
    lgbm.evaluate()
    logger.info("Saving model.....")
    lgbm.save_model(MODEL_FILE_PATH)


def predict_probability(df):
    logger.info("Loading Model.....")
    model = load_model_from_mlflow(mlflow_tracking_server_uri=MLFLOW_TRACKING_SERVER_URI, experiment_name=MLFLOW_EXPERIMENT_NAME, dest_path=MLFLOW_MODEL_DOWNLOAD_PATH)
    logger.info("Predicting probabilities....")
    y_pred = model.predict_proba(df)[:, 1]
    return y_pred


parser = argparse.ArgumentParser(description='Run training')
parser.add_argument('--job-type', dest='job_type', type=str, required=True, help='type of job training/prediction')
parser.add_argument('--feature-selection', dest='feature_selection', type=str, required=True,
                    help='if feature selection is required')

args = parser.parse_args()
logger.Logger(LOG_NAME, LOG_FILE_NAME, LOG_FORMAT)

if args.job_type == 'train':
    # import dataset
    train_identity_df = pd.read_csv(TRAIN_IDENTITY_FILE_PATH, index_col=TRAIN_IDENTITY_FILE_INDEX_VARIABLE)
    train_transaction_df = pd.read_csv(TRAIN_TRANSACTION_FILE_PATH, index_col=TRAIN_TRANSACTION_FILE_INDEX_VARIABLE)
    logger.info(
        "Train Identity DF Length: {} Features: {}".format(train_identity_df.shape[0], train_identity_df.shape[1]))
    logger.info("Train Transaction DF Length: {} Features: {}".format(train_transaction_df.shape[0],
                                                                      train_transaction_df.shape[1]))

    # combine both the data to form the train dataset
    train_df = get_transaction_identity_merge_df(train_transaction_df, train_identity_df)
    del train_identity_df, train_transaction_df
    gc.collect()

    # isolate target variable 'isFraud'
    train_df_X, train_target = isolate_target_variable(train_df, TARGET_VARIABLE)

    # preprocess fetaures
    logger.info("Pre-Processing.......")
    train_df_X_preprocessed = preprocess(train_df_X)

    if args.feature_selection == "true":
        # feature selected
        train_df_X_preprocessed_selected = train_df_X_preprocessed[SELECTED_FEATURES]
        del train_df_X_preprocessed
        gc.collect()
        # training process
        model_train(train_df_X_preprocessed_selected, train_target, LIGHTGMB_MODEL_PARAMS)

    else:
        # training process
        model_train(train_df_X_preprocessed,  train_target, LIGHTGMB_MODEL_PARAMS)


elif args.job_type == 'prediction':
    test_identity_df = pd.read_csv(TEST_IDENTITY_FILE_PATH, index_col=TEST_IDENTITY_FILE_INDEX_VARIABLE)
    test_transaction_df = pd.read_csv(TEST_TRANSACTION_FILE_PATH, index_col=TEST_TRANSACTION_FILE_INDEX_VARIABLE)
    logger.info("Test Identity DF Length: {} Features: {}".format(test_identity_df.shape[0], test_identity_df.shape[1]))
    logger.info("Test Transaction DF Length: {} Features: {}".format(test_transaction_df.shape[0],
                                                                     test_transaction_df.shape[1]))

    # combine both the data to form the train dataset
    test_df = get_transaction_identity_merge_df(test_transaction_df, test_identity_df)
    del test_transaction_df, test_identity_df
    gc.collect()

    # preprocess fetaures
    logger.info("Pre-Processing.......")
    test_df_X_preprocessed = preprocess(test_df)

    if args.feature_selection == "true":
        # feature selected
        test_df_X_preprocessed_selected = test_df_X_preprocessed[SELECTED_FEATURES]
        del test_df_X_preprocessed
        gc.collect()
        # prediction
        y_pred = predict_probability(test_df_X_preprocessed_selected)
    else:
        # prediction
        y_pred = predict_probability(test_df_X_preprocessed)

    df = pd.DataFrame()
    df['TransactionID'] = test_df_X_preprocessed.index
    df['isFraud'] = y_pred
    logger.info("Saving submission.csv")
    df.to_csv('submission.csv', index=False)

else:
    sys.exit("Enter Valid Job Type")
