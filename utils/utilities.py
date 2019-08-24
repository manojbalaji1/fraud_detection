from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split as sklearn_train_test_split
import random


def train_test_split(X, y, test_size, random_state=random.randint(1, 1000)):
    return sklearn_train_test_split(X, y, test_size=test_size, random_state=random_state)


def split_email_id(email_id: str):
    return email_id.split('.')


def encode_categorial_features_fit(df):
    categorical_encoders = dict()
    for c in df.columns:
        encoder = LabelEncoder()
        encoder.fit(df[c].astype(str).values)
        categorical_encoders[c] = encoder
    return categorical_encoders


def encode_categorial_features_transform(df, encoders):
    out = pd.DataFrame(index=df.index)
    for c in encoders.keys():
        if c in df.columns:
            out[c] = encoders[c].transform(df[c].astype(str).values).tolist()
    return out


def df_categorical_variable_summary(df):
    least = df.min().to_frame('Least')
    highest = df.max().to_frame('Highest')
    mode = df.mode().transpose()
    mode.columns = ['Mode']
    categorical_variable_summary = least.join(highest) \
        .join(mode)
    categorical_variable_summary['Variable Name'] = categorical_variable_summary.index

    categorical_variable_summary['# of Levels'] = categorical_variable_summary['Highest'] - \
                                                  categorical_variable_summary['Least'] + 1
    categorical_variable_summary = categorical_variable_summary[
        ['Variable Name', '# of Levels', 'Least', 'Highest', 'Mode']]
    return categorical_variable_summary


def df_continuous_variable_summary(df):
    continuous_features_summary = df.describe().transpose()
    continuous_features_summary['variable'] = continuous_features_summary.index
    continuous_features_kurtosis = df.kurtosis().to_frame('kurtosis')
    continuous_features_kurtosis['variable'] = continuous_features_kurtosis.index
    continuous_features_skewness = df.skew().to_frame('skewness')
    continuous_features_skewness['variable'] = continuous_features_skewness.index
    continuous_data_summary = continuous_features_summary.merge(continuous_features_skewness, on='variable') \
        .merge(continuous_features_kurtosis, on='variable')
    continuous_data_summary["NA's"] = df.shape[0] - continuous_data_summary['count']
    continuous_data_summary = continuous_data_summary.drop('count', axis=1)
    continuous_data_summary['Skewed (Y/N)'] = ['N' if 0.5 >= skew >= -0.5 else 'Y' for skew in
                                               continuous_data_summary['skewness']]
    continuous_data_summary = continuous_data_summary[
        ['variable', 'min', '25%', '50%', 'mean', '75%', 'max', "NA's", 'Skewed (Y/N)', 'skewness', 'kurtosis']]
    return continuous_data_summary


def generate_reports(prefix, df, continuous_variable_list, categorical_variable_list):
    continuous_variable_df = df[continuous_variable_list]
    continuous_variable_summary_df = df_continuous_variable_summary(continuous_variable_df)
    categorical_variable_df = df[categorical_variable_list]
    categorical_variable_summary_df = df_categorical_variable_summary(categorical_variable_df)
    continuous_variable_summary_df.to_excel('reports/' + prefix + 'Continuous_Variable_Analysis_Report.xlsx')
    categorical_variable_summary_df.to_excel('reports/' + prefix + 'Categorical_Variables_Analysis_Report.xlsx')


# Helper functions
def value_counts(df, keepna: bool):
    return df.value_counts(dropna=not keepna)


def null_count(df):
    return df.isnull().sum(axis=0)


def get_transaction_identity_merge_df(transaction_df, identity_df):
    return transaction_df.merge(identity_df, how='left', left_index=True, right_index=True)


def get_features_sliced_df(df, feature_list):
    return df[feature_list]
