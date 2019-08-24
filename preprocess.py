from utils.utilities import split_email_id, encode_categorial_features_fit, encode_categorial_features_transform, \
    df_categorical_variable_summary, get_features_sliced_df
from sklearn.impute import SimpleImputer
import pandas as pd


class PreProcessCategoricalVariables:
    categorical_features_imputed_df = None
    categorical_features_imputed_encoded_df = None

    def __init__(self, df, variable_list):
        self.categorical_features_df = get_features_sliced_df(df, variable_list)

    def impute_null_values(self, categorical_fillna_dict):
        self.categorical_features_imputed_df = self.categorical_features_df.fillna(categorical_fillna_dict)

    def preprocess_email(self):
        self.categorical_features_imputed_df['P_emaildomain_vendor'] = self.categorical_features_imputed_df[
            'P_emaildomain'].map(lambda x: split_email_id(x)[0])
        self.categorical_features_imputed_df['P_emaildomain_extension'] = self.categorical_features_imputed_df[
            'P_emaildomain'].map(lambda x: split_email_id(x)[-1])
        self.categorical_features_imputed_df['R_emaildomain_vendor'] = self.categorical_features_imputed_df[
            'R_emaildomain'].map(lambda x: split_email_id(x)[0])
        self.categorical_features_imputed_df['R_emaildomain_extension'] = self.categorical_features_imputed_df[
            'R_emaildomain'].map(lambda x: split_email_id(x)[-1])
        self.categorical_features_imputed_df = self.categorical_features_imputed_df.drop(
            ['R_emaildomain', 'P_emaildomain'], axis=1)

    def preprocess_and_encode(self, categorical_fillna_dict):
        try:
            self.impute_null_values(categorical_fillna_dict)
            self.preprocess_email()
            self.categorical_features_imputed_encoded_df = encode_categorial_features_transform(
                self.categorical_features_imputed_df,
                encode_categorial_features_fit(
                    self.categorical_features_imputed_df))
            return self.categorical_features_imputed_encoded_df
        except Exception:
            print("Please Impute Null variables properly!")

    def summary(self):
        if self.categorical_features_imputed_encoded_df is None:
            df_categorical_variable_summary(self.categorical_features_imputed_df)
        else:
            df_categorical_variable_summary(self.categorical_features_imputed_encoded_df)


class PreProcessContinuousVariables:
    continuous_features_imputed_df = None
    continuous_features_imputed_array = None

    def __init__(self, df, variable_list):
        self.continuous_features_df = get_features_sliced_df(df, variable_list)

    def impute_null_values(self):
        imp_mean = SimpleImputer(strategy='mean')  # for median imputation replace 'mean' with 'median'
        imp_mean.fit(self.continuous_features_df)
        self.continuous_features_imputed_array = imp_mean.transform(self.continuous_features_df)
        self.continuous_features_df = pd.DataFrame(self.continuous_features_imputed_array,
                                                   columns=self.continuous_features_df.columns,
                                                   index=self.continuous_features_df.index)

    def preprocess(self):
        self.impute_null_values()
        return self.continuous_features_df
