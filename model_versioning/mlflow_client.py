from mlflow.tracking import  MlflowClient
from utils import logger


class CustomerMlflowClient:
    def __init__(self, tracking_server_uri, experiment_name):
        try:
            self.mlflow_client = MlflowClient(tracking_server_uri)
            logger.info("established mlflow rest-api client")
        except Exception as e:
            logger.error(str(e))

        try:
            self.experiment_id = self.set_experiment(experiment_name)
            logger.info("started mlflow experiment {} with id {}".format(experiment_name, self.experiment_id))
        except Exception as e:
            logger.error(str(e))

    def logger(self, params, metrics, local_artifact_path, mlflow_artifact_path=None):
        run = self.mlflow_client.create_run(self.experiment_id)
        run_id = run.info.run_id
        logger.info("staring new run with id: {}".format(run_id))
        logger.info("logging parameter to mlflow tracking server")
        self.log_params(run_id, params)
        logger.info("successfully logged parameter to mlflow tracking server")
        logger.info("logging model metrics to mlflow tracking server")
        self.log_metrics(run_id, metrics)
        logger.info("successfully logged model metrics to mlflow tracking server")
        logger.info("logging model artifact to mlflow tracking server")
        self.log_artifact(run_id, local_artifact_path)
        logger.info("successfully logged model artifact to mlflow tracking server")
        logger.info("exiting run with id: {}".format(run_id))

    def set_experiment(self, experiment_name):
        experiment = self.mlflow_client.get_experiment_by_name(experiment_name)
        if experiment is None:
            return self.mlflow_client.create_experiment(experiment_name)
        else:
            return experiment.experiment_id

    def log_params(self, run_id: int, params):
        for key, value in params.items():
            self.mlflow_client.log_param(run_id=run_id, key=key, value=value)

    def log_metrics(self, run_id: int, metrics):
        for key, value in metrics.items():
            self.mlflow_client.log_metric(run_id=run_id, key=key, value=value)

    def log_artifact(self, run_id: int, artifact):
        self.mlflow_client.log_artifact(run_id=run_id, local_path=artifact)

    def get_latest_artifact(self, dest_path):
        run_info = self.mlflow_client.list_run_infos(self.experiment_id)
        latest_run_info = run_info[0]
        file_name = self.mlflow_client.list_artifacts(run_id=latest_run_info.run_id)[0].path
        complete_artifact_path = latest_run_info.artifact_uri + '/' + file_name
        self.mlflow_client.download_artifacts(run_id=latest_run_info.run_id, path=complete_artifact_path, dst_path=dest_path)
        return dest_path+file_name

