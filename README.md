# fraud_detection

First please install all dependencies using the following commands<br>

<code>pip install -r requirements.py</code>
<br>
After installing dependencies, start the mlflow server using the following commands<br>

<code>mlflow serve -p [PORT] --host [HOST] --workers [NUM_OF_WORKERS] --backend-store-uri [BACKEND_STORE_URI] --default-artifact-root [ARTIFACT_ROOT_FOLDER_PATH]</code>
<br>

Set all required configuration related to code in config/settings.py file<br>

Once all above instructions are followed, the code can be run using the following commands<br>

## For Training
for using selected features set --feature-selection true
<code>python main.py --job-type train --feature-selection false</code>

## For Inferencing
for using selected features set --feature-selection true
<code>python main.py --job-type prediction --feature-selection false</code>
<br>

Data will be added once the contest is done
