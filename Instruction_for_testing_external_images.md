
## Pre-requiste

- Prepare the test dataset folder. It must have the classes to be tested as folder-name and should contain the images of the respective class. Make sure that you have folders for all the classes.

- Refer to `categories.json` for the name of the classes.

- The `eval.py` script will reference the `latest_config.json` for model and dataset loading and automatic evaluation.

- Please edit the `validation_dataset_path` in `latest_config.json`

- Run `eval.py`. You should get loss, accuracy, and the confusion matrix on the prediction.