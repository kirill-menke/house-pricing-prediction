# Moscow House Pricing Prediction

Installing all required packages: <br />
`` pip install -r "requirements.txt" ``

Running the regressor on the provided test set: <br />
``python regressor.py test`` <br />
This will generate a ``result.csv`` file for submission on kaggle.

Running the regressor on a validation set (which is a random subset of the training data): <br />
``python regressor.py val`` <br />
This will print out the RMSLE score of the regression.
