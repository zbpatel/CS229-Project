# CS229-Project
CS229 Fall 2020 Term Project. Members: Zac Patel (zbpatel), Ghulam Mustafa

File Description:

Notebooks:

**preprocess_data.ipynb:** Jupyter Notebook that intakes a raw data file output by our test machine, cleans the data, 
resamples it from seconds granularity to minutes and saves the result to disk. Other files in the project that perform
analysis on the data require it to be processed by this script.

**EDA.ipynb:** Jupyter Notebook that contains some exploration of our dataset, as well as an experiment with PCA and a
discussion of our analysis.

**predict_future_output.ipynb** Jupyter Notebook that contains our experiments in using regression techniques to predict
the future output of our electron guns.

Helper Files:

**experiment_base.ipynb:** a file that contains some starter code from which new experimental notebooks can be started.

**utils.py:** contains a series of helper functions that handle common tasks like splitting our dataset into dated slices or
separating the different quartile values in the dataset after resampling.

Custom Models:

**bla_avg_model.py:** contains the source for our baseline model in predict_future_output. This model implements a very
standard version of linear regression from scikit learn, but averages the output of all electron guns when training. As 
such, it is intended to represent an "average model" that doesn't have any knowledge of the differences of individual
guns.

Other notes:
The data used in this experiment was proprietary data collected by one of the authors that we are unable to distribute
with the work. The files in this experiment expect the data to be in CSV format. The location for the dataset can be
specified by setting strings that are typically located at the top of each notebook.