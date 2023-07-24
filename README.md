# FLHuskies_PTTF

Repository to share scripts and data for: [Pushback to the Future: Predict Pushback Time at US Airports](https://www.drivendata.org/competitions/group/competition-nasa-airport-pushback/)
Used by team FLHuskies from the University of Washington Tacoma.

## Members

Faculty:
  - Dr. Martine De Cock
  - Dr. Anderson Nascimento


PhD Students:
  - Sikha Pentyala


Undergraduates:
  - Kyler Robison
  - Yudong Lin
  - Daniil Filienko
  - Trevor Tomlin

## Data Directory
This repository does not contain the data as it is too large.

Scripts operate under the assumption that there is a directory named "_data" in the 
root of the repository.

It has an underscore so that it stays out of the way up at the very top.

Furthermore, they assume that the directory has a structure as follows:

```
_data
├── private
│   ├── <airport>
│   │   ├── <airport>_AAL_mfs.csv  # contains rows for AAL airline, all columns
│   │   ├── <airport>_AAL_standtimes.csv  # contains rows for ALL, all columns
│   │   ├── ...
│   │   ├── <airport>_UPS_mfs.csv  # contains rows for UPS airline, all columns
│   │   └── <airport>_UPS_standtimes.csv
│   ├── ...
├── public
│   ├── <airport>
│   │   ├── <airport>_config.csv
│   │   ├── <airport>_etd.csv
│   │   ├── <airport>_first_position.csv
│   │   ├── <airport>_lamp.csv
│   │   ├── <airport>_mfs.csv
│   │   ├── <airport>_runways.csv
│   │   ├── <airport>_standtimes.csv
│   │   ├── <airport>_tbfm.csv
│   │   └── <airport>_tfm.csv
│   ├── ...
├── train_labels_phase2
│   ├── phase2_train_labels_<airport>.csv
│   └── ...
└── submission_format.csv
```

If it is desired to work with compressed tables to save storage space, the directory should appear as follows:

```
_data
├── private
│   ├── <airport>
│   │   ├── <airport>_AAL_mfs.csv.bz2  # contains rows for AAL airline, all columns
│   │   ├── <airport>_AAL_standtimes.csv.bz2  # contains rows for ALL, all columns
│   │   ├── ...
│   │   ├── <airport>_UPS_mfs.csv.bz2  # contains rows for UPS airline, all columns
│   │   └── <airport>_UPS_standtimes.csv.bz2
│   ├── ...
├── public
│   ├── <airport>
│   │   ├── <airport>_config.csv.bz2
│   │   ├── <airport>_etd.csv.bz2
│   │   ├── <airport>_first_position.csv.bz2
│   │   ├── <airport>_lamp.csv.bz2
│   │   ├── <airport>_mfs.csv.bz2
│   │   ├── <airport>_runways.csv.bz2
│   │   ├── <airport>_standtimes.csv.bz2
│   │   ├── <airport>_tbfm.csv.bz2
│   │   └── <airport>_tfm.csv.bz2
│   ├── ...
├── train_labels_phase2
│   ├── phase2_train_labels_<airport>.csv.bz2
│   └── ...
└── submission_format.csv.bz2
```

Scripts that can read and use these compressed tables should be supplied a single command line argument "compressed".
Some scripts are built to automatically use the compressed files if no uncompressed versions are found.


## CSV Files

All raw .csv files in the entire project are excluded by .gitignore, except for compressed (.csv.bz2) files.


# Solution - Pushback to the Future
## Setup

1. This solution with the following steps is guaranteed to run on x64 Ubuntu Server 20.04. Although it is very 
likely to run on other operating systems as well.
2. Install Python 3.10.9
3. Install the following packages manually with pip:
   
   - `pandas==1.5.3`
   - `lightgbm==3.3.5`
   - `numpy==1.23.5`
   - `pandarallel==1.6.4`
   - `tqdm==4.65.0`
   - `scikit-learn==1.2.2`
   - `pydantic==1.10.11`
   - `flwr==1.4.0`
   - `matplotlib==3.7.1`
   - `torch==2.0.0`
   - `seaborn==0.12.2`
   - `tensorflow==2.12.0`

   or use `pip install -r requirements.txt`

4. Ensure that the "data" directory is located and formatted as specified in data/README.md

## Download Pretrained Models
The pretrained models and encoders should be placed in the assets folder. The default implementation with PyTorch model is provided in the assets, however can be substituted for an alternative TensorFlow model.
 
 
## Run Training
Run the script `inference/federated.py` if csv data was already compiled in train_tables and validation_tables and you need to re-train the model with different parameters, or `inference/main.py -t T` option, which will likely take many hours to complete, but will execute the entire pipeline, from raw data to the models, re-training the model, if needed, on the data not present in the submission.csv, but present in the _data directory. If any error encountered while running the training method during the main.py pipeline, which appeared to happen on certain enviroments, we recommend simply running the federated.py after appropraite intermediate csv files were already generated, which will train a model and then proceeding with `inference/main.py` command for an appropraite inference of the files contained in submission.csv

It will output multiple files, `model_{n}.pt` and `encoders.pickle` in the default assets directory. These will take about 500mb of storage. In the process, the train tables will also be generated and saved, these will take about 25 gb of storage. The guiding assumption with the models_n is that the best model would be chosen to continue inference, based on the values outputted by the server, however the detault model is model 15, because of 15 rounds of training, found to be sufficient to produce an MLP model of certain approximately comparable accuracy to lightgbm.

## Run Inference
1. Obtain a variable that contains the result of the `load_model()` function in `main.py`, by default this function is assumed to be the assets folder.
2. The function to make a prediction is `main()` in `inference/main.py`. It makes predictions for any number of flights,
but for only one timestamp and airport at a time. 
3. `main()` requires, among other things:
   - `models`: the tuple of models and encoders obtained in step 1
   - the raw data tables filtered by timestamp between the prediction time and 30 hours prior
   - `submission_format`: a dataframe of the flights and timestamps to make predictions for
4. Call the function with the required inputs and what will be returned is the `submission.csv` csv file
with the predictions in the `minutes_until_pushback` column.



