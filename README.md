# FLHuskies2 Solution

Created by the team FLHuskies2 out of the University of Washington Tacoma.

## Execution Steps

1. This solution with the following steps is guaranteed to run on x86-64 Ubuntu Server 20.04. Although it is very likely to run on other operating systems.
2. Install Python 3.10.8
3. Install the following packages with pip:
    
   - `pandas==1.5.3`
   - `lightgbm==3.3.5`
   - `numpy==1.24.2`
   - `pandarallel==1.6.4`
   - `tqdm==4.65.0`
   - `scikit-learn==1.2.2`
4. Ensure that the "data" directory is located and formatted as specified in data/README.md
5. Run the script `main.py`, it will likely take many hours to complete, but will execute the entire pipeline, from raw data to the models.


## Using The Model
1. Run the solution script to generate a table of processed features for input to the model.
2. The model requires two files, `encoders.pickle` and `models.pickle`.
3. `encoders.pickle` is a python dictionary with keys as the encoded columns and OrdinalEncoders as the values. 
Load this pickle file.
4. To prep processed data for prediction, loop through the keys (column names) of the encoders dictionary, 
replace the columns of the input data with the columns that are returned when the `fit()` function of the encoder is called with the
old column as the argument.
5. Now load the `models.pickle` file, it is also a python dictionary, but the keys are the airport names and the values
are the models. The data is ready to be passed into an airport model's `predict()` function.



