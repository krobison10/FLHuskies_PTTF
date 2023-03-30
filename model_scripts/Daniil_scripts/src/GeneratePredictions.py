from src.CreateMaster import *
import pandas as pd


def GeneratePredictions(to_submit: pd.DataFrame, models_path: str,
                        master_table: pd.DataFrame) -> pd.DataFrame:
    """
    Loads all the models from models_path and generates a dataframe at an airport-timestamp-lookahead-config 
    level indicating the likelihood of the configuration being active at the given airport, timestamp
    and lookahead minutes ahead of time. The models utilized are Catboost objects trained independently
    for each airport
    
    :param pd.DataFrame to_submit: Dataframe containing the desired resulting output to be saved
    :param str models_path: Parent directory where the models are stored
    :param pd.Dataframe master_table: Features to generate the prediction over at an airport-timestamp level
    
    :return pd.Dataframe to_submit: Dataframe containing the grid with all the predictions
    """

    # Initialize empty placeholder to store all predictions
    predictions = pd.DataFrame()

    # Iterate through each airport to append the predictions
    for airport in to_submit['airport'].unique():
        to_predict = master_table[master_table['airport'] == airport].copy()

        for l in LOOKAHEADS:
            
            current_model = CatBoostClassifier()
            current_model.load_model(os.path.join(models_path, 'version_9', f'model_{airport}_target_{l}'))
            classes = current_model.classes_
            
            probabilities = current_model.predict_proba(to_predict[current_model.feature_names_])
            current_predictions = to_predict[['airport', 'timestamp']].copy()
            current_predictions['lookahead'] = l

            # Add the predictions for every class as a new column
            for i in range(len(classes)):
                current_predictions[classes[i]] = probabilities[:, i]
                
            # Stack the lookahead columns into a single column
            current_predictions = current_predictions.set_index(
                ['airport', 'timestamp', 'lookahead']).stack().reset_index()
            current_predictions.columns = ['airport', 'timestamp', 'lookahead', 'config', 'active']

            # Concatenate current predictions with the previously computed ones
            predictions = pd.concat([predictions, current_predictions])

    predictions = predictions.groupby(['airport', 'timestamp', 'lookahead', 'config']).first().reset_index()

    to_submit = to_submit.merge(predictions, how='left', on=['airport', 'timestamp', 'lookahead', 'config'])
    
    # Adjust submission in case there are additional configs in submission format file
    to_submit['active'] = to_submit['active'].fillna(0)
    
    cat_sum = to_submit.groupby(['airport', 'timestamp', 'lookahead']).active.sum().reset_index()
    cat_sum.columns = ['airport', 'timestamp', 'lookahead', 'cat_sum']
    
    to_submit = to_submit.merge(cat_sum, how = 'left', on = ['airport', 'timestamp', 'lookahead'])
    to_submit['active'] = to_submit['active'] / to_submit['cat_sum']
    to_submit.drop(columns = 'cat_sum', inplace = True)
    
    return to_submit



def BuildDummyPrediction(sub_format_path, prediction_path):
    """
    Saves the same file as in submission format path in the prediction path
    """
    
    DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
    predictions = pd.read_csv(sub_format_path, parse_dates=["timestamp"]).copy().reset_index(drop=True)
    predictions.drop(columns = 'active', inplace = True)
    
    probs = (1 / predictions.groupby(['airport', 'timestamp', 'lookahead']).size()).reset_index()
    probs.columns = ['airport', 'timestamp', 'lookahead', 'active']
    predictions = predictions.merge(probs, how = 'left', on = ['airport', 'timestamp', 'lookahead'])
    
    predictions.to_csv(prediction_path, date_format=DATETIME_FORMAT, index=False)