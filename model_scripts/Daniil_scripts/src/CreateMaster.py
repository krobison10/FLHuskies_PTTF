from src.CreateMaster import *
import pandas as pd

from src.ExtractFeatures import (LoadRawData, CrossJoinDatesAirports, ExtractAirportconfigFeatures,
                                 ExtractRunwayArrivalFeatures, ExtractRunwayDepartureFeatures,
                                 ExtractLampFeatures, ExtractETDFeatures,
                                 ExtractMfsFeatures, ExtractGufiTimestampFeatures, AddTargets, Adjust)


def CreateMaster(data_path: str, airports, start_time: str, end_time: str, with_targets=False) -> pd.DataFrame:
    """
    Loads all the raw tables from start_time to end_time into a dictionary 
    and sequentially extracts features for each information block merging it into
    a combined master table at the airport-timestamp level with a 15minute aggregation level
    
    :param str data_path: Parent directory where the data is stored
    :param List[str] airports: List indicating which airports to create the master table for
    :param str start_time: Timestamp to read from
    :param str end_time: Timestamp to read up to 
    :param Bool with_targets: Bool indicating whether to include the targets - only for training
    
    :return pd.Dataframe master_table: Dataframe at an airport-timestamp level with all the relevant features
    """
    
    # Load raw data and store it in a dictionary that maps airport + key -> pd.DataFrame
    raw_data = LoadRawData(data_path=data_path, airports=airports, start_time=start_time, end_time=end_time)

    # Create cross join of all the dates between start_time and end_time at a 15min frequency
    master_table = CrossJoinDatesAirports(airports=airports, start_time=start_time, end_time=end_time)

    # Extract features for the selected data blocks and append them to the master table
    master_table = ExtractLampFeatures(master_table, raw_data['lamp'])
    master_table = ExtractETDFeatures(master_table, raw_data['etd'])
    master_table = ExtractGufiTimestampFeatures(master_table, raw_data['first_position'], 'first_position')
    master_table = ExtractMfsFeatures(master_table, "KSEA")

    # Adjust master table in order not to have errors in edge cases in prediction time
    master_table = Adjust(master_table)

    # # In case we want the master table for training we include the targets
    # if with_targets:
    #     master_table = AddTargets(master_table)

    return master_table