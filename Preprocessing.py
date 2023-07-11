import os
from pathlib import Path as path
import pandas as pd
import numpy as np

# class - Preprocessing:
# a class that encapsulates some common functions to preprocess the data
# variant classes can be inherited if any customized functions need to be implemented
# all functions will be declared as @classmethod thus they can directly be accessed without any instantiation
class Preprocessing:

    # file structure:
    # src -> all python source files
    # src/data -> data folder
    @classmethod
    def load_data_csv(cls, filename:str, param_delimiter) -> pd.DataFrame:
        print("-- loading dataset ...")
        # get file path - this should work on Windows / linux / Mac
        currentdir = os.path.join(os.path.dirname(__file__))
        filepath = currentdir + "/data/" + filename

        filename = path(filepath)
        if not filename.exists():
            print("-- file doesn't exist ")
            return pd.DataFrame() # if file not found return an empty dataframe
        else:
            print("-- found file ")
            data = pd.read_csv(filename, delimiter=param_delimiter)
            print("-- done")
            return data


    @classmethod
    def save_data_csv(cls, df:pd.DataFrame, filename:str):
        print("-- saving dataset ...")
        # get file path - this should work on Windows / linux / Mac
        currentdir = os.path.join(os.path.dirname(__file__))
        filepath = currentdir + "/data/" + filename

        filename = path(filepath)
        df.to_csv(filename, index=False) # index = False, exclude the row index
        print("-- done")


    @classmethod
    def data_info(cls, data:pd.DataFrame) -> None:
        # print some basic info of the dataset

        # overview
        print("overview of the dataset: ")
        print(data.shape)

        # the first 5 lines
        print("the first 5 lines of the dataset: ")
        print(data.head(5))

        # print the column names
        print("the column names: ")
        print(data.columns)

        # print info about each field
        print("info about each field")
        print(data.info())


    # get missing data info
    # print the missing data info to the console
    # @return: None
    @classmethod
    def get_missing_data_info(cls, data:pd.DataFrame) -> None:
        print("missing data info about the dataset: ")
        print()

        # get the fields with NaN values and sort
        total_missing_data_info = data.isnull().sum().sort_values(ascending=False)
        print("field - with - NaN - values ----------------------------------------------------")
        print(total_missing_data_info)
        print()

        # get the percentage of the NaN fields
        # percentage allows us to apply different filling strategies, i.e., fill with median / remove the NaN records
        missing_data_percentage =(data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
        missing_data = pd.concat([total_missing_data_info, missing_data_percentage], axis=1, keys=['Total numbers', 'Percentage'])
        print("info about missing_data ---------------------------------------------------------")
        print(missing_data)
        print()
        

    # get missing attribute list
    @classmethod
    def get_missing_attribute_list(cls, data:pd.DataFrame) -> list:
        total_missing_data_info = data.isnull().sum().sort_values(ascending=False) # total_missing_data_info -> type: pandas.core.series
        
        # add missing attributes to a temp list
        missing_attribute_list = []

        # specify null value, usually it is equal to 0 -> 0 can be int/float/double
        # it should however be noted that, this threshold can be altered if users want to get rid of rows with low occurrences        
        NULL_VALUE_THRESHOLD = 0
        EPSILON = 1e-8 # if more accuracy is desired, for example 1e-15 can be used
        for Attri in total_missing_data_info.index:
            if(abs(total_missing_data_info[Attri] - NULL_VALUE_THRESHOLD) > EPSILON):
                missing_attribute_list.append(Attri)
        
        # now let's judge if the list is empty
        if not missing_attribute_list:
            print("no missing attributes found, will return an empty missing attribute list")
            return missing_attribute_list
        else:
            print("missing attribute list constructed, the following list will be obtained")
            print(missing_attribute_list) # print functions can slow down the efficiency a bit, now let's just use it to prompt some info
            return missing_attribute_list


    # WARNING:
    # this function is used for removing ALL rows which contain empty attributes
    # extra care needs to be taken -> some attributes are not really 'NaN', there can be intervals
    # i.e., an attribute can be recorded every for example 10 rows
    # process NaN values in the dataset (if any) according to the missing attributes
    # @return: pd.DataFrame
    @classmethod
    def process_missing_data(cls, data:pd.DataFrame, mode:str='REMOVE') -> pd.DataFrame:

        # different mode enables different filling operations
        match mode.strip(): # mode may contain spaces, i.e. mode='REMOVE '
            case 'REMOVE':
                # warning 
                print("WARNING: -----------------------------------------------------------------------")
                print("process_missing_data_removing is being called")
                print("this function will remove all rows with NaN values, please proceed with caution")
                print()

                # get missing_attribute_list
                missing_attribute_list = Preprocessing.get_missing_attribute_list(data)

                # if missing_attribute_list is empty / not empty
                if not missing_attribute_list:
                    print("no missing attributes provided, will not process missing data")
                    return
                else:
                    print("missing attribute provided, will proceed")

                    # before removing
                    print("missing data info BEFORE NaN values removed --------------------------------------")
                    missing_data_info_before = data.isnull().sum().sort_values(ascending=False)
                    print(missing_data_info_before)
                    print()

                    # remove the missing data
                    # for each Attri in missing_attribute_list, remove the rows which contain them
                    for Attri in missing_attribute_list:
                        print("removing all rows with {0} == NULL".format(Attri))
                        data = data.drop(data[data[Attri].isnull()].index)
                    
                    print("done")

                    # validate                   
                    print("missing data info AFTER  NaN values removed --------------------------------------")
                    missing_data_info_after = data.isnull().sum().sort_values(ascending=False)
                    print(missing_data_info_after)
                
            case 'MEDIAN':
                # filling the NaN data with median values
                pass
            case 'AVG':
                # filling the NaN data with avg values
                pass
        
        # end of match
        return data


    # process the duplicate data in the dataset
    # @return: a new dataframe
    @classmethod
    def process_duplicate_data(cls, data:pd.DataFrame, print_info:bool = False) -> pd.DataFrame:

        # the info printed by data.duplicated() may not be very useful for the users
        # since it just shows columns and true/false
        # thus this function should allow users to select whether to print/not print the duplicated info
        if print_info:
            print("info about the duplicate data in the dataset: ----------------------------------------")
            print(data.duplicated())
        
        # return the new dataset after removing duplicates
        return data.drop_duplicates()


    # process outliers
    # @param: multi - a multiplier, elements greater than multi times the standard deviation are considered outliers
    @classmethod
    def process_outlier_data(cls, data:pd.DataFrame, multi:float=3.0) -> pd.DataFrame:
        print("basic statistics info: ------")
        print(data.describe())

        # for the fields which are numbers, the following script will get rid of the records which are bigger than
        # multi times of its std values
        # \example
        # data = pd.DataFrame(np.random.randn(1000,3))
        # print(data)
        # print(data.describe())
        # print(data.std())
        # print(data[(np.abs(data)>(3*data.std())).any(1)])
        # \example
        # it should be noted -> only for attributes whose value type are numerical numbers

        # validation
        print("before the shape is {0}".format(data.shape))
        print("after the shape is {0}".format(data.shape))

        return data

    
def main():
    # entry point
    data = Preprocessing.load_data_csv("Mobiliteitstrend__per_rit_en_motief_18102022_113630.csv", ";")
    print(data)
    #Preprocessing.data_info(data)

    # get info about missing data
    Preprocessing.get_missing_data_info(data)

    # remove missing data
    data = Preprocessing.process_missing_data(data, mode='REMOVE')

    # process duplicate data
    data = Preprocessing.process_duplicate_data(data)
    print(data.duplicated()) # validate
    print(data)

    # process outlier data
    #data_without_outliers = Preprocessing.process_outlier_data(data_without_duplicates, multi=3.0)
    pass

if __name__ == "__main__":
    main()   
    pass
    