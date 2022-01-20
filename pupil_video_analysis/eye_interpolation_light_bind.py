from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from nptyping import NDArray
import glob

from pandas.core.frame import DataFrame

Bodyparts = str
Coordinate = str
Model = str
DLCKey = Tuple[Model, Bodyparts, Coordinate]


def remove_scorer(data: pd.DataFrame):
    newnames = list(map(lambda c: (c[1], c[2]), data.keys()))
    idx = pd.MultiIndex.from_tuples(newnames)
    return data.set_axis(idx, axis=1)


def replace_low_likelihood_nan(data: pd.DataFrame, threshold: float) -> NDArray :
    likelihood_over_threshold = data > threshold
    return np.array(likelihood_over_threshold[likelihood_over_threshold])


def contains_x(key: DLCKey):
    return "x" in key


def contains_y(key: DLCKey):
    return "y" in key


def contains_likelihood(key: DLCKey):
    return "likelihood" in key


def make_dataframe_before_interpolated (outlitter_to_nan : NDArray,idx_list : list) -> pd.DataFrame:

    Dataframe_before_interpolated = pd.DataFrame(data = outlitter_to_nan)
    idx = pd.MultiIndex.from_tuples(idx_list)
    Dataframe_before_interpolated.set_axis(idx, axis=1,inplace=True)
    return Dataframe_before_interpolated


def as_output_filename(h5_path: Path,data_name: str):
    parent = h5_path.parents[0]
    stem = h5_path.stem.split("DLC")[0]
    filepath_without_extension = parent.joinpath("interpolated_data").joinpath(
        stem)
    return str(filepath_without_extension) + data_name + ".h5"

#add
def is_light(key: DLCKey):
    return "FT10_light" in key or "peak_light"  in key


if __name__ == '__main__':


    h5p = "D:/Niki-pupil/pupil_video_analysis/pupil_video_analysis/videos_h5/"
    h5s = glob.glob(h5p+"*.h5")
    likelihood_threshold = 0.9

    for h5 in h5s:

        print(f"processing " + h5)
        
        tracked_data = pd.read_hdf(h5)
        tracked_data = remove_scorer(tracked_data)

        x_idx = list(filter(contains_x,tracked_data.keys()))
        y_idx = list(filter(contains_y,tracked_data.keys()))
        likelihood_data = list(filter(contains_likelihood,tracked_data.keys()))
        
        light_idx = list(filter(is_light,tracked_data.keys()))
        light_x_idx = list(filter(contains_x,light_idx))
        light_y_idx = list(filter(contains_y,light_idx))
        light_likelihood_idx= list(filter(contains_likelihood,light_idx))
        
        light_x_data = tracked_data[light_x_idx]
        light_y_data = tracked_data[light_y_idx]
        light_likelihood_data = tracked_data[light_likelihood_idx]

        x_array = np.array(tracked_data[x_idx])
        y_array = np.array(tracked_data[y_idx])
        likelihood_array_replaced = replace_low_likelihood_nan(tracked_data[likelihood_data],likelihood_threshold)

        x_outlitter_to_nan =  x_array * likelihood_array_replaced
        y_outlitter_to_nan =  y_array * likelihood_array_replaced
        
        x_Dataframe_before_interpolated = make_dataframe_before_interpolated(x_outlitter_to_nan,x_idx)
        x_Dataframe_before_interpolated[light_x_idx] = light_x_data

        y_Dataframe_before_interpolated = make_dataframe_before_interpolated(y_outlitter_to_nan,y_idx)
        y_Dataframe_before_interpolated[light_y_idx] = light_y_data

        data_before_interpolated = pd.concat([x_Dataframe_before_interpolated,y_Dataframe_before_interpolated,light_likelihood_data ],axis=1)

        data_before_interpolated.sort_index(axis = 1,inplace=True)

        data_after_interpolated = data_before_interpolated.astype("float64").interpolate()
        
        #add



        output_path_nan = as_output_filename(Path(h5),"_outlitter_to_NaN")
        output_path_interpolated = as_output_filename(Path(h5),"_interpolated")
        output_path_light = as_output_filename(Path(h5),"_light")
        
        data_before_interpolated.to_hdf(output_path_nan,"key")
        data_after_interpolated.to_hdf(output_path_interpolated,"key")



 
 
 
