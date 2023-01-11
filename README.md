## Repo for K. Zammit's Work on Ringed Seal Detector ##


## excel_outputs directory ##

- Stores annotation tables to work with, outputs from create_annot_table_all
______________________


## figs directory ##

- Stores plots for visualizing data
______________________


## input_files directory ##

- Has recipe and spectro gen files
______________________


## call_script_meridian_utils.py ##

- Script to edit to call functions within meridian utils
______________________


## call_script_pretrain_utils.py ##

- Script to edit to call functions within pretrain utils
______________________


## create_annot_table_all.py ##

- Most recent script to create an annotation table from the selection tables provided
- Outputs two different excel sheets, which are in the "excel_outputs" folder 
- "all_annot.xlsx" is all annotations in one sheet before any formatting
- "formatted_annot.xlsx" is formatted annotations: 
	- Added in columns for start, end, label, annotation ID, filename
	- start time is equal to file offset time from original files 
	- delta time was recalculated for all because was missing for some, as end time - begin time
	- end time was the file offset + delta time
- "formatted_annot_manual.xlsx":
	- manually deleted unnecessary columns, this sheet can be used to generate mini test runs
______________________


## create_dataset_from_annot.py ##

- First go at splitting a dataset from annotation tables. 
- Randomly shuffles the dataframe, and then takes the first bunch to be training, and the remainder to be testing
- The split percentage is defined in the input command 
- Generates training and validation sets and copies them to a training and validation folder
______________________

## meridian_utils.py ##

- Basically the MERIDIAN tutorial
- Create a database, create a detector, train a classifier functions 
______________________

## pretrain_utils.py ##

- Functions to compute statistics and visualize data before creating database for training
- TODO: Spectrogram plotting
______________________

## sort_data.py ##

- My OG attempt at creating annotation csvs, but I did it sheet based instead of one sheet 
- This has been replaced by create_annotation_table_all.py, can be removed from repo 
______________________


