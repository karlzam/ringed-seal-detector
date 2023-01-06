## Repo for K. Zammit's Work on Ringed Seal Detector ##

## call_script.py ##

- Calls MERIDIAN based functions from meridian_utils.py to create a database, train a classifier, and create a detector
- User must set pathways to where things live
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
- Need to add spectrogram function 
______________________

## sort_data.py ##

- My OG attempt at creating annotation csvs, but I did it sheet based instead of one sheet 
- This has been replaced by create_annotation_table_all.py
______________________

## recipe.json ##

- Recipe file for ResNet training from MERIDIAN tutorial
______________________

## spec_config_ruwan.json ##

- Ruwan's spectrogram config file 
______________________


