## Repo for K. Zammit's Work on Ringed Seal Detector ##

## Main Folder: 
- contains jupyter notebooks of most up to date model runs 
- dev-script.py is used by KZ when needs to do quick debugging or development 
- detector_cli.py will be where I develop the command line interface 
- deploy-detector-new-data.ipynb is jupyter notebook for applying an already trained model file on new data 


## Folder: _config ##

- recipe files and config files 
______________________


## Folder: _dev_scripts ##

- original scripts used to develop functions and tools, no longer used 

	- Dev scripts are: 
	# 1_create_database.py ##
	- creates database .h5 file from annotation csvs and spectrogram config
	- calls meridian_utils create_database function 
	- needs the "all_file_durations.xlsx" file which is currently hardcoded from the input folder
	- currently forces all labels to be 1 for "bark" 
	- makes annotation tables into ketos format, generates selection tables with set duration, generates negative segments 
	- creates audio segments of defined length, drops segments passed the end of the file, and then creates spectrograms 

	## 2_train_classifier.py ##
	- creates a .kt model 
	- uses spectro config, database file, recipe file 
	- calls meridian_utils train_classifier function which uses ResNet structure

	## 3_create_detector.py ##
	- creates a detections.csv file for the audio files within the "audio" folder 
	- uses the trained model and audio .wav files 
	- calls meridian_utils create_detector function 

	## 4_performance_metrics.py ##
	- outputs an excel sheet with a column added to the original annotations file if that annotation was detected or not 
	- uses "if the detection time encompasses the annotation time, call it true", so the detection could be much longer than the annotation


	Steps: 
	1. Concatenate selection tables and add "site_name" column 
	2. Format the table: calculate start and end times, rename some columns, add annotation_id (number of annotation within the same file)
	3. Split files into train, validation, and test (currently randomly)

	## meridian_utils.py ##
	- houses all meridian based functions to create database, train classifier, and create detector, and steps necessary to do so

## Folder: _lockbox ##
- all annotation tables (original and edited)
- original_file_locations.xlsx: original locations of files on drive
- all_file_durations.xlsx: file durations, used by create database 
- manual database selection tables and 
______________________


## Folder: _notes ##

- folder with my notes in markdown so the readme file doesn't get confused
______________________


## Folder: _tests ##

- previous jupyter notebooks with model runs that are not ideal or old
______________________


## Folder: _tools ##

- useful tools for development 

1. annotation_stats: to plot figs about annotations 
2. create_annotation_csvs.py
	- creates annotation tables from raven tables 
	- inputs are output directory for tables, pathway to folder containing all selection tables, and output filenames 
	- outputs .csv files and a .xlsx with a list of all annotations 
3. create_noise_segments: generates noise segments and plots the spectrograms for inspection
4. manual_database: dev script used to generate splits for datasets from annotation tables 
5. spectrogram: calls spectrogram utils which segments audio and plots spectrograms according to json file 
6. spectrogram_utils: see above
7. to_raven_format: takes detector output and converts to format readable by raven 
8. toolbox: calls functions: calc_file_durations, plot_call_length_scatter, write_file_locations, rename_ulu_2022_files, inspect_audio_files
9. toolbox_utils: see above 
______________________
