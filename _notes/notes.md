# Notes for Detector #
(file started Sept 11)

## Q's for Fabio ##

1. How to evaluate performance metrics for the data?

______________________

## To do's ##

- Edit script so it drops selections with negative start times
- Understand Nyquist frequency and why the max plot is half the sampling rate
- Calculate spectrogram stats
- Create manual datasets
- Create a read the docs for all of my notes:
  - https://docs.readthedocs.io/en/stable/tutorial/
  - https://readthedocs.org/dashboard/
  - Logged in with github account
- Create yml file for ketos 2.7 and backup to git
______________________

## Completed ##

- Edit selection table creator to drop selections that end after the file has ended 
- Create script to grab audio files from the validation file for the audio folder 
- Generate a random split database with all of the annotations and see how that detector works

______________________

## Long Term Goals ##

- Thorough walk through of code and make sure you understand every single line 
- Look at Farid's SNR work
- Calculate statistics on the annotations (length, time of year, location, etc), make some nice figures
- Create manual database; go through the data and look at really good ones, talk to Fabio and Sebastien about this
- Run detector on the manual database 
- Run detector with different spectrogram configs  
- Look at new architecture with Sebastien  
______________________

## Spectrograms ##

- **Rate**:
    - OG rate for Cape Bathurst file is 48000Hz (Hz = 1/sec) 
    - Najeem was saying this is because you need two points to plot a waveform, and so with a 48000Hz sampling rate you could plot up to 24000Hz - understand this better  
    - Sampling rate --> max plot will be half of this --> Nyquist frequency
    - Notes from meeting with Fabio: Fabio says it might cause problems later if the input spectrograms have different sampling rates, and so by removing the rate I left it as the initial sampling rate, which could be different for each file
        - Librosa.load can load in a wav file and will return an array with the samples, and an int that is the sampling rate. If you don't want to resample the, you should pass sr=None as an argument to librosa.load
        - Need to have an integer number of spectrograms, the problem could have been I had a fraction number. If the audio loader gets to the end of the audio signal and there aren't enough samples to make a whole spectrogram, it will try to pad the fragment so it can make a spectrogram that is the same length as the others. The duration of the files will be the number of samples/sampling rate. So, if you load it like this:
        - sig, rate =librosa.load("file.wav", sr=none)
        - The duration will be
        - len(sig)/rate
        - if you divide that number by the duration of your model's input and the remainder is not zero, it means that there aren't enough samples to make another spectrogram. In that case, the audio loader can try to pad so you get a whole spectrogram. But you can also pass pad=False to the AudioFrameLoader, which will just discard the last bit of audio if it's not enough to make a spectrogram of the specified length
        - Me:
        - I set the duration of the spectrogram to be 2 seconds in the config, and the rate, and some other parameters. The wav file is converted to a spectrogram using the config, and if it can't be divided up equally into spectrograms defined by the config, it will try to create a padded spectrogram for the last one, unless I set pad=False in the AudioFrameLoader. And in your opinion, is pad=False a risky one in case there is an annotaton right at the end of the wav file?
        - You got it mostly right: the only thing is that the audio is split and then spectrogram are computed. But it's practically the same thing. I think pad=False is fine, specially with short spectrograms like 2s. (the max you'll loose is 2s at the end of a file)
        - Another thing that would help would be to filter out any selections that don't fit the file before you create the database.
        - Looking at your error message again, it looks like some selections fall outside the boundaries of the file. For example, there's one that starts at 300s and ends at 302s, but maybe that file is only 300s long? If all your files are the same length, it is really easy: you can just drop any selections with a start time smaller than 0 and and end time greater than the file duration. But if the files have different durations, you might want to be more specific and check that the selections are within the file on a file by file basis 11:12 we should add an option to do that when creating the selection tables 
    - Augmentation techniques currently: making multiple spectrograms from one original one by snapping it in different locations in the frame  
______________________

## Architecture ##

- Fabio was saying this is a "data centric" problem and should focus more on the data than the model architecture  

______________________

## Data ##

- Fabio was saying that I should make manual datasets where the training and validation sets don't have the same sites or times so it's not biased to understanding the temporal information

______________________

## Ketos Functions ##

- Sl.select: selects portions from the wav file according to the annotation file and several other available parameters. An augmentation technique. This makes multiple spectrograms from one annotation, with a step size through the annotation and a minimum overlap of the annotation in the newly generated spectrogram. For the validation set, remember if you do augmentation that your results could be biased, example if a good bark is being detected in multiple spectrograms because it's been multiplied, could skew to higher results for precision and recall than actually because the sample is being seen more than once.
- Sl.create_rndm_selections: need to make sure the file durations file is only looking at wav files that exist within the annotations file, or else it'll think that the files with no annotations are all noise segments which isn't true.
- BatchGenerator: refresh on epoch end, if you organized the samples in a specific way, don't shuffle
- Process: need to make sure that annotations aren't being counted twice, maybe the group keyword? Look into this more
- Create_detector: for a 2 second duration, the maximum step size should be 2 seconds. If you do 3, you'll be missing 1 second of data. You'd want to do a step size smaller than 2 seconds typically because that would mean you wouldn't miss calls that are sitting on the border of two spectrograms, but this also increases duplicate counting. Need to merge detections so we're not double counting. The buffer adds a certain amount of time to the section it's looking at. The batch size is purely for computational reasons, instead of loading the entire wav files which are ginormous you can load X spectrograms at a time. Can probably remove batch size from my scripts because I have so little data  
- Augmentation in create database step: for training, we want more samples than we have. could do the same type of augmentation, but need to keep in mind that duplication is possible in the performance metrics. same thing for the mistakes. if you want to use this model for individual calls, might be easier to not use augmentation. for a binary detector, ok to do augmentation.
______________________

## Selection Tables ##

- Sept 12 2023: ST2, Site 27, found minor error where yelps where marked as 'keep for detector', fixed and updated repo 
- Ulu files: original subfolder structure had multilayers, updated and renamed files to have the site name and station at the front of each file 
  - because of this, had to change the filename and begin path in each annotation file, done by script. This is why they say "path updated"
______________________

## Training Definitions ##

- train: used in training
- validation: used in training
- test: NOT USED IN TRAINING