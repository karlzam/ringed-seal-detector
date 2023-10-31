# Weekly Notes #

## Oct-30-Nov-3-2023 ##

*Goals for the week:*
- Create manual database 
  - Look at Najeem and Bill's email to see how to handle the improperly named filenames 
  - Create script that will output the number of annotations per month per site 
  - Create a graphic to help understand this - ie. histogram
- Start writing thesis

### Oct-31-2023 ###
- Read COD paper (DONE)
- Read NARW paper again
- Start writing thesis

Q's for Fabio: 
- Do you use anything fancy for hyperparameter tuning?
  - write in a command line interface 
- Have you heard of clearML? 
  - MLFlow, DVC tools to help track ML dev, Ruwan didn't use these, good if you're doing big production models
- Doublechecking work? 
  - Create list of parameters in scripts that need to be thought of 
- Have thought about using YOLO to draw boxes instead? Guessing it doesn't work great because of the nature of our data?
  - Sometimes don't need it for the specific project 
  - Blurry borders make it hard
  - Inexact annotations make it difficult 
  - Would need longer duration segments for the input data 

## Oct-16-20-2023 ##

**Week Goals**:
- Create manual database and run with 2 sec duration spectro

### Oct-18-2023 ###
- Before left for CAA, had just run test with multiple spectro durations, seems to have worked! 
- Just finished updating the script so it also runs the "create-detector" portion of the script 

To Do's for Today: 
- Create yml file for ketos 2.7 and backup (DONE)
- Look at Najeem and Bill's email to see how to handle the improperly named filenames 
- Create script that will output the number of annotations per month per site 
- Create a graphic to help understand this 
- Create idea for how to split the data

Pasting in from notes: 
Notes from meeting w Fabio:  
Can use the same noise segments each time. Save the selection table for them  
For doing spectrogram tests, can use the whole dataset instead of a subset because don't have a lot of data points  
For test set, 10-15% is fine because we don't have a lot of data. The smaller the test set, the less realistic it is, and therefore maybe not as comprehensive of a detector  
For adding more data: do you want a better detector with more data, or save more data to have a more comprehensive test on a less ideal detector? Should always add in the data that is available.  
Can alter the block sizes and layers through the recipe file. But, because of our data, there is always room to improve the data making it a more data-centric problem than a model-centric problem. Should spend more time improving the data than the model. You'll get a lot more improvement on improving the dataset than just the architecture of the model. Should compare with other model architectures too, but probably won't be as important as the data. DenseNet already exists in ketos in a simpler and smaller form. Most bioacoustic applications have lots of room for improvement in the data.  
Can calculate spectrogram stats for one frequency range, unlike what Ruwan did, because there isn't really a spread for RS barks.  
Comparing with more classical ML models: SVM, RF, linear discriminant analysis etc do better with a smaller number of features for structured data. These do not do feature engineering for you like in DL models, so you'd need to determine the important features of the bark and pass those into the models (ex. Pk freq, max amp, etc). Requires a lot more knowledge about RS calls and possible variations in the calls. If you change the environment for new data, it might not work as well if the carefully engineered parameters are different. Using RF, etc, on some derived parameters might be better than trying to use the spectrogram as input. DL removes the need for initial feature engineering, with the promise that the network can learn the important features on it own. In reality, we still do some feature engineering, like converting the wavforms into spectrograms first.  
Data augmentation techniques when we don't have a lot of data:  
Creating new samples by overlaying a good bark on a background sample. Prefer to do this in the time domain and then just chunk it in there. Could be an ideal method for RS because the sames are short. Risk is artifacts, could put those artifacts in the noise segments to train it against recognizing them. Use a package called "kayra" for this.  
Synthesize the call with a mathematical function, then can add it in to spectrograms 
"create detector" comes from training a classifier, and then USING that classifier to act like a detector  
