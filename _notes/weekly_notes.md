# Weekly Notes #

## Nov-6-10-2023 ##

*Goals for the week:*
- Pick rule for first manual database 
- Finish grad workshop presentation
- Create annotation table for manual database
- Generate plots of all spectrograms in that db 
- Edit selection table when errors found 

### Nov-7-2023 ###

Meeting w Fabio: 
- Look at annotation stats, discuss db
  - No one obvious way 
  - Few common things: 
    - can this model detect ringed seal calls regardless of place/time/etc?
      - train on everything, test only on ulu 2022 
    - is it important to work on locations and instruments?
      - reserve those for testing only 
      - trained on ulu, test on another location, etc 
      - works BUT also train on data from this location
        - take one location, 
  - Start:
    - use ulu and kk data, split 
      - can't show that it works in new places, show on new site, without any changes, see how it does, run and fine tune the model with data from that site
      - need an extra dataset that you didn't use for train/test, some for fine-tuning and some for testing
      - leave out pp and use cb for training too 
      - run detector trained on ulu, cb, kk -> on pp -> fine tuning phase with pp, couple epochs with pretrained model 
      - when pretrained model, take optimized values instead of random values, freeze some layers, start freezing from end of network, if that doesn't work, start unfreezing the last layers 
      - unfreeze the last layer, one or two epochs 
    - ulu and kk, how to split? 
      - table in db from each site, you can change the proportion of each in each batch
      - in ketos: create a batch generator for each table, then joint batch generator from those two 
      - pass the joint one to the model, when you create it you can set the proportions
      - this gives you a bit more freedom 
      - in the beginning: try to use the majority of kk data and top up with ulu, and then test with more ulu data


## Oct-30-Nov-3-2023 ##

*Goals for the week:*
- Create manual database 
  - Look at Najeem and Bill's email to see how to handle the improperly named filenames (DONE)
  - Create script that will output the number of annotations per month per site (DONE)
  - Create a graphic to help understand this - ie. histogram (DONE)
- Start writing thesis

### Nov-2-2023 ###

- Spoke w Bill about annotations, notes: 
  - Update script to monthly 
  - Pick a manual database rule and just go with it 
- Noticed Sachs Harbour data was missing, go look at the annotations provided by Bill 
- A table of metadata is appropriate for thesis

Q's For Fabio: 
- Can I test on a subsample of the data for testing the spectrogram parameters? 
  - I think previously we said because it trains so quickly to use the whole db, but I see Ruwan used a subsample
- Manual databases:
  - Have annotation stats now, what should I do? Very heavily biased towards Ulu 
- Double-check my code perhaps if we have time? Want to make sure I'm using the right stuff
  - Will update to ketos run command for the last step

### Nov-1-2023 ###
- Wrote script and output plots for annotation stats, both time and site dependent 
- Updated all dates to correct dates for PP, CB300, CB50
- Next step is to create a manual database
- Currently shows very heavily weighted to 2022 ulu data, discuss w Bill

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
