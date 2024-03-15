# Weekly Notes #

## Mar 11-15

- The Adam optimizer has a sort of learning rate scheduler built into it, so it's not a huge deal to not 
- adjust the learning rate for fine tuning, and it's not trivial in the ketos package
- query in python: https://www.pytables.org/usersguide/condition_syntax.html
- Learn SQLite to get a bit of experience: https://en.wikipedia.org/wiki/SQLite

- For citing ketos: 
  - as a website documentation 
  - can describe model framework without ketos 
  - say i used this package to do this
  - for the tonal noise reduction, say i used this function
  - include code with publication then don't have to explain as much


Script to get hdf5 data: 

    file = r'E:\baseline-with-normalization-reduce-tonal\final-baseline-db-normalized-rtn.h5'
    
    h5file = open_file(file, mode="r")
    
    table = h5file.get_node('/test/data')
    
    # by row index
    # table[0]['label']
    # table[0][['label, data']
    
    # to find the id
    # loop through all the rows, add a condition, if this row id is in my list, do something to data
    # save this data to a file as you go
    
    ids = [1, 2, 3, 4]
    rows = []
    for row in table:
        if row['id'] in ids:
            rows.append(row['data'])
            # create a function that plots the data, remove the append
            if row['id'] == ids[-1]:
                break


## Feb 19 - 23

### To Do's: 
- Try adding noise 
- Try the enhance signal stuff and reduce tonal noise 
- Fine-tuning the best baseline model 

Questions for Fabio: 
- Adding noise made everything worse, have you seen this before?
- Going over Farid's SNR calculations, why does he do a left and right calculation? 
- Would I be able to try your thesis method with the different representation for spectrograms?

## Feb 12 - 16 

### To Do's:
- Finish normalization stuff 
- Add noise to spectrograms and apply to baseline and one with normalization 

### Notes & Q's:
- Should I be normalizing the wav form or the spectrogram?
- In ketos, how to do this specifically? 

Sebastien says for each epoch to add a different type of noise to the spectrograms. 

Adding noise into the database directly: 
- database_interface.py: 
  - line 940, x = next(loader), x.data.data returns the array 
  - line 1194 also accesses the raw data array to write to the file 
  - would need to understand how these two different write functions interact 

## Feb 5 - 9 ##

### To Do's:
- Start writing for two hours a day 
- Do saliency maps 
- Look at normalizing the data 
- Finish fine-tuning stuff and figure out how to unfreeze blocks 

Looking at normalization:
- the baseAudio transforms are: self.allowed_transforms = {'normalize': self.normalize, 
                                   'adjust_range': self.adjust_range}
- for a magnitude spectrogram, this is updated to also include: self.allowed_transforms.update({'blur': self.blur, 
                                        'enhance_signal': self.enhance_signal,
                                        'reduce_tonal_noise': self.reduce_tonal_noise,
                                        'resize': self.resize})

-Default normalize statement: Normalize the waveform to have a mean of zero (mean=0) and a standard deviation of unity (std=1) before computing the spectrogram. Default is False.

## Jan 29 - Feb 2 ##

### To Do's: 
- Send Elise & Bill data 
- Figure out how to load model 
- Figure out how to access last layer to do a t-sne plot 
- Do saliency maps and t-sne

Q's for Fabio: 
- Confirm fine-tuning was done correctly? 
- How to load model? 
  - Specifically, for saliency plots and t-sne of fully connected layer representation
    - t-sne: it'll be an array with 2 entries right, one for each class? and then the softmax layer takes it and converts it into the confidence score? 
- Noticed the raw output vs the not raw output looks to be swapped after going through the "transform function"? Why? 
- I understand it's a subclassed model from keras, but when I try to use the functions listed online, you need a 
"model.input", which doesn't look to be defined, and if I try to get the output from an intermediate layer, it says it's not connected and so there's no output
- I understand you need to call the model on an instance before you can get the weights, which I've accessed 
- How to load checkpoint files?

Meeting w Fabio: 
Fine Tuning

- Replace top: keeps feature extraction layers, replaces the classification head
- Freeze & unfreeze blocks: gives more control
- Run it longer, try a different learning rate potentially if you need to
- LR: test if showed some problems, if loss was oscillating a lot the learning rate might be too large, leave this as the last thing to do bc it is working
- Fabio would do: test the model, measure the performance compared to the original before you try other things, repeat that measurement for each thing you try
- Try to unfreeze some feature extraction blocks starting from the last one, which is the closest to the classification layer that you replaced
- If you need to unfreeze more layers, the feature extraction layers might not be optimal for the new dataset, the deeper you go, the more drastic the difference will be - could also take the model and retrain it entirely from the loaded model
- The less you freeze, the more the optimizer has freedom, you might end up with a model that is different than the original - this probably wouldn’t be a problem in my case
- Might get to a point where the performance isn't so different than the original dataset - performance improvement might plateau and see how far you can go
- You can go a bit more specific and unfreeze specific layers within a block, but that probably won't make a difference, try the others first
- "unfreeze block 6", you can look at the code and see the individual unfrozen layers in there, up to block level there are methods
 

Accessing Intermediate Layers

- T-sne: rough idea of where things are, there is a problem with this kind of analysis, the dimensionality reduction, trying to visualize HD into two or three, mapped projections, you always lose something, all of these have the same principal where you need the feature map to generate the plots
- Model.model.layers
- Resnet ketos arch obj -> model is the tensorflow loaded model -> has attributes
- Layers is nested: layers.layers[1].layers, etc
- Could create a new tensorflow model and create a list of new layers, taking layers from trained model
- My_layers = model.model.
- "get feature extractor" method sort of already gets a list of layers
- Right before the fully connected layer, take that to features
- Tf.keras.models.Sequential(pre_trained_model.model.layers[0:4])
- Feature_extractor.trainable = false
- This is an instance of the sequential class, this will be a tensorflow model
- You'll get 32x128 (batch size, last conv layer)
- To call it with an input:
- Feature_extractor(inputs) where the input is the spectrogram, the same thing you pass to the entire model
 

Saliency maps:

- Pass input through model, get the value of the outputs at that layer, before you go to the next layer, you take that and reshape that into a 2D array and plot as a heat map, each of those values you have will map to a time-frequency bin
- Basic idea: plot which neurons are activated when you pass through a certain layer, each element of the activation can be mapped into the input
- Start with 256x256, map 16x16, if that is activated, it means that the values that are in the corresponding area are making that activate, that’s an indication that info was important
- Parallel: where was the model looking in this image to call it a dog or a cat
- "what makes that layer activate, and how do you measure it?" - the weight of the parameters, each of the 32x32 pixels of the output of the convolutional layer, has a weight and a bias to it, you can look at the values
- You need to plot the weights and also the values of the output
- Often doesn't look great because you get really sharp changes, often people use lots of image processing techniques to use kernel smoother to make a transition look nicer
- Write a function, takes a trained model in, specify which layer you want to look at, assemble up to that layer, call layer on inputs you provide, gives output of [batch size, dimensions], project that back to the original dimensions to see what part overlaps with the original image, plot with the same axis as the input

KZ Notes from script: 
# fully connected layer weights
    # model.model.layers[4].get_weights()

    # internet says you can get the output of the layers by doing:
    # inp = model.model.input
    # outputs = [layer.output for layer in model.layers]
    # BUT, I get "Layer res_net_arch_1 is not connected, no input to return."

    # this returns the output of the softmax layer - just want to load one before that
    #model.run_on_batch(batch_data['data'], return_raw_output=True)

    # could be useful: https://github.com/philipperemy/keract

    # to run on one thing:
    #output2 = model.run_on_instance(batch_data['data'][0], return_raw_output=False)
    # returns 0: [0], 1: [0.8529221]

    #output2 = model.run_on_instance(batch_data['data'][0], return_raw_output=True)
    # returns array, [[0.8529221 0.14707792]]

    #... interesting why are these flipped?

    # So we've used the "subclassed Model" keras type, not the functional API



## Jan 15 - 19 ##

To Do's: 
- Manually check the remaining negatives for the not centered results 
- Rerun with balanced samples 
- Run with multiple different seeds and fix the ensemble script
- Test different spectrogram parameters
- Talk to Fabio about post processing steps 
- Show results for randomly generated noise samples 
- Try fine-tuning for PP with the best generated result

## Last Week of Dec ##

- Post-processing detector performance: 
  - centered signals
    - overlap increases the chance of capturing signal in favourable window
    - merge consecutive detections, apply filtering etc
- Fine tuning stuff:
  - instead of building from recipe, just load it
    - add a new classification head: take the weights from the pretrained model, create a new classification head and train the model again.
      - Here is where you pick how much you want to freeze 
      - leave all of the convolutional layers frozen and just train the new classification head 
      - you'll be using the same feature extraction
      - start with that 
    - and then try going up the network and unfreeze the last block if that's not working 
    - important things: what data you have 
      - two scenarios: worked for all sites you trained on, introduce new site, add data in from both if you want to train on original data and pearce point data 
      - second: fine-tuning for that specific site, use data only from new site, Fabio thinks this is the best bet for getting the detector working in each place

## Dec 11 - 15 ##

Q's for Fabio for next week:
- Fine tuning the existing detector walkthrough? Read your paper
- Why do we refresh the batch generator on epoch end for training and not validation? 
  - Resamples at the end of each epoch so different batches in every epoch 
- Should I try with early stopping? 
- Time shifting samples, back to the ketos selections question: I think I should do this? and I could do the same for negatives if I set the duration to 1sec with a shift of up to 0.5s?
- Normalizing data: I see the batch norm, but I'm currently not normalizing the input data right? Or does the MagSpectrogram do this?
- Should I try resnet-18?

### Dec 13

Today To Do's:
- Write up model architecture summary and create graphic (DONE)
- Email Sebastien about ensemble learning and clarify you're doing it right (DONE)
- Get optimal spectrogram parameters from Raven for each site, and for test site PP and Ulu2023
- Read Mariana's paper
- Call Mummo

Long term to do's: 
- cli 
- add noise and blurring to spectrograms 
- try even shorter durations
- add dropout layer after the first layer 
- try with dropout in general in the architecture 
- add site name back into annot tables 
- edit output so its by file instead of by spectrogram

### Dec 12
- Set up ensemble learning, ran on 1sec and pp
- Brain too tired to write todo's 
- Try to write up summary of what trying to do for Fabio for meeting

## Dec 4 - Dec 8 2023 ##

### Dec 8 ###

Today: 
- Write thank you note to donor awards people 
- Saliency maps
  - https://colab.research.google.com/drive/19eB3uwAtCKZgkoWtMzrF0LTJ-htF_KE7#scrollTo=_9FKJVOAC8a7
  - .kt file is not directly relatable to a keras Model object
  - Has a ResNetInterface object, and resnet.model returns a ketos ResNetArch object 
  - This is a subclassed model
  - Couldn't figure out how to get keras to understand this model format 
  - Ketos can output ".kt" or ".pb" format, but the .pb does not output the correct metadata to load as a Keras model 
  - resnet.model is not a h5 format, so can't use keras load model 
  - https://www.tensorflow.org/tutorials/keras/save_and_load#save_the_entire_model
  - https://www.tensorflow.org/guide/keras/making_new_layers_and_models_via_subclassing
- Test 1 sec duration spectrograms
- Code to close tables:
  - import tables
  - tables.file._open_files.close_all()

Long Term: 
- Understand how mag spectrogram is generated 
- Set up command line interface
- Edit scripts to use the method Fabio was talking about with joint batch generators, etc
- PyTables tutorial 
- Saliency maps using xplique
- Deep ensemble learning 
- Multiple spectrogram representations in training data 
- Noise: add blurring, gaussian noise, cropping, dropout after first layer, mix-up
- Try dropout throughout the network 
- Try pretrained efficient net B0, fine-tuning the last layer 
- Try to find a big model that's on speech data, that's not overly big (whisper)
- Semi-supervised, helpful if there's unlabeled data 
- Train only on unlabelled data (simCLR), contrastive model (loss is difference between two inputs if you add noise and blurring in two siamese networks), read about this 
- Try adding site number as an array in the last layers (FC -> SITE # -> FC -> AS USUAL)

### Dec 7 ###

Meeting w Bill:
- Denoising 
- How are the spectrograms being compared if the background is so different? 

Meeting w Sebastien: 
- Discuss current model & results 
- Other models? Other suggestions? 
- Denoising - no
- How are the spectrograms being compared if the background is so different? 
- How do the skip connections do anything for ResNet? 
- How to determine array size before and after convolution? 

- multiple spectrogram representations for each selection 
- add gaussian noise to the spectrogram which would help w overfitting for the multiple sites 
- uncertainty probability calibration:
  - What you get at the end of the classifier is a "score" between 0 and 1, its not warranted that its the probability.
  - calibrate data: conformal prediction: mapie package 
- deep ensembles: train 5 times, make the seed is different everytime, predict using each model for each segmented spectrogram, make the average of each one, and standard deviation

AUGMENTATION:
- Mix up: keras probably has a package. Adding samples together with a multiplication factor, read about it, seems like a simple thing to try 
- Noise
- Blurring
- Multiple spectrogram representations 
- Cropping 
- Add dropout after the first layer 

OTHER MODEL TYPES:
- Try dropout throughout the network 
- Try pretrained efficient net B0, fine-tuning the last layer 
- Try to find a big model that's on speech data, that's not overly big (whisper)

SSL: 
- Semi-supervised, helpful if there's unlabeled data 
- Train only on unlabelled data (simCLR), contrastive model, read about this 

METADATA:
- Might as well try 
- add in a site number
- xplique - keras package, saliency map 

### Dec 6 ###
- Code map (DONE)
- Write up how the model works with the kernels and stuff (DONE)
  - See if I can understand the sizes of the input data in the future
- Try 1 second spectros w augmentation
- Update readme file 
- Set up command line interface 
- Edit scripts to use the method Fabio was talking about with joint batch generators, etc
- PyTables tutorial
- Look at how mag spectrogram is calculated

### Dec 4,5 ###
Overall goals:
- Set up command line interface
- Fix confusion matrix plot function (DONE)
- Edit scripts to use the method Fabio was talking about with joint batch generators, etc 
- Test shorter spectrogram parameters 
- PyTables tutorial 

Questions to answer: 
- Array shape?
- When does the data actually get converted into a spectrogram array? How is the size determined? 
- What does the model architecture look like exactly? What does the kernel stride etc do exactly? 

HDF5 and pytables:
- Hierarchical data format 
- Project data is often complex, these files allow for storage of large amounts of data 
- Allows to group in folder structure 
- Can attach metadata

Summary stuff:
- Using mag spectrogram, hamming window, min freq 100 max freq 1200

Walk through of create database code: 
- Read in the excel sheets with the annotations 
- Load in the spectrogram config as a dict with: 
  - rate, window, step, freq_min, freq_max, window_func, type, duration
  - Type is "mag spectrogram" which is a ketos audio spectrogram class 
  - In the "load_audio_representation" ketos function: 
    - go to "parse_audio_representation"
    - returns a dict with all da info 
  - create a database using the "create database" function from ketos database interface class 
    - start  line 903 in database_interface

Walk through of train classifier code: 
- open file using pytables in read mode
  - Lots of steps here, but just opening the file and checking the file path using pytables 
- Throw extra data into the last batch file if not evenly divided
- Using FScore loss 
- batch normalization using tf.keras.layers.BatchNormalization
- Using dropout rate of 0
- In resnet.py, looks like we do: 
  - Conv initial
  - blocks [2,2,2] ("sequential")
  - batch normalization 
  - relu
  - average pool 
  - fully connected 
  - softmax

- Generate batches using the batch generators
  - These shuffle the data into batches of the defined size and put the rest into the last batch 
  - You can refresh the data in the batch at the end of epoch or not 
- Using the "ResNetInterface" class from Ketos 
  - the generators, log dir, and checkpoint dir are set before calling the train_loop method of the ResNetInterface class 
  - Currently not doing anything fancy, no early stopping 
- In the train loop: 
  - Outputting a checkpoint every 5th epoch, saves model weights 
  - Currently outputting a csv for the log, not outputting the tensorflow summary 
  - not using early stopping 
  - reset the training and validation loss 
  - for the first batch (which in this example is 16 spectrograms), 
    - get the training data (both x and y, ie. data and label)
    - train_X is a {16, 1500, 56, 1} array 
    - train_Y is a 16,2 array (labels like [1. 0.], [0. 1.])
    - Go to the next train step using that train_X and train_Y
    - where it won't let me step into the _train_step function but....
      - predict the output labels using the model 
      - calc the loss using the loss function on the labels and predictions 
      - apply the gradient to the optimizer, where the gradient is: 
        - tape.gradient(loss, self.model.trainable_variables)
      - get the mean loss for the step? (line 1213 of nn_interface.py)
      - the loss function is the F1Score loss from ketos dev_utils losses 
    - Once it's looped through all batches, continue 
    - then run on val data using the _val_step function:
      - same thing as training except no gradient calc and no applying them
    - Print out the verbose stuff 
    - lots of stuff for early stopping 
    - save to the log


## Nov 27 - Dec 1 2023 ##

- Finished creating manual dataset 
  - Checked every negative segment and classified which type of noise it was 
  - Replaced segments containing RS yelps or missed barks 
- Because I used 2 sec, I can't change this now - I can go smaller but the classifications won't necessarily hold 
- Need to now edit selection tables to not have start and end times outside of the file 
- Test different spectrogram parameters 
- Try writing out resNet on my own and really understanding what's happening 
- Plotting saliency maps or something? 
- Update script to use the method Fabio was talking about instead (using joint batch generator)
- Next steps: 
  - Look at false detections in depth and write up a summary 
  - Test other spectrogram parameters (specifically let's try the frequency and rate)
  - Maybe add in multiclass?
  - Write out resnet on my own 
  - Add the jupyter notebook to the read the docs 
  - Do command line interface set up 
  - Fix confusion matrix plot (and figure out which one is right! I'm guessing Ruwans) - yes Ruwans is right

Q's for Fabio: 

1. For the selection tables, I'm loading them in already at the specified duration. Will this cause problems? 
   - It's ok not to use it 

2. Why am I still getting padding comments when I've manually verified all of these selections from plotting? No where near that many had padding visible when I manually inspected them. Is it because of the "rate" in the cfg? Before I created a function "drop out of bounds selections" and dropped selections with times after or before a wav file, but I don't want to do this bc of the manual-ness of the dataset. Also why would this be happening?
   - Did indeed have positive selections both before and after the true end of the file 
   - Won't matter too much to drop those examples 
   - Do this step before creating the database 

3. Is there a better way to load the tables into the database than I'm doing it now? I remember you were talking about train/ulu/pos, train/ulu/neg kinda structure, but I wasn't sure how to do it
    - There will be greater flexibility in the future if you adjust to this way, so try it out 

4. Next steps? Probably testing spectrogram settings, maybe if I reintroduce sl.select I can do different durations (less than 2s) easily? And add in augmentation?
   - Test spectrogram settings 
   - Duration: can go down to 1.5 or 1 but the classifications on noise segments won't hold anymore potentially

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
