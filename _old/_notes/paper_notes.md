# Notes from Papers #

## Performance of a deep neural network at detecting North Atlantic right whale upcalls ##
**Kirsebom et al. 2020**

Structure of paper: 
- Intro:
  - Introduce species 
  - Explain area 
  - Explain motivation for conservation efforts (vessel speed, etc)
  - Explain why PAM is needed 
  - Explain classification algorithms and older ones 
  - Explain performance of these classification algorithms 
  - Introduce ML
  - Introduce DL and example applications
  - CNNs with examples and spectrogram ones
- Acoustic Data Collection
  - Dates, locations, map
  - Equipment type and duty cycle 
  - Differences in data recorded, SNR variability
- Generation of training datasets, neural network design, training protocol
  - First looked at detector for NARW previously in existence 
  - Talked about annotations and where they got them from
  - Extracted segment length, centered on call
  - Train test split, used a t val to split to an 85:15 split, cool diagram
- Spectrogram and SNR Computation
  - Downsample rate
  - scale (dB)
  - window size, step size, window function
  - No effort to correct for systematic difference in signal amp
  - Estimate SNR of each sample (with steps)
- NN Architecture: 
  - binary classification (define the classes)
  - duration of spectrograms 
  - brief mention of the CNN architecture based on He et al 2016
  - how passed to fully connected network and gives probabilities 
- Training Protocol:
  - GPU with memory stats 
  - Batch sizes with termination after a preset number of epochs 
  - optimizer, with recommended vals (lr 0.001, decay 0.01, b1 of 0.9, b2 of 0.999)
  - Didn't tune hyperparameters 
  - network trained to maximize the F1 score
  - trained using 5-fold cross validation, N=100
  - Then trained on full data without cross validation 
  - Repeated 9 times with different random gen seeds to assess the sensititivty of the training outcome to initial conditions
- Compared to linear discriminant analysis
- Detection algorithm:
  - detect in continuos data we:
    - segment to window size of 3s with step size of 0.5s
    - each segment passed into DNN classifier 
    - found best to smoothen the classification scores using a 2.5s wide averaging window 
    - greatly reduced the number of false positives
    - applied detection threshold
  - to compute recall, precision, false-positive:
    - merged adjacent bins into detection events 
    - had a temporal buffer
    - false positive when annotated overlap was less than 50%
- Results of detection and classification tasks
  - 9 independent training sessions 
  - looked at increasing size of dataset 
  - looked at discarding low SNR samples
  - plots of precision and recall vs LDA
- Summary & conclusion
  - Discussion:
    - DNN have larger capacity to handle variance in the data than LDA
    - General review on what was already talked about
    - Results summarized
    - A brief discussion on papers that were released during the writing of their paper
  - Conclusion: 
    - Shown that DNNs work for the task
    - improved performance by expanding variance of dataset
    - "consider a wider temporal context"

Notes:
- Classical methods plateau at 50% recall when false detections are kept below 10%
- Last decade, ML are now the way to go
- CNNs have been used to analyze info in spectrograms (with examples)
- Locations vary in flow noise, background, strum, knocks, tidal currents, surface motion
- SNR variability due to different locations 
- Statement about not enhancing SNR before feeding to neural network
- Quasi-random time shifts are desirable for DNN classifier bc they encourage network to learn a more general, time translation invariant, representation of the upcall 
- F1 score defined as the "harmonic mean of precision and recall", attaches equal importance to the two

![plot](kirsebom.png)
