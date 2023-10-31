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
- Results of detection and classification tasks
- Summary & conclusion

Notes:
- Classical methods plateau at 50% recall when false detections are kept below 10%
- Last decade, ML are now the way to go
- CNNs have been used to analyze info in spectrograms (with examples)
- Locations vary in flow noise, background, strum, knocks, tidal currents, surface motion
- SNR variability due to different locations 
- Statement about not enhancing SNR before feeding to neural network
- Quasi-random time shifts are desirable for DNN classifier bc they encourage network to learn a more general, time translation invariant, representation of the upcall 

![alt text](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true)