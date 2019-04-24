------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------Sentiment analysis of images using transfer learning--------------------------------------
----------------------------------------------Abhishek Vasudevan--------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------


The three python scripts used for training and testing are available on Blackboard. The remaining data files are available at: https://drive.google.com/drive/folders/1aZupURbXcwoDOf6vxFkXliD09QtCTMeG?usp=sharing

After downloading all files, unzip the "Agg_AMT_Candidates.zip" and "data.zip" files. After this, your directory structure should look like this:

root
|
|__ train_final.py
|__ train_final2.py
|__ test_data.py
|__ three_agree.txt
|__ four_agree.txt
|__ five_agree.txt
|__ Agg_AMT_Candidates
          |__(all test images)
|__ data
      |__(all train images)
|__ weights_best_densenet.hdf5
|__ weights_best_vgg.hdf5
|__ t4sa_text_sentiment.tsv

File details:

1) train_final.py: Code for training VGG19 model
2) train_final2.py: Code for training DenseNet121 model
3) test_data.py: Code used for testing the models, outputs evaluation metrics
4) three_agree.txt, four_agree.txt, five_agree.txt: Files needed for evaluating the models.
5) Agg_AMT_Candidates: This folder contains the testing images from the Twitter testing dataset
6) data: These contain the images (50k images from original B-T4SA dataset) for training models released by the authors of the base paper
7) weights_best_densenet.hdf5: Contains the weights resulting from training the DenseNet architecture defined in train_final2.py
8) weights_best_vgg.hdf5: Contains the weights resulting from training the VGG architecture defined in train_final.py
9) t4sa_text_sentiment.tsv: Contains sentiment polarity confidence values for the images in the data folder

Requirements for running the python scripts:
python 3.6.5
tensorflow 1.13.1


How to run the scripts:

To train the VGG19, run the python file train_final.py:

                        python3 train_final.py


To train the DenseNet121 model, run the python file: train_final2.py

                        python3 train_final2.py

To test the models, run the python file: test_data.py

                        python3 test_data.py

  You need to input 1 for testing DenseNet121 and input 2 for testing VGG19
  You can choose to view images one by one by using interactive mode (input 1). Or to just evaluate all images without displaying them, input 0
