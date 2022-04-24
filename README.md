**Baby Cry Detector** 

A deep learning based embedded AI project providing 7/24 service of real-time monitoring of your baby 

Dongyi Ma (Mandy)

https://youtu.be/MwAbkkQyHkw (YouTube)

https://studio.edgeimpulse.com/public/92154/latest (Edge-impulse project page)

https://github.com/mandymadongyi/Baby-Cry-Detector-CASA0018 (GitHub Repo)

## Introduction

This is a neural network based deep learning project that intends to provide a 7/24 real-time monitoring service for parents to detect baby cries from other sounds at home with Arduino nano and notify them using onboard built-in LED indicators.  Generally, infants communicate with the world through crying. They cry when they feel hungry, uncomfortable, or painful. No caregiver can pay attention to each of their babies for 24 hours a day, 7 days a week. Thus, a remote real-time monitoring baby cry detector helps employed parents from missing any signal from their babies to better take care of them.

The baby cry detectors available on the market mainly focus on the output methods like building apps or smart parenting devices (lullaby players/night light) for commercial use.  IOS apps such as ChatterBaby and Shush, as well as Android apps like Baby Monitor and BabyCam could provide calls/messages/push notifications to alert young parents of baby crying (Baby Gear Essentials, 2022).  Instead of creating a commercialized product, this project focuses on enhancing the machine learning model performance. It advances the work of Ish Ot Jr (2022), who provides the basic procedure for building a baby monitor powered by tinyML and Edge Impulse on Arduino Nano. The main purpose of this project is to collect a comprehensive dataset and subsequently create a machine learning model with fine-tuned parameters for higher accuracy attainment compared with the model from Ish Ot Jr. 

## Research Question

Can a neural network-based embedded AI project be built to detect baby cries from all sounds at home and alert their parents?

 

## Application Overview

This project contains three parts: 1) Build an audio dataset by manually recording baby cry sounds and noise sounds at home. 2) Create a deep learning model by conducting 30 experiments with different DSP blocks, learning blocks, parameter settings as well as model architecture and find out the best performance model to detect whether a baby is crying. 3) Deploy the best-trained model onto an Arduino nano board to keep sensing the sounds at home and use its built-in LED lights (red, green, blue) on the board to real-time indicate whether the baby is crying or not. 

For the application process, audios are captured by Arduino built-in microphone each second and converted into spectrograms, then TF Lite will run the model to classify sensed sound into 2 types, crying and noise. If the crying possibility is over 0.7, the red light turns on. If the noise possibility is over 0.7, the green light turns on. If none of these happens, blue light is turned on. Red means crying, green means not crying, and blue means not sure. In short, when the baby’s cry is detected, a red light will be turned on to alert the parents. (Warden & Situnayake, 2020)

<img width="916" alt="1" src="https://user-images.githubusercontent.com/91919718/164993822-4ecbbe06-1dd4-4eb8-ba88-66b03a0f5c2f.png">

Figure 1. Application diagram of the building blocks of the baby cry detector

## Data

#### Data source:

I manually recorded voices samples from copyrighted online resources, including Sound Jay, Kaggle, Epidemic Sound and YouTube with the keywords “baby cry”, “infants cry” “noise at home” and “sounds at home” search. I also asked my friends to record their sons/daughters crying to enrich the dataset. 

#### Datasets:

The total number of collected data is 846 pieces, with a total length of 14minutes and 6 seconds. Among them, two labels are used for classification. 

·   Crying - contains audio of more than 50 babies crying of different gender, ages, religions, ethnicity, and race, including hunger cry, colic, and sleep cry. (MedicineNet.com, 2022) The total amount of data with this label is 289. 

·   Noise – contains different possible sounds at home, including people talking, singing, coughing, sneezing, footsteps, washing machine whirring, music, dogs barking, cats-meow, TV sound, typing sound, sounds of thunder and of rain, adult female crying and adult male crying. The total amount of data with this label is 557.

They are then divided into 660 training data (78%) and 186 testing data (22%).  

|        | Training | Testing  | Total |
| ------ | -------- | -------- | ----- |
| Crying | 235      | 54       | 289   |
| Noise  | 425      | 132      | 557   |
| Total  | 660(78%) | 186(22%) | 846   |

Table 1. Datasets description

#### Data pre-processing: 

Transforming and standardising data into a common format helps to improve the quality of data and create system consistency. Data is pre-cleaned through manually choosing dedicated and clear high audio quality samples to record from a vast amount of baby crying sounds online., After recording each 16-second audio, they were converted into 1-second audios for standardising sample size purpose, removing the sounds that do not clearly feature a baby crying to make the data cleaner. 

<img width="231" alt="image" src="https://user-images.githubusercontent.com/91919718/164993883-5e1c64a6-0909-4c21-845c-e4260983e8de.png">


Figure 2. Splitting one piece of 16 seconds audio into 5 pieces of 1-second audio, removing the “silence”

 

​    ![Chart  Description automatically generated](file:////Users/dongyima/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image003.png)![Chart  Description automatically generated](file:////Users/dongyima/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image004.png)![Chart  Description automatically generated](file:////Users/dongyima/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image005.png)

Figure 3 - 5. Three examples of final crying data

​    ![Graphical user interface  Description automatically generated](file:////Users/dongyima/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image006.png)![Chart  Description automatically generated](file:////Users/dongyima/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image007.png)![Chart  Description automatically generated](file:////Users/dongyima/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image008.png)

Figure 6 - 8. Examples of final noise data (people talking, typing, music)  

![Table  Description automatically generated](file:////Users/dongyima/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image009.png)![Table  Description automatically generated](file:////Users/dongyima/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image010.png)

​        

Figure 9 & Figure 10. Datasets before and after length standardisation

Based on the spectrogram of the sample data, I assume crying data might have a higher frequency than other noise sounds at home. Further research is done by applying a machine learning model to reveal the other hidden features in crying data.

 

## Model

After collecting, data is then sent to processing blocks and learning blocks for feature generation and machine learning. Then the model can be trained with different parameter settings before testing and deployment. 

![Diagram  Description automatically generated](file:////Users/dongyima/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image011.png)

Figure 11. Workflow of creating, training, testing and deploying the model

For audio processing deep learning projects, Edge Impulse recommends 3 types of DSP blocks (MFCC, MFE, Spectrum) and 2 types of learning blocks (Classification, Transfer Learning). (Edge Impulse Documentation, 2022)

Keeping the other variables (dropout rate, data augmentation on/off, number of epochs, validation set size, number of layers and its neurons for a certain learning mode) unchanged, 6 experiments are conducted using all possible combinations of DSP blocks and learning modes. It is worth noting that transfer learning only works with MFE. Based on the accuracy performance and training loss, the combination of a DSP of Spectrogram and Classification learning mode gives the highest model accuracy of 90.83% and lowest training loss of 0.21 and is then chosen to be the final model architecture for this project.

 

![img](file:////Users/dongyima/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image012.png)

Figure 12. 6 experiments conducted with different processing and learning blocks to define the final model architecture

![Graphical user interface, application, Teams  Description automatically generated](file:////Users/dongyima/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image013.png)

Figure 13. Final model architecture in Edge Impulse

## Experiments

30 experiments are conducted before finding the best-trained model. 

![Table  Description automatically generated](file:////Users/dongyima/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image014.png)

Figure 14. Experiments records

During experimentation, seven parameters are investigated to improve model performance, including input layer features, learning rate, neural network architecture, dropout rate, number of epochs and data augmentation. Experiment 1-6 defines model architecture, experiment 7-13 found the optimal learning rate, experiment 13-19 helped to define neural network architecture. Experiment 20-24 looked at the optimal dropout rate, experiment 25 investigated data augmentation should be toggled on or off, and experiment 26-30 searched for the best number of epochs. 

![Graphical user interface, text, application, email  Description automatically generated](file:////Users/dongyima/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image015.png)

Table 2. Parameters changes made during experiments

After all, the optimal model is chosen, which uses a reshape layer with 65 columns, 1 Conv and dropout layer, 1 dense and drop out layer and one flatten layer to classify input data into 2 categories. 

Input features number: 6435

Learning rate: 0.008

Neurons: 8, 256

Number of Layers: 6

Dropout rate: 0.45

Data augmentation: Off

Training epochs: 30

Accuracy on validation set: 93.20%

Accuracy on testing set: 93.01%

## Results and Observations

####  Results:

•    Model accuracy increased from 90.83% with the recommended setting provided by Edge Impulse to 93.01% with the designed model and fine-tuned parameters. 

•    The confusion matrix shows that incorrect prediction of crying is 9.8% of total crying data.

 

![Chart  Description automatically generated](file:////Users/dongyima/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image016.png)![Chart, scatter chart  Description automatically generated](file:////Users/dongyima/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image017.png)![Chart  Description automatically generated](file:////Users/dongyima/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image018.png)

Figure 15. Default model performance on testing data

Figure 16. Trained model performance on testing data

Figure 17. Trained model performance on validation data

 

#### Observations:

**![A screenshot of a computer  Description automatically generated with medium confidence](file:////Users/dongyima/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image019.png)**

Figure 18. Parameter changes and their corresponding impact on accuracy, listed by experiment order on the left, by percentage changes from high to low on the right

•    I started with an accuracy of 90.83% when learning rate is 0.01, dropout is 0.1 in 2 Conv layers NN running 30 epochs. After that, the learning rate is changed to 0,008 to lower down loss from 0.21 to 0.10. Then change the second 1d Conv layer to a dense layer to increase accuracy to 92.08%, and change reshape columns from 5 to 65 to further increase accuracy to 92.50%, Finally, adjust dropout to 0.45 to improve accuracy to 93.01%. However, turning on data augmentation decreases accuracy by 21.66% and running more epochs is not helpful to improve accuracy. Dropout has the biggest impact on model accuracy but is useful to reduce the risk of overfitting.

•    Fewer layers and neurons lead to underfitting, where the model is not trained enough to predict accurately, but more layers, neurons and parameters can’t ensure better performance since they may cause overfitting where the model matches the training data set too well but fails to fit in with unknown dataset. 

•    Slowing down the learning rate has little impact on model accuracy but helps to prevent quick overfitting.

•    Reshape layer is important, especially for time series data. Reshape column’s value defines the input of Conv layer and should be fixed by the number of coefficients of the DSP block, otherwise, model accuracy suffers. (Flatten, Reshape, and Squeeze Explained - Tensors for Deep Learning with PyTorch, 2022)

•    The dense layer can incorporate with Conv layer and works well for processed data. 

•    Running more or fewer epochs cannot ensure better or worse performance. 

•    Choosing the right DSP and learning blocks are the most important things. Neural network architecture and dropout also influence how the model behaves. 

#### Reflection: 

Overall, this project advances Ish Ot Jr work in two aspects: 

\1) The total amount and variety of types of data, along with cleaning and standardising datasets. 

\2) The created model and fine-tune the model parameter for higher accuracy attainment. 

Future work may include: 

a) Improving the dataset and model (feature settings, 2d Conv layer, optimizer, and activation function etc.),

b) Classify different types of baby cry i.e., hunger cry, colic, sleep. 

c) Add a command recognizer between TFLite and command responder to improve accuracy by aggregating the results before determining whether a keyword is heard. 

d) Develop more output methods (alert with a Bluetooth speaker, build a monitoring app and auto-comfort the crying baby) 

e) Planning for enclosure and power budget for real-world deployment.

(Word Count: 1634)

## Bibliography

Data source:

•    [https://www.soundjay.com/baby-crying-sound-](https://www.soundjay.com/baby-crying-sound-effect.htm)[effect.htm](https://www.soundjay.com/baby-crying-sound-effect.htm)[l](https://www.soundjay.com/baby-crying-sound-effect.html)



•    https://www.kaggle.com/datatangai/infant-cry-speech-data-by-mobile-phone

•    https://www.epidemicsound.com/track/IfihVESIQr/?_us=adwords&_usx=11302689053_&utm_source=google&utm_medium=paidsearch&utm_campaign=11302689053&utm_term=&gclid=CjwKCAjw8sCRBhA6EiwA6_IF4Vjs1-ycNhy7AR9Q65F9lc5xS-9zkv6EE2qmjRzSLJnKg0llbYwr1BoCCuEQAvD_BwE

•    https://www.youtube.com/watch?v=0aMFEmPO8G0&t=23s

Reference:

\1.   Baby Gear Essentials. 2022. The 10 Best Baby Monitor Apps of 2021 (Free and Paid). [online] Available at: <https://babygearessentials.com/baby-monitor-apps/> [Accessed 8 April 2022].

\2.   Deeplizard.com. 2022. Flatten, Reshape, and Squeeze Explained - Tensors for Deep Learning with PyTorch. [online] Available at: <https://deeplizard.com/learn/video/fCVuiW9AFzY> [Accessed 9 April 2022].

\3.   Docs.edgeimpulse.com. 2022. Processing blocks - Edge Impulse Documentation. [online] Available at: <https://docs.edgeimpulse.com/docs/tutorials/processing-blocks> [Accessed 9 April 2022].

\4.   Ish Ot Jr. 2022. B: A Baby Monitor Powered by tinyML and Edge Impulse!. Arduino Project Hub. [online] Available at: <https://create.arduino.cc/projecthub/ishotjr/babl-a-baby-monitor-powered-by-tinyml-and-edge-impulse-f5045f> [Accessed 5 April 2022].

\5.   Medicinenet.com. 2022. [online] Available at: <https://www.medicinenet.com/what_are_the_3_types_of_baby_cries/article.htm> [Accessed 9 April 2022].

\6.   Warden P, Situnayake D. (2020). TinyML. Second Edition. Sebastopol: O’Reilly, Page 132-133. 



 



## Declaration of Authorship

I, Dongyi Ma, confirm that the work presented in this assessment is my own. Where information has been derived from other sources, I confirm that this has been indicated in the work.

*Dongyi Ma*

 

2022.04.11

 

