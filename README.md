# facemask-condition-detection

In this project, a facemask detection model using is presented. This system consists of two stages. The first stage is the SSD MobileNetV2, and for the second stage, a custom CNN network is used. The seconde stage has been added to improve the accuracy of the classification. The model is also implemented in a real-time gate controlling system using a servo motor and Raspberry Pi 3.

## Training
we used FMDD dataset for training 

for training, use the first and second stage trainers codes available in face-mask-detection folder

## Results
samples of 1st stage detections

![image](https://user-images.githubusercontent.com/105539041/171268951-efa96495-a4a8-45ae-ae7d-08996ecc5fed.png)
![image](https://user-images.githubusercontent.com/105539041/171268961-8b470df3-3655-4757-85e9-9c01889587a2.png)
![image](https://user-images.githubusercontent.com/105539041/171268972-c3c7031c-eb22-42a6-bc4e-621fd51f1f94.png)
![image](https://user-images.githubusercontent.com/105539041/171268996-1d8e2863-868a-4083-9fae-e67685c4abd4.png)


samples of real-time facemask detection with gate controlling results

![facemask_detection2](https://user-images.githubusercontent.com/105539041/171257781-5885bd2d-0c79-426c-8923-773490902377.png)
![facemask_detection](https://user-images.githubusercontent.com/105539041/171257734-96013d76-acd0-45a1-9aac-30389b36140b.png)
![facemask_detection3](https://user-images.githubusercontent.com/105539041/171283983-4e6d6277-74dc-4b8e-b4a6-ae99d6b9b4ab.png)
