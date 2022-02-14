# PGCN: Progressive Graph Convolutional Network for Spatial-Temproal Traffic Forecasting

![image](https://user-images.githubusercontent.com/31876093/153811234-3e04f806-63e0-4ab5-9ee2-eedd67b737bb.png)

This is a PyTorch implementation of Progressive Graph Convolutional Network in the paper entitled "PGCN: Progressive Graph Convolutional Network for Spatial-Temproal Traffic Forecasting"
The paper is currently under review for KDD '22.

## Progressive Graph Construction
![image]("https://user-images.githubusercontent.com/31876093/153811523-e3977ecb-a39d-4add-98a3-366139ecfda6.png")
Using adjusted cosine similarity values, the model constructs progressive graph to reflect the changes in traffic states at each time step. 

## Performance Comparison 
![image](https://user-images.githubusercontent.com/31876093/153811602-29dd99a7-5cc9-48a6-9962-ee6ecb7714a8.png)
Evaluation results on four real-world datasets shows that our model consistently outputs state-of-the-art results.
