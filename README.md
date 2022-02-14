# PGCN: Progressive Graph Convolutional Network for Spatial-Temproal Traffic Forecasting

![image](https://user-images.githubusercontent.com/31876093/153813727-71ffc4aa-8ced-401e-bfa0-a72fc088b319.png)

This is a PyTorch implementation of Progressive Graph Convolutional Network in the paper entitled "PGCN: Progressive Graph Convolutional Network for Spatial-Temproal Traffic Forecasting"
The paper is currently under review for KDD '22.

## Progressive Graph Construction
![image](https://user-images.githubusercontent.com/31876093/153813757-0f18e904-c3a2-4f73-ac23-4b2c5ba6420d.png)

Using adjusted cosine similarity values, the model constructs progressive graph to reflect the changes in traffic states at each time step. 

## Performance Comparison 
#### Datasets
- PeMS-Bay: Highway traffic speed data from 325 sensors in Bay Area [1]
- METR-LA: Highway traffic flow data from 207 sensors in LA [1]
- Urban-core: Urban traffic speed data from 304 sensors in Seoul, South Korea [2]
- Seattle: Highway traffic speed data from 323 sensors in Greater Seattle Area [3]

#### Results
![image](https://user-images.githubusercontent.com/31876093/153811602-29dd99a7-5cc9-48a6-9962-ee6ecb7714a8.png)

Evaluation results on four real-world datasets shows that our model consistently outputs state-of-the-art results.

##
##### code implementation
Code for PGCN has been implemented by modifying codes from Graph WaveNet (https://github.com/nnzhan/Graph-WaveNet)[4] 

##### References
[1] Li, Y., Yu, R., Shahabi, C., & Liu, Y. (2017). Diffusion convolutional recurrent neural network: Data-driven traffic forecasting. arXiv preprint arXiv:1707.01926.

[2] Shin, Y., & Yoon, Y. (2021). A Comparative Study on Basic Elements of Deep Learning Models for Spatial-Temporal Traffic Forecasting. arXiv preprint arXiv:2111.07513.

[3] Cui, Z., Henrickson, K., Ke, R., & Wang, Y. (2019). Traffic graph convolutional recurrent neural network: A deep learning framework for network-scale traffic learning and forecasting. IEEE Transactions on Intelligent Transportation Systems, 21(11), 4883-4894.

[4] Wu, Z., Pan, S., Long, G., Jiang, J., & Zhang, C. (2019). Graph wavenet for deep spatial-temporal graph modeling. arXiv preprint arXiv:1906.00121.

