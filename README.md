# WiFi-ADG
The source codes of the experiments include three files: WIFI_ADG_ae.py, WIFI_ADG_model.py, and WIFI_ADG_run.py.

TABLE I indicates the classification results of three classifiers with 8 cross-validation experiments. TABLE I includes 8 child folders: “classifier_0”-“classifier_7”, which correspond to 8 cross-validation experiments. Each of these 8 child folders consists of 3 well-trained classifier network models: C_room, C_act, and C_id, respectively. For child folder “classifier_0”, all three models have been trained with a training set including data1.mat-data7.mat and resdata.mat, and they are tested with the testing set data0.mat. For child folder “classifier_1”, all three models have been trained with a training set including data0.mat, data2.mat-data7.mat, and resdata.mat, and they are tested with the testing set data1.mat, and so on. 

TABLE II indicates the 8-fold cross-validation results of ADG network with 6-10 layers, respectively. TABLE II includes 5 child folders, each of which corresponds to an ADG network with a specified layer. Each of these 5 child folders consists of 8 child folders, corresponding to 8 cross-validation experiments. For example, the child folder “6-layer ADG” consists of 8 child folder: “classifier_0”-“classifier_7”. For child folder “classifier_0”, the network model C_room has been trained with a training set including data1.mat-data7.mat and resdata.mat, and it is tested with the testing set data0.mat.

TABLE III indicates the 8-fold cross-validation results of ADG network with different beta values. TABLE III includes 8 child folders, which correspond to 8 cross-validation experiments. Each child folder further includes 3 child folders: “beta0_1”, “beta0_01”, and “beta0_001”, which correspond to our model with beta values setting to be 0.1, 0.01, 0.001, respectively. 

TABLE IV-1 indicates the 8-fold cross-validation results of our scheme with untargeted preservation. Accordingly, TABLE IV-1 consists of 8 child folders. Each child folder further includes three folders: “A-act”, “A-id”, and “A-room”, representing the models with adversarial activity, adversarial identification, and adversarial location, respectively. 

TABLE IV-2, which corresponds to TABLE IV-1, indicates the results of the competing scheme.

TABLE V indicates the results of the targeted adversarial preservation. TABLE V includes 3 child folders: “T-act”, “T-id”, and “T-room”. “T-act” includes 7 child folders: “act0”-“act6”, corresponding to 7 targeted activities. “T-id” includes 10 child folder: “id0”-“id6”, corresponding to 10 targeted volunteers. “T-room” includes 6 child folders: “room0”-“room5”, corresponding to 6 targeted rooms. 

In conclusion, this paper trains 151 network models with 8-fold cross validation. Those well-trained models are available at https://github.com/siwangzhou/WiFi-ADG. Nine datasets, including data0.mat, data1.mat, …, data7.mat and resdata.mat, are with the sizes of more than 400 MB, where resdata.mat has 1080 samples and each of the other 8 datasets has 4200 samples, respectively. Those datasets are available at  https://pan.baidu.com/s/1fOhaYD1vq39JWXPExtsCIg. 

Any questions, please contact us at swzhou@hnu.edu.cn.
