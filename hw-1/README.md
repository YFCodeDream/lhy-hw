# LiHongYee HW-1 Self Implementation

1. dataset has already been in 'dataset' directory, don't need to download them
2. this model use linear layer block (very simple :) ), including linear layer, relu activation function, and batch norm
3. training strategy is various. learning rate decay is used in this training procedure
4. if you want to use mic_select_features, please modify the 'loader_kwargs' parameter
5. this program also support manual feature selection, just infer the indices of remained features and set 'select_features' to 'manual'