% 草稿

% 定义yolox检测器的参数
jkclasses = {'A','B','C'}; % 定义类别
networkNamefdgg = "tiny-coco"; %变量名不重要，由定义的先后顺序决定

detector = yoloxObjectDetector(networkNamefdgg,jkclasses,InputSize=[256 256 3]);
%%detector.InputSize = [256 256 3]; 这个方式是错误的
disp(detector);


%%定义一个双输出神经网络，该神经网络在给定二维图像作为输入的情况下预测分类标签和数值。

%指定类和响应的数量。
numClasses = 10;
numResponses = 1;

%创建一个空的神经网络
net = dlnetwork;

%定义网络主分支的各层和 softmax 输出
%layers参数-网络层-数组(在深度学习层列表中进行查询，自定义层也在其中)
layers = [
    imageInputLayer([28 28 1],Normalization="none")

    convolution2dLayer(5,16,Padding="same")
    batchNormalizationLayer
    reluLayer(Name="relu_1")

    convolution2dLayer(3,32,Padding="same",Stride=2)
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,32,Padding="same")
    batchNormalizationLayer
    reluLayer

    additionLayer(2, Name="add")

    fullyConnectedLayer(numClasses)
    softmaxLayer(Name="softmax")

];

net = addLayers(net,layers);

layers = [
    convolution2dLayer(1,32,Stride=2,Name="conv_skip")
    batchNormalizationLayer
    reluLayer(Name="relu_skip")
    
];

%添加跳过连接
net = addLayers(net,layers);
net = connectLayers(net,"relu_1","conv_skip");
net = connectLayers(net,"relu_skip","add/in2");

%为回归输出添加全连接层
layers = fullyConnectedLayer(numResponses,Name="fc_2");
net = addLayers(net,layers);
net = connectLayers(net, "add", "fc_2");

%figure
%plot(net)

%对应frameSegmenter-dlnetwork

%%
save("BBB.mat","detector","net")