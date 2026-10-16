import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_features=2,lstm_hidden_size=128, num_classes=3):

        super(LSTMModel, self).__init__()

        self.lstm1 = nn.LSTM(
            input_size=input_features,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.lstm2 = nn.LSTM(
            input_size=lstm_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.classifier=nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        # x的形状为（N，1，2，1024）

        # 数据整形
        x=x.squeeze(1)  #->（N，2，1024）
        x=x.permute(0, 2, 1)   #->(N,1024,2)

        out1,_=self.lstm1(x)
        out2,_=self.lstm2(out1)
        last_step_output=out2[:,-1,:] #->(N,128)

        logits=self.classifier(last_step_output)

        return logits

def build_model(classes):
        return LSTMModel(num_classes=classes)
