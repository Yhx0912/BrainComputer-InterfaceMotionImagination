import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mne.decoding import CSP


## 2.1 C_mean
class myC_mean():
    def __init__(self,data):
        self.data = data
        
    def get_C_mean_data(self):
        data_mean = self.data.mean(axis=1,keepdims=True) 
        data_C_mean = data_mean.squeeze(1)
        return(data_C_mean)

class myPCA():
    def __init__(self,data):
        self.data = data
        self.model_PCA = PCA()
        self.model_PCA.fit(self.data)
        
    def get_PCA_nums(self):
        explained_variance_ratio =self.model_PCA.explained_variance_ratio_.cumsum()
        index = np.argmax((explained_variance_ratio >0.9)) 
        print("方差解释能力达到90%以上至少需要：",index+1,"个主成分")
        return(index+1)
    def get_plot_PVE(self):
        #可视化  画累计百分比，这样可以判断选几个主成分
        plt.plot(self.model_PCA.explained_variance_ratio_, 'o-')
        plt.xlabel('Principal Component')
        plt.ylabel('Proportion of Variance Explained')
        plt.title('PVE')
    def get_plot_Cumulative_PVE(self):   
        # 解释到90%以上了
        plt.plot(self.model_PCA.explained_variance_ratio_.cumsum(), 'o-')
        plt.xlabel('Principal Component')
        plt.ylabel('Cumulative Proportion of Variance Explained')
        plt.axhline(0.9, color='k', linestyle='--', linewidth=1)
        plt.title('Cumulative PVE')
        
    def fit(self,X_train,X_test,n_components):
        model = PCA(n_components = n_components)
        model.fit(X_train)
        #得到主成分得分
        X_train_pca = model.transform(X_train)
        X_test_pca = model.transform(X_test)
        return X_train_pca,X_test_pca
    

class myCSP():
    def __init__(self, data, label, n_components):
        # 创建CSP提取特征，这里使用10个分量的CSP
        csp = CSP(n_components=n_components, reg=None, log=False, norm_trace=False)
        self.mycsp = csp.fit(data, label)
        
    def get_train_data(self, data, label):
        # csp提取特征
        X_train_csp = self.mycsp.fit_transform(data, label) 
        return X_train_csp
    
    def get_test_data(self, test_data, test_label):
        X_test_csp = self.mycsp.fit_transform(test_data, test_label)
        return X_test_csp
    
    
class myCNN1d(nn.Module):
    def __init__(self):
        super(myCNN1d, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 维度 [32，64, 641]
        self.cnn = nn.Sequential(
            nn.Conv1d(64, 32, 7, 1, 3),  # [32, 640]
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2, 2, 0),  # [32, 320]

            nn.Conv1d(32, 32, 7, 1, 3),  # [32, 320]
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2, 2, 0),  # [32, 160]

            nn.Conv1d(32, 32, 7, 1, 3),  # [32, 160]
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2, 2, 0),  # [32, 80]

        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 80, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

