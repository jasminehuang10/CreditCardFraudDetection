#用户：jasmineHuang

#日期：2019-02-27   

#时间：14:16   

#文件名称：PyCharm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns

creditcard_data=pd.read_csv('creditcard.csv')
#查看数据概览
print(creditcard_data.info())
print(creditcard_data.describe())
#查看正常和欺诈的比例
count_classes=creditcard_data.Class.value_counts()
count_classes.plot(kind="bar")
plt.show()
#诈骗和正常的时间序列下交易发生的频率
plt.subplot(2,1,1)
plt.hist(creditcard_data.Time[creditcard_data.Class==1],bins=50)
plt.title('fraud')
plt.ylabel('transaction numbers')
plt.subplot(2,1,2)
plt.hist(creditcard_data.Time[creditcard_data.Class==0],bins=50)
plt.title('normal')
plt.ylabel('transaction numbers')
plt.subplots_adjust(wspace=0,hspace=0.5)
plt.show()
#诈骗和正常的交易金额的频率
plt.subplot(2,1,1)
plt.hist(creditcard_data.Amount[creditcard_data.Class==1],bins=30)
plt.title('fraud')
plt.ylabel('transaction amount')
plt.subplot(2,1,2)
plt.hist(creditcard_data.Amount[creditcard_data.Class==0],bins=30)
plt.title('normal')
plt.ylabel('transaction amount')
plt.subplots_adjust(wspace=0,hspace=0.5)
plt.show()
#各特征和因变量之间的关系
#获取自变量特征列表
features=[x for x in creditcard_data.columns if x not in ['Time','Amount',
                                                          'Class']]
plt.figure(figsize=(12,28*4))
#隐式指定网格的行列数
gs=gridspec.GridSpec(28,1)
for i,cn in enumerate(features):
    ax=plt.subplot(gs[i])
    sns.distplot(creditcard_data[cn][creditcard_data.Class==1],bins=50,
                 color='red')
    sns.distplot(creditcard_data[cn][creditcard_data.Class == 0] , bins=50 ,
                 color='green')
    ax.set_title(str(cn))
plt.subplots_adjust(wspace=0,hspace=0.5)
plt.savefig('各变量与class的关系.png',transparent=False,bbox_inches='tight')
#先把数据分为欺诈和正常，按比例产生训练和测试组
fraud=creditcard_data[creditcard_data.Class==1]
normal=creditcard_data[creditcard_data.Class==0]
x_train=fraud.sample(frac=0.7)
x_train=pd.concat([x_train,normal.sample(frac=0.7)])
x_test=creditcard_data.loc[~creditcard_data.index.isin(x_train.index)]
y_train=x_train.Class
y_test=x_test.Class
x_train=x_train.drop(['Class','Time'],axis=1)
x_test=x_test.drop(['Class','Time'],axis=1)
print(x_test.shape,x_train.shape,y_test.shape,y_train.shape)
#用逻辑回归的方法进行建模分析（二分类问题&全是数值型）
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

rfmodel=RandomForestClassifier()
rfmodel.fit(x_train,y_train)
print(rfmodel)

#查看混淆矩阵
ypred_lr=rfmodel.predict(x_test)
print(metrics.confusion_matrix(y_test,ypred_lr))
#查看分类报告
print(metrics.classification_report(y_test,ypred_lr))
#查看预测精度和覆盖面-Accuracy:0.999508-Area under the curve:0.891833
print('Accuracy:%f'%(metrics.accuracy_score(y_test,ypred_lr)))
print('Area under the curve:%f'%(metrics.roc_auc_score(y_test,ypred_lr)))