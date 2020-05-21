# coding: utf-8

#model based CF
from __future__ import division  
import numpy as np  
import scipy as sp  
from numpy.random import random  

class  SVD_CF:  
    def __init__(self, X, k=20):  
        ''''' 
            k  is the number of latent componets 
        '''  
        self.X = X  
        self.k = k  
        self.mu = np.mean(self.X['listen_count'])  #listen_count
        
        #init parameters
        self.bi={}  
        self.bu={}  
        
        self.qi={}  
        self.pu={}  
        
        self.ItemsForUser={}  #每个Item对应的用户
        self.UsersForItem={}  #每个用户对哪些Item打过分
        
        for i in range(self.X.shape[0]):  
            uid=self.X['user'].values[i]  #user
            i_id=self.X['song'].values[i] #song 
            rat=self.X['listen_count'].values[i]  #listen_count
            
            self.ItemsForUser.setdefault(i_id,{})  
            self.UsersForItem.setdefault(uid,{}) 
            
            self.ItemsForUser[i_id][uid]=rat  
            self.UsersForItem[uid][i_id]=rat  
            
            self.bi.setdefault(i_id,0)  
            self.bu.setdefault(uid,0)  
            
            self.qi.setdefault(i_id,random((self.k,1))/10*(np.sqrt(self.k)))  
            self.pu.setdefault(uid,random((self.k,1))/10*(np.sqrt(self.k)))  
                    
    #根据当前参数，预测用户uid对Item（i_id）的打分
    def pred(self,uid,i_id):  
        self.bi.setdefault(i_id,0)  
        self.bu.setdefault(uid,0)  
        
        self.qi.setdefault(i_id,np.zeros((self.k,1)))  
        self.pu.setdefault(uid,np.zeros((self.k,1)))  
        
        if (self.qi[i_id].all()==None):  
            self.qi[i_id]=np.zeros((self.k,1))  
        if (self.pu[uid].all()==None):  
            self.pu[uid]=np.zeros((self.k,1))  
        
        ans=self.mu + self.bi[i_id] + self.bu[uid] + np.sum(self.qi[i_id]*self.pu[uid])  
        
        return ans  
    
    #gamma：为学习率
    #Lambda：正则参数
    def train(self,steps=50,gamma=0.04,Lambda=0.15):  
        for step in range(steps):  
            print('the ',step,'-th  step is running')  
            rmse_sum=0.0 
            
            #将训练样本打散顺序
            kk = np.random.permutation(self.X.shape[0])  
            for j in range(self.X.shape[0]):  
                
                #每次一个训练样本
                i=kk[j]  
                uid=self.X['user'].values[i]  #user
                i_id=self.X['song'].values[i] #song 
                rat=self.X['listen_count'].values[i]  #listen_count                
                #预测残差
                eui=rat-self.pred(uid,i_id)  
                #残差平方和
                rmse_sum+=eui**2  
                
                #随机梯度下降，更新
                self.bu[uid]+=gamma*(eui-Lambda*self.bu[uid])  
                self.bi[i_id]+=gamma*(eui-Lambda*self.bi[i_id]) 
                
                temp=self.qi[i_id]  
                self.qi[i_id]+=gamma*(eui*self.pu[uid]-Lambda*self.qi[i_id])  
                self.pu[uid]+=gamma*(eui*temp-Lambda*self.pu[uid])  
            
            #学习率递减
            gamma=gamma*0.93  
            print("the rmse of this step on train data is ",np.sqrt(rmse_sum/self.X.shape[0]))  
            #self.test(test_data)  
            
    def test(self,test_X):  
        output=[]  
        sums=0  
        #test_X=np.array(test_X)  
          
        for i in range(test_X.shape[0]):  #对每个测试样本
            pre=self.pred(test_X['user'].values[i],test_X['song'].values[i])  #预测打分
            output.append(pre)  
            sums+=(pre-test_X['listen_count'].values[i])**2     #残差平方和
        rmse=np.sqrt(sums/test_X.shape[0])  #均方误差
        print("the rmse on test data is ",rmse)
        test_X['prd_play_count']=output
        self.getscore(test_X)
        return output  
    
    def getscore(self,test_X):
        test=test_X
        precisionUp=0
        prd_top_sum=0
        true_top_sum=0
        for uid in test['user'].values:
            all_user0_play_info=test[test['user']==uid]
            prd_top=all_user0_play_info.sort_values(by='prd_play_count',ascending=False).head(20) 
            true_top=all_user0_play_info.sort_values(by='listen_count',ascending=False).head(20) 
            for true_index in range(len(true_top)):
                if true_top['song'].values[true_index] in prd_top['song'].values:
                    precisionUp=precisionUp+1

            prd_top_sum=prd_top_sum+len(prd_top['user'])
            true_top_sum=true_top_sum+len(true_top['user'])
        precision= precisionUp/true_top_sum
        recall= precisionUp/prd_top_sum
        print("precision:"+str(precision),"recall:"+str(recall))
    