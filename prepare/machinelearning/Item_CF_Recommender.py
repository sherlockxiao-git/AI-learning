# Refer to https://github.com/llSourcell/recommender_live for more details
from __future__ import division  
import numpy as np
import pandas

class item_base_recommender:
    mu=0
    
    def __init__(self,X):
        self.X = X
        mu = np.mean(self.X['play_count'])  #average play count
        self.ItemsForUser={}   #user播放过的所有song
        self.UsersForItem={}   #给song打过分的所有user
        self.SongIntex={}   #user播放过的所有song
        self.UserIntex={}   #给song打过分的所有user
        
        
        for i in range(self.X.shape[0]):
            userid=self.X['user'].values[i]  #user
            songid=self.X['song'].values[i] #song
            rat=self.X['play_count'].values[i]  #listen_count
            self.UsersForItem.setdefault(songid,{})
            self.ItemsForUser.setdefault(userid,{})

            self.UsersForItem[songid][userid]=rat  
            self.ItemsForUser[userid][songid]=rat   
            #self.similarity.setdefault(i_id,{})        
        pass  
        
        for intex,Songitem in zip(range(len(self.UsersForItem.keys())),self.UsersForItem.keys()):
            self.SongIntex.setdefault(Songitem,intex)
        
        for intex,Useritem in zip(range(len(self.ItemsForUser.keys())),self.ItemsForUser.keys()):
            self.UserIntex.setdefault(Useritem,intex)

        
        n_Items = len(self.UsersForItem) #数组的索引从0开始，浪费第0个元素
        
        
        
        print(n_Items-1)
        #n_Items 的原因是要测试集数据可能会有新的item出现，完全写死可能会导致问题
        self.similarity = np.zeros((n_Items *2, n_Items*2), dtype=np.float)
        self.similarity[:,:] = -1
        
           
    #计算Item i_id1和i_id2之间的相似性
    def sim_cal(self, i_id1, i_id2):
        
        i_id1_index=self.SongIntex.get(i_id1)
        i_id2_index=self.SongIntex.get(i_id2)

        if self.similarity[i_id1_index][i_id2_index]!=-1:  #如果已经计算好
            return self.similarity[i_id1_index][i_id2_index]  
        
        si={}  
        for user in self.UsersForItem[i_id1]:  #所有对Item1打过分的的user
            if user in self.UsersForItem[i_id2]:  #如果该用户对Item2也打过分
                #print self.UsersForItem[i_id2]
                si[user]=1  #user为一个有效用用户
        
        #print si
        n=len(si)   #有效用户数，有效用户为即对Item1打过分，也对Item2打过分
        
        if (n==0):  #没有共同打过分的用户，相似度设为1.因为最低打分为1？
            self.similarity[i_id1_index][i_id2_index]=0  
            self.similarity[i_id2_index][i_id1_index]=0  
            return 0  
        
        #所有有效用户对Item1的打分
        s1=np.array([self.UsersForItem[i_id1][u] for u in si])  
        
        #所有有效用户对Item2的打分
        s2=np.array([self.UsersForItem[i_id2][u] for u in si])  
        
        sum1=np.sum(s1)  
        sum2=np.sum(s2)  
        sum1Sq=np.sum(s1**2)  
        sum2Sq=np.sum(s2**2)  
        pSum=np.sum(s1*s2)  
        
        #分子
        num=pSum-(sum1*sum2/n)  
        
        #分母
        den=np.sqrt((sum1Sq-sum1**2/n)*(sum2Sq-sum2**2/n))  
        if den==0:  
            self.similarity[i_id1_index][i_id2_index]=0  
            self.similarity[i_id2_index][i_id1_index]=0  
            return 0  
        
        self.similarity[i_id1_index][i_id2_index]=num/den  
        self.similarity[i_id2_index][i_id1_index]=num/den  
        return num/den  
    
    
        #预测USER对Songid的点击量
    def predict(self,uid,i_id):  
        sim_accumulate=0.0  
        rat_acc=0.0  
        
        if(i_id == 599):    
            print(self.UsersForItem[i_id])
            
        for item in self.ItemsForUser[uid]:  #用户uid打过分的所有Item
            sim = self.sim_cal(item,i_id)    #该Item与i_id之间的相似度
            if sim<0:continue  
            #print sim,self.user_movie[uid][item],sim*self.user_movie[uid][item]  
            
            rat_acc += sim * self.ItemsForUser[uid][item]  
            sim_accumulate += sim  
        
        #print rat_acc,sim_accumulate  
        if sim_accumulate==0: #no same user rated,return average rates of the data  
            return  self.mu  
        return rat_acc/sim_accumulate  
    
    
       #测试
    def test(self,test_X):  
        #test_X=np.array(test_X) 
        output=[]  
        sums=0  
        print("the test data size is ",test_X.shape)
        for i in range(test_X.shape[0]):  
            user_id = test_X['user'].values[i]  #user id
            song_id = test_X['song'].values[i] #Song_id 
        
            #设置默认值，否则用户或item没在训练集中出现时会报错
            self.UsersForItem.setdefault(song_id,{})  
            self.ItemsForUser.setdefault(user_id,{})
            
            self.SongIntex.setdefault(song_id,len(self.SongIntex.keys()))
            self.UserIntex.setdefault(user_id,len(self.UserIntex.keys()))
            
            pre=self.predict(user_id, song_id)  
            output.append(pre)  
            #print pre,test_X[i][2]  
            sums += (pre-test_X['play_count'].values[i])*(pre-test_X['play_count'].values[i])
  
        rmse=np.sqrt(sums/test_X.shape[0])  
        print("the rmse on test data is ",rmse)
        test_X['prd_play_count']=output        
        return output,test_X 
    
    def getscore(self,test_X):
        test=test_X
        precisionUp=0
        prd_top_sum=0
        true_top_sum=0
        for uid in test['user'].values:
            all_user0_play_info=test[test['user']==uid]
            prd_top=all_user0_play_info.sort_values(by='prd_play_count',ascending=False).head(20) 
            true_top=all_user0_play_info.sort_values(by='play_count',ascending=False).head(20)
            for true_index in range(len(true_top)):
                if true_top['song'].values[true_index] in prd_top['song'].values:
                    precisionUp=precisionUp+1

            prd_top_sum=prd_top_sum+len(prd_top['user'])
            true_top_sum=true_top_sum+len(true_top['user'])
        precision= precisionUp/true_top_sum
        recall= precisionUp/prd_top_sum
        print("precision:"+str(precision),"recall:"+str(recall))
  