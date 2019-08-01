import pandas
import numpy as np
import os
import re
import matplotlib.pyplot as plt1

filename = '2014.csv'
pic_save_path = '2014pic'
if not os.path.exists(pic_save_path):
    os.mkdir(pic_save_path)

dataframe = pandas.read_csv(filename,usecols=[1,7,23],names=None,encoding='ISO-8859-1',header=None,low_memory=False)
dataframe = dataframe[1:-1]

sdate = dataframe[1]
sdate = np.asarray(sdate)
num = dataframe[23]

num = np.asarray(num).astype(float).astype(int)
flag = dataframe[7]
flag = np.asarray(flag).astype(int)
s_num_list = []
#2013年末6412063
#2014年末6580049
#2015年末6336114
#2016年末6420086
#2017年末6419936
sum=6412063
s_date_list = []
new_sdate = []
for i in sdate:
    tmp = i.split(' ')
    if len(tmp) != 0:
        new_sdate.append(tmp[0])
    else:
        new_sdate.append(0)
        print('DATE SPLIT ERROR!')
sdate = np.asarray(new_sdate)
for s in range(len(sdate)):
    if flag[s]==0:
        sum=sum+num[s]
    else:
        sum=sum-num[s]
    if sdate[s] != sdate[s - 1]:
        s_num_list.append(sum)
        s_date_list.append(sdate[s-1])
    elif s==len(sdate)-1:
        s_num_list.append(sum)
        s_date_list.append(sdate[s])
s_num_array=np.asarray(s_num_list)
s_date_array=np.asarray(s_date_list)
month_list = []
date_mo_dict = {}
for i in s_date_array:
    tmp = i.split()
    if len(tmp)!=0:
        tmp = tmp[0]
        _tmp = re.split('/',tmp)
        if len(_tmp)!=0:
            month= int(_tmp[1])
        else:
            month = 0
            print('GET MONTH ERROR!')
        month_list.append(month)

for j in range(len(month_list)):
    if month_list[j] not in date_mo_dict.keys():
        date_mo_dict[month_list[j]] = [s_num_array[j],s_date_array[j]]
    else:
        tmp_=date_mo_dict[month_list[j]]
        date_mo_dict[month_list[j]]=np.vstack((tmp_,[s_num_array[j],s_date_array[j]]))
for k in date_mo_dict.keys():
    array = date_mo_dict[k]

    # array = array[array[:,1].argsort()]
    num_,date_list= np.split(array,[1],axis=1)
    r = []
    for i in date_list:
        tmp = i[0].split('/')
        if len(tmp) != 0:
            r.append(int(tmp[2]))
        else:
            r.append(0)
            print('SPLIT DATE ERROR')
    sort_dict = dict(zip(r,array))
    list_sorted = sorted(sort_dict.keys())
    sort_array = []
    for i in list_sorted:
        sort_array.append(sort_dict[i])
    num_, date_list= np.split(sort_array, [1], axis=1)
    num_ = num_.reshape(num_.shape[0]).astype(int)
    date_list = date_list.reshape(date_list.shape[0])
    date_list = list(date_list)
    print(num_)
    print(date_list)
    fig = plt1.figure(figsize=(9,11))
    plt1.plot(date_list,num_)
    plt1.xticks(rotation=245)
    plt1.tick_params(labelsize=11)

    plt1.savefig('./{}/{}_month.png'.format(pic_save_path,k))
    # if k in [1,3,5,7,8,10,12]:
    #     day_list = np.zeros(32)
    #     d=0
    #     for x in date_list:
    #         day_=re.split('/',x)
    #         day=int(day_[2])
    #         day_list[day-1]=day_list[day-1]+int(num_[d])
    #         d=d+1
    # elif k == 2:
    #     day_list = np.zeros(29)
    #     d = 0
    #     for x in date_list:
    #         day_ = re.split('/', x)
    #         day = int(day_[2])
    #         day_list[day - 1] = day_list[day - 1] + int(num_[d])
    #         d = d + 1
    # else:
    #     day_list = np.zeros(31)
    #     d = 0
    #     for x in date_list:
    #         day_ = re.split('/', x)
    #         day = int(day_[2])
    #         day_list[day - 1] = day_list[day - 1] + int(num_[d])
    #         d = d + 1
    # num_list.append([int(i) for i in day_list])

#print(np.asarray(num_list).shape)
