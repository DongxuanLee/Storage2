import pandas
import numpy as np
import time
# filename = '2015.csv'

def read_file1(filename):
    dataframe = pandas.read_csv(filename,usecols=[2,9,21,33],names=['date','flag','num','batch'],encoding='ISO-8859-1',header=None,low_memory=False)
    dataframe = dataframe[1:]
    return dataframe
def BatchStatistics(filename):
    batch_dict = {}
    dataframe = read_file1(filename)
    for i,data in dataframe.iterrows():
        if data['batch'] not in batch_dict.keys():
            batch_dict[data['batch']] = np.asarray([data['flag'],data['date'],data['num']]).reshape((1,3))
        else:
            tmp_ = batch_dict[data['batch']]
            batch_dict[data['batch']] = np.vstack((tmp_,[data['flag'],data['date'],data['num']]))
    day_list = []
    print(batch_dict['100480'])
    for k in batch_dict.keys():
        array = batch_dict[k]
        array = np.asarray(array)
        flag,date,num = np.split(array,3,axis=1)
        flag = flag.reshape(flag.shape[0]).astype(int)
        date = date.reshape(date.shape[0])
        new_date = []
        for j in date:
            tmp = j.split(' ')
            if len(tmp) !=0:
                new_date.append(tmp[0])
            else:
                new_date.append(0)
                print('DATE SPLIT ERROR!')
            date = np.asarray(new_date)
        num = num.reshape(num.shape[0]).astype(float)
        sum = 0
        date1 = time.mktime(time.strptime('2018/10/22',"%Y/%m/%d"))
        for i in range(len(flag)):
            if flag[i] ==0 or flag[i] ==1:
                sum = num[i]
                date1 =  time.mktime(time.strptime(date[i],"%Y/%m/%d"))
            elif flag[i] ==3:
                sum = sum-num[i]
                if sum <=0:
                    date2 = time.mktime(time.strptime(date[i],"%Y/%m/%d"))
                    # print(date1)
                    # print(date2)
                    day_ = int((date2 - date1)/(24*60*60))
                    if day_>=0:
                        day_list.append(day_)
    day_max = np.max(day_list)
    print(day_max)
    day_mean = np.mean(day_list)
    print(day_mean)
    day_median = np.median(day_list)
    # print(day_median)


def main():
    year = input('input the year you want to look:')
    # filename = ('{}.csv'.format(year))
    jsonname = '{}_goods_batch.json'.format(int(year)-1)
    # dataframe = read_file(filename)
    # sdate = dataframe[2]
    # new_sdate = []
    # for i in sdate:
    #     tmp = i.split(' ')
    #     if len(tmp) != 0:
    #         new_sdate.append(tmp[0])
    #     else:
    #         new_sdate.append(0)
    #         print('DATE SPLIT ERROR!')
    # sdate = np.asarray(new_sdate)
    # flag = dataframe[9]
    # flag = np.asarray(flag).astype(int)
    # goods = dataframe[14]
    # goods = np.asarray(goods)
    # num = dataframe[21]
    # num = abs(np.asarray(num).astype(float))
    # batch = dataframe[33]
    # batch = np.asarray(batch)
    # position = dataframe[35]
    # position = np.asarray(position)
    # goodsplit(goods,batch,sdate,flag,num,jsonname,year)
    # MaxGoods(jsonname,year)
    MaxStore(jsonname,year)
def read_file(filename):
    dataframe = pandas.read_csv(filename,usecols=[2,9,14,21,33,35],names=None,encoding='ISO-8859-1',header=None,low_memory=False)
    dataframe = dataframe[1:]
    return dataframe
def read_json(jsonname,year):
    if year == '2013':
        goods_dict = {}
    else:
        with open(jsonname,'r') as ff:
            goods_dict = json.load(ff)
        for i in goods_dict.keys():
            goods_dict[i] = np.asarray( goods_dict[i])
    return goods_dict
def goodsplit(goods,batch,sdate,flag,num,jsonname,year):
    goods_dict = read_json(jsonname,year)
    for i in range(len(goods)):
        if goods[i] not in goods_dict.keys() :
            if flag[i] ==0 or flag[i]==1:
                goods_dict[goods[i]] =np.asarray([batch[i],sdate[i],num[i]]).reshape((1,3))
        else:
            if flag[i]==0 or flag[i]==1:
                tmp_ = goods_dict[goods[i]]
                goods_dict[goods[i]]= np.vstack((tmp_,[batch[i],sdate[i],num[i]]))
    with open('{}_goods_batch.json'.format(year), 'w') as j:
        for i in goods_dict.keys():
            goods_dict[i]=goods_dict[i].tolist()
        json.dump(goods_dict, j)
def batchinfo(batch,flag,sdate,num):
    batch_dict = {}
    for i in range(len(sdate)):
        if batch[i] not in batch_dict.keys():
            batch_dict[batch[i]]=np.asarray([flag[i],sdate[i],num[i]]).reshape((1,3))
        else:
            tmp_ = batch_dict[batch[i]]
            batch_dict[batch[i]] = np.vstack((tmp_,[flag[i],sdate[i],num[i]]))
    batch_time = {}
    for k in batch_dict.keys():
        array = batch_dict[k]
        array = np.asarray(array)
        sum = 0
        date1 = time.mktime(time.strptime('2018/10/22', "%Y/%m/%d"))
        for i in array:
            if i[0]==0 or i[0]==1:
                sum = i[2]
                date1 = time.mktime(time.strptime(i[1],"%Y/%m/%d"))
            elif i[0]==3:
                sum = sum-num[i]
                if sum<=0:
                    date2 = time.mktime(time.strptime(i[1],"%Y/%m/%d"))
                    day_ = int((date2-date1)/(24*60*60))
                    if day_>=0:
                        batch_time[k] = day_

def MaxGoods(jsonname,year):
    goods_dict = read_json(jsonname, year)
    batch_num_list = []
    max_batchs = 1
    max_goods = '1017'
    for k in goods_dict.keys():
        goods_dict[k] = np.asarray(goods_dict[k])
        batch_nums = goods_dict[k].shape[0]
        batch_num_list.append(batch_nums)
        if batch_nums>max_batchs:
            max_batchs= batch_nums
            max_goods = k
    print(max_batchs)
    print(max_goods)
def MaxStore(jsonname,year):
    goods_dict = read_json(jsonname, year)
    max_nums=1
    store_num_list = []
    max_goods = '1017'
    for k in goods_dict.keys():
        sum = 0
        for i in goods_dict[k]:
            sum = sum+float(i[2])
        if sum ==14535071.0:
            print(k)
    #     store_num_list.append(sum)
    #
    #     if sum>max_nums :
    #         max_nums = sum
    #         max_goods = k
    # print(sorted(store_num_list,reverse=True))
    # # print(max_goods)

if __name__ == '__main__':
    main()