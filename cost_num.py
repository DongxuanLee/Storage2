import pandas
import numpy as np
import os
import re
import matplotlib.pyplot as plt1
import json

zp_cost = 10
sp_cost = 8

def main():
    year= input('input the year you want to look:')
    if year =='2013':
        x=0
    elif year == '2014':
        x=6504137.725
    elif year == '2015':
        x=6766288
    elif year == '2016':
        x=6336114
    elif year == '2017':
        x=6871567
    elif year == '2018':
        x=6328173
    filename= ('{}.csv'.format(year))
    jsonname = ('./all-json/{}_lastday.json'.format(int(year)-1))
    txtname = ('./all-cost-txt/{}_c_year.txt'.format(int(year) - 1))
    pic_save_path = ('./cost_pic/{}pic'.format(year))
    if not os.path.exists(pic_save_path):
        os.mkdir(pic_save_path)
    dataframe = read_file(filename)
    sdate = dataframe[2]
    new_sdate = []
    for i in sdate:
        tmp = i.split(' ')
        if len(tmp) != 0:
            new_sdate.append(tmp[0])
        else:
            new_sdate.append(0)
            print('DATE SPLIT ERROR!')
    sdate = np.asarray(new_sdate)
    flag = dataframe[7]
    flag = np.asarray(flag).astype(int)
    pid = dataframe[14]
    pid = np.asarray(pid)
    num = dataframe[21]
    num = abs(np.asarray(num).astype(float))
    pos_flag = dataframe[27]
    pos_flag = np.asarray(pos_flag).astype(int)
    position = dataframe[35]
    position = np.asarray(position)
    s_date_array = date_set(sdate)
    s_num_array = cal_num(x,num,sdate,flag,pid)
    cost = cal_cost(num,sdate,flag,position,pos_flag,pid,year,jsonname,txtname)
    month_split(s_date_array,s_num_array,cost,pic_save_path)

def read_file(filename):
    dataframe = pandas.read_csv(filename,usecols=[2,7,14,21,27,35],names=None,encoding='ISO-8859-1',header=None,low_memory=False)
    dataframe = dataframe[1:]
    return dataframe
def read_json(jsonname):
    with open(jsonname,'r') as ff:
        pos_dict = json.load(ff)
        # if jsonname =='2013_lastday.json':
        #     pos_dict2 = dict(
        #         zip(['4701', '1', '3819', '4127', '21820', '671', '23185', '19754'], [240, 5000, 5449,10252, 2, 251, 1704, 554]))
        #     pos_dict = {**pos_dict, **pos_dict2}
        # elif jsonname =='2014_lastday.json':
        #     pos_dict2 = dict(
        #         zip(['312','4639','20720','23671'],[284,300,5,50]))
        # pos_dict = {**pos_dict, **pos_dict2}
    return pos_dict
def date_set(sdate):
    s_date_list = []
    for s in range(len(sdate)):
        if sdate[s] != sdate[s - 1]:
            s_date_list.append(sdate[s - 1])
        elif s == len(sdate) - 1:
            s_date_list.append(sdate[s])
    s_date_array = np.asarray(s_date_list)
    return s_date_array

def cal_num(x,num,sdate,flag,pid):
    sum = x
    s_num_list = []
    for s in range(len(sdate)):
        if pid[s] =='35508'or pid[s] =='35509':
            sum = sum
        else:
            if s ==0:
                sum = num[0]
            else:
                if sdate[s]!=sdate[s-1]:
                    s_num_list.append(sum)
                if flag[s] == 0:
                    sum = sum + num[s]
                else:
                    sum = sum - num[s]
                if s ==len(sdate)-1:
                    s_num_list.append(sum)

    s_num_array = np.asarray(s_num_list)
    # print(s_num_array[-1])
    return s_num_array

def cal_cost(num,sdate,flag,position,pos_flag,pid,year,jsonname,txtname):

    cost = []
    if year == '2013':
        pos_dict = {'4694':2000}
        zp = 0
        sp = 0
    else:
        pos_dict = read_json(jsonname)
        with open(txtname, 'r') as f:
            lines = f.readlines()
            line = []
            for i in lines:
                line.append(i)
        zp = float(line[0])
        sp = float(line[1])
    for s in range(len(sdate)):
        # if position[s] == '1':
        #     print('-------可爱的分割线--------')
        #     print(s)
        #     print(flag[s])
        #     print(num[s])
        #     print(pos_dict[position[s]])
        if pid[s] =='35508' or pid[s] =='35509':
            print('塑料袋')
        else:
            if s<len(sdate)-2and position[s] == position[s + 1] and num[s] == num[s + 1] and flag[s] == 1 and flag[s + 1] == 0 :
                flag[s] = 0
                flag[s + 1] = 1
            if flag[s] == 0:
                if position[s] not in pos_dict.keys():
                    pos_dict[position[s]] = num[s]
                    if pos_flag[s] == 0:
                        zp = zp + 1
                    else:
                        sp = sp + 1
                else:
                    pos_dict[position[s]] = pos_dict[position[s]] + num[s]
            else:
                if position[s] not in pos_dict.keys():
                    print ('out error but i have no idea!')
                    pos_dict[position[s]] = num[s]
                    pos_dict[position[s]] = pos_dict[position[s]] - num[s]
                    if pos_dict[position[s]] <= 0:
                        del pos_dict[position[s]]
                else:
                    pos_dict[position[s]] = pos_dict[position[s]] - num[s]
                    if pos_dict[position[s]] <= 0:
                        del pos_dict[position[s]]
                        if pos_flag[s] == 0:
                            zp = zp - 1
                        else:
                            sp = sp - 1
        pos_list = []
        if sdate[s] != sdate[s - 1]:
            for i in pos_dict.keys():
                pos_list.append(i)
            cost_sum = zp * zp_cost + sp * sp_cost
            cost.append(cost_sum)
        if s == len(sdate) - 1:
            for i in pos_dict.keys():
                pos_list.append(i)
            cost_sum = zp * zp_cost + sp * sp_cost
            cost.append(cost_sum)
            with open('./all-json/{}_lastday.json'.format(year),'w') as j:
                json.dump(pos_dict,j)
            with open('./all-cost-txt/{}_c_year.txt'.format(year), 'a') as tt:
                tt.write(str(zp) + '\n')
                tt.write(str(sp) + '\n')
    return cost

def month_split(s_date_array,s_num_array,cost,pic_save_path):
    month_list = []
    for i in s_date_array:
        tmp = i.split()
        if len(tmp) != 0:
            tmp = tmp[0]
            _tmp = re.split('/', tmp)
            if len(_tmp) != 0:
                month = int(_tmp[1])
            else:
                month = 0
                print('GET MONTH ERROR!')
            month_list.append(month)
    date_mo_dict = {}
    for j in range(len(month_list)):
        if month_list[j] not in date_mo_dict.keys():
            date_mo_dict[month_list[j]] = [s_num_array[j],cost[j], s_date_array[j]]
        else:
            tmp_ = date_mo_dict[month_list[j]]
            date_mo_dict[month_list[j]] = np.vstack((tmp_, [s_num_array[j],cost[j],s_date_array[j]]))
    for k in date_mo_dict.keys():
        array = date_mo_dict[k]
        num_,cost_,date_list = np.split(array, 3, axis=1)
        r = []
        for i in date_list:
            tmp = i[0].split('/')
            if len(tmp) != 0:
                r.append(int(tmp[2]))
            else:
                r.append(0)
                print('SPLIT DATE ERROR')
        sort_dict = dict(zip(r, array))
        list_sorted = sorted(sort_dict.keys())
        sort_array = []
        for i in list_sorted:
            sort_array.append(sort_dict[i])
        sort_array = np.asarray(sort_array)
        num_,cost_,date_list = np.split(sort_array, 3, axis=1)
        num_ = num_.reshape(num_.shape[0]).astype(float).astype(str)
        cost_ = cost_.reshape(cost_.shape[0]).astype(float)
        date_list = date_list.reshape(date_list.shape[0])
        date_list = list(date_list)
        paint(num_,cost_,pic_save_path,k)

def paint(x,y,pic_save_path,k):
    plt1.figure(figsize=(9,11))
    plt1.plot(x,y)
    plt1.xticks(rotation=245)
    plt1.tick_params(labelsize=11)
    plt1.savefig('./{}/{}_month.png'.format(pic_save_path,k))

if __name__ == '__main__':
    main()