# encoding: utf-8
import csv
import sys
import random
import string


def buildRow():
    url = "http://"+string.join(random.sample("abcdefghijklmnopqrstuvwxyz", 5),"")
    sourceip = "192.168.1."+str(random.randint(1,10))
    sourceport = str(random.randint(50000,56000))
    targetip = "192.168.1."+str(random.randint(1,10))
    targetport = str(80)
    length = str(random.randint(1024,5000000))
    return (sourceip,sourceport,targetip,targetport,length,url)


for i in range(1, 2):
    filename = "/Users/boris/DevOps/python/tools/org/boris/python/part-"+string.join(random.sample("abcdefghijklmnopqrstuvwxyz", 5),"")+".csv"
    csvfile = open(filename, "w")
    writer = csv.writer(csvfile)
    lenght = 0
    data =[]
    count = 0
    while(lenght<1024*1024*40):
        row = buildRow()
        data.append(row)
        for x in range(0,5):
            lenght = lenght+len(row[x])+5
        count = count + 1
        if count % 1000 == 0 :
            print "文件:"+filename+"--"+str(count)
    writer.writerows(data)
    csvfile.close()
