import pymongo

myclient = pymongo.MongoClient("mongodb://admin:123456@172.16.19.81:27017/")
print(myclient.list_database_names())
mydb = myclient["runoobdb"]
mycol = mydb["xxm"]