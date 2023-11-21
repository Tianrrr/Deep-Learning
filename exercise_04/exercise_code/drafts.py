import numpy as np


# df = pd.DataFrame([[1,5,3], [4, 5,7], [7, 8,3]],index=["cobra", "viper", "sidewinder"],columns=["asc","shield","target"])


# data = df.loc[:,df.columns!='target']
# print(data)

# a = []
# for i in range(2):
#     data_dict = {}
#     data_dict['features'] = data.iloc[i]
#     data_dict['target'] = df['target'].iloc[i]
#     a.append(data_dict["features"])

# print(a)

# print('--------------------')
# print(np.stack(a).shape)

# # print(data_dict["features"])


# selected_columns = ['asc', 'target']
# mn, mx, mean = df.min(), df.max(), df.mean()

# column_stats = {}
# for column in selected_columns:
#     crt_col_stats = {'min' : mn[column]
#                      'max' : mx[column],
#                      'mean': mean[column]}
#     column_stats[column] = crt_col_stats  

# print(column_stats) 

# print('---------------------------------')

# df = pd.DataFrame({"A":[12, 4, 5, None, 1], 
#                    "B":[7, 2, 54, 3, None], 
#                    "C":[20, 16, 11, 3, 8], 
#                    "D":[14, 3, None, 2, 6]}) 
  
# print(df)
# # skip the Na values while finding the mean 
# print(df.mean(axis = 0,))

# x = np.array([[1,2,3],[4,2,1],[5,3,7],[1,1,2]])
# print(x)
# print(x.shape)
# batch_size,_ = x.shape
# # print(batch_size)
# w = np.array([[1],[1],[1],[1]])
# X = np.concatenate((x,np.ones((batch_size,1))),axis=1)
# print(X)
# # print(np.dot(X,w))

# y = np.array([1,2,3,4])
# print(X)
# print(y)
# print(y*(1-y)*X)
# for i in tqdm(range(10)):
#     time.sleep(1) # do something
#     pass
# class person(object):
    
#     def __init__(self, name, adresse,urtl='123123'):
#         self.adresse = adresse
#         self.name = name
#         self.urtl = urtl
#     def sleep(self):
#         print('sleeping')


# class student(person):
#     def __init__(self,school, *args, **kargs):
#         super(student,self).__init__(*args, **kargs)
#         self.school = school


# s = student(name='xiaoli',adresse='asdadasdasd',school=111)
# print(s.school)


x = np.array([3, 1, 2])
print(np.argsort(x))
    
print(not None)