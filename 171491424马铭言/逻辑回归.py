
import pandas

 

data=pandas.read_csv('D:\\DATA\\pycase\\number2\\4.4\\Data.csv')

 

# 1 进行数据质量的分析（缺失值、异常值、一致性分析）基本描述，检查空值

 

data.describe()

 

# 此处逻辑回归模型，此外数据量足够大，使用清除方法

 

data=data.dropna()

 

data.shape

 

 

# 2 数据变换

# 对离散特征进行虚拟变量处理

# 分开为后续预测做蒲地奥，直接调用

 

dummyColumns=[

       'Gender', 'Home Ownership', 

    'Internet Connection', 'Marital Status',

    'Movie Selector', 'Prerec Format', 'TV Signal'

    ]

 

# 将逻辑变量进行类型转换

 

for column in dummyColumns:

    data[column]=data[column].astype('category')

    

dummiesData=pandas.get_dummies(

        data,

        columns=dummyColumns,

        prefix_sep=" ",

        drop_first=True

        )

 

# 以性别为例，通过去重查看处理效果,查看某列属性的方法，两种，“。”和【】

 

dummiesData.columns

 

data.Gender.unique()

 

data['Gender'].unique()

 

dummiesData['Gender Male'].unique()

 

"""

博士后    Post-Doc

博士      Doctorate

硕士      Master's Degree

学士      Bachelor's Degree

副学士    Associate's Degree

专业院校  Some College

职业学校  Trade School

高中      High School

小学      Grade School

"""

# 有大小离散特征的转化

 

educationLevelDict = {

    'Post-Doc': 9,

    'Doctorate': 8,

    'Master\'s Degree': 7,

    'Bachelor\'s Degree': 6,

    'Associate\'s Degree': 5,

    'Some College': 4,

    'Trade School': 3,

    'High School': 2,

    'Grade School': 1

}

 

# 增加数值变量

 

dummiesData['Education Level Map']=dummiesData['Education Level'].map(educationLevelDict)

 

freqMap = {

    'Never': 0,

    'Rarely': 1,

    'Monthly': 2,

    'Weekly': 3,

    'Daily': 4

}

dummiesData['PPV Freq Map'] = dummiesData['PPV Freq'].map(freqMap)

dummiesData['Theater Freq Map'] = dummiesData['Theater Freq'].map(freqMap)

dummiesData['TV Movie Freq Map'] = dummiesData['TV Movie Freq'].map(freqMap)

dummiesData['Prerec Buying Freq Map'] = dummiesData['Prerec Buying Freq'].map(freqMap)

dummiesData['Prerec Renting Freq Map'] = dummiesData['Prerec Renting Freq'].map(freqMap)

dummiesData['Prerec Viewing Freq Map'] = dummiesData['Prerec Viewing Freq'].map(freqMap)

 

dummiesData.columns

 

# 选取特征值

 

dummiesSelect = [

    'Age', 'Num Bathrooms', 'Num Bedrooms', 'Num Cars', 'Num Children', 'Num TVs', 

    'Education Level Map', 'PPV Freq Map', 'Theater Freq Map', 'TV Movie Freq Map', 

    'Prerec Buying Freq Map', 'Prerec Renting Freq Map', 'Prerec Viewing Freq Map', 

    'Gender Male',

    'Internet Connection DSL', 'Internet Connection Dial-Up', 

    'Internet Connection IDSN', 'Internet Connection No Internet Connection',

    'Internet Connection Other', 

    'Marital Status Married', 'Marital Status Never Married', 

    'Marital Status Other', 'Marital Status Separated', 

    'Movie Selector Me', 'Movie Selector Other', 'Movie Selector Spouse/Partner', 

    'Prerec Format DVD', 'Prerec Format Laserdisk', 'Prerec Format Other', 

    'Prerec Format VHS', 'Prerec Format Video CD', 

    'TV Signal Analog antennae', 'TV Signal Cable', 

    'TV Signal Digital Satellite', 'TV Signal Don\'t watch TV'

]

 

inputData = dummiesData[dummiesSelect]

 

# 选取结果值

 

outputData = dummiesData[['Home Ownership Rent']]

 

 

# 导入逻辑回归的方法

 

 

from sklearn import linear_model

 

lrModel = linear_model.LogisticRegression()

 

lrModel.fit(inputData, outputData)

 

lrModel.score(inputData, outputData)

 

 

## 数据预测准备，需要对数据进行同样的标准化处理才可以进行预测

 

newData = pandas.read_csv(

    'D:\\DATA\\pycase\\number2\\4.4\\newData.csv', 

    encoding='utf8'

)

 

# 变量转换需要和样本的准换类型相一致一致

 

for column in dummyColumns:

    newData[column] = newData[column].astype(

        'category', 

        categories=data[column].cat.categories

    )

 

newData = newData.dropna()

 

# 直接调用样本的方法

 

newData['Education Level Map'] = newData['Education Level'].map(educationLevelDict)

 

newData['PPV Freq Map'] = newData['PPV Freq'].map(freqMap)

newData['Theater Freq Map'] = newData['Theater Freq'].map(freqMap)

newData['TV Movie Freq Map'] = newData['TV Movie Freq'].map(freqMap)

newData['Prerec Buying Freq Map'] = newData['Prerec Buying Freq'].map(freqMap)

newData['Prerec Renting Freq Map'] = newData['Prerec Renting Freq'].map(freqMap)

newData['Prerec Viewing Freq Map'] = newData['Prerec Viewing Freq'].map(freqMap)

 

dummiesNewData = pandas.get_dummies(

    newData, 

    columns=dummyColumns,

    prefix=dummyColumns,

    prefix_sep=" ",

    drop_first=True

)

 

inputNewData = dummiesNewData[dummiesSelect]

 

lrModel.predict(inputData)
