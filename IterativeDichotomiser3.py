# import pandas as pd
# import math

# def Entropy(yes, no):
#     if yes == 0 or no == 0:
#         return 0
#     else:
#         total = yes + no
#         return -((yes/total)*math.log2(yes/total) + (no/total)*math.log2(no/total))
    

# # Read the data from the csv file
# data = pd.read_csv('id3_data.csv')

# # Find the unique values in the first column and their counts in data use pandas unique and value_counts
# uniqueValues0 = data.iloc[:, 0].unique() # [' 37 - 38' ' < 37' ' > 38']
# uniqueValues1 = data.iloc[:, 1].unique() # ['Average' 'High' 'Low']
# uniqueValues2 = data.iloc[:, 2].unique() # ['Normal' 'Abnormal']
# uniqueValues3 = data.iloc[:, 3].unique() # ['Public' 'Private']
# uniqueValues4 = data.iloc[:, 4].unique() # ['Yes ' 'No '], Is the target.

# # Count the amount of each unique value in the target column
# countYes = data.iloc[:, 4].value_counts()['Yes '] # 9
# countNo = data.iloc[:, 4].value_counts()['No '] # 5

# # Calculate the entropy of the target column
# entropy = Entropy(countYes, countNo) # 0.9402859586706311

# # delete all rows where the the first column is not uniqueValues0[0] and delete all other columns except the target 
# data00 = data[data.iloc[:, 0] == uniqueValues0[0]].iloc[:, 4]

# # Count the amount of each unique value in the target column
# countYes00 = data00.value_counts().get('Yes ', 0)
# countNo00 = data00.value_counts().get('No ', 0)
# entropy00 = Entropy(countYes00, countNo00)
# print(entropy00)

# # delete all rows where the the first column is not uniqueValues0[1] and delete all other columns except the target 
# data01 = data[data.iloc[:, 0] == uniqueValues0[1]].iloc[:, 4]

# # Count the amount of each unique value in the target column
# countYes01 = data01.value_counts().get('Yes ', 0)
# countNo01 = data01.value_counts().get('No ', 0)
# entropy01 = Entropy(countYes01, countNo01)
# print(entropy01)

# # delete all rows where the the first column is not uniqueValues0[2] and delete all other columns except the target 
# data02 = data[data.iloc[:, 0] == uniqueValues0[2]].iloc[:, 4]

# # Count the amount of each unique value in the target column
# countYes02 = data02.value_counts().get('Yes ', 0)
# countNo02 = data02.value_counts().get('No ', 0)
# entropy02 = Entropy(countYes02, countNo02)
# print(entropy02)

# # Calculate the average information entropy
# averageInformationEntropy = (len(data00)/len(data))*entropy00 + (len(data01)/len(data))*entropy01 + (len(data02)/len(data))*entropy02
# print(averageInformationEntropy)


import pandas as pd
import math

def Entropy(yes, no):
    if yes == 0 or no == 0:
        return 0
    else:
        total = yes + no
        return -((yes/total)*math.log2(yes/total) + (no/total)*math.log2(no/total))

def CalculateEntropyForColumn(data, columnIndex):
    uniqueValues = data.iloc[:, columnIndex].unique()
    totalLength = len(data)
    averageEntropy = 0

    for value in uniqueValues:
        filteredData = data[data.iloc[:, columnIndex] == value].iloc[:, -1]
        countYes = filteredData.value_counts().get('Yes ', 0)
        count_no = filteredData.value_counts().get('No ', 0)
        entropyValue = Entropy(countYes, count_no)
        averageEntropy += (len(filteredData) / totalLength) * entropyValue

    return averageEntropy

# Read the data from the csv file
data = pd.read_csv('id3_data.csv')

print(data)
# Calculate entropy for each column except the last one
maxInformationGain = 0
for i in range(data.shape[1] - 1):
    avgEntropy = CalculateEntropyForColumn(data, i)
    baseEntropy = Entropy(data.iloc[:, -1].value_counts().get('Yes ', 0), data.iloc[:, -1].value_counts().get('No ', 0))
    if baseEntropy - avgEntropy > maxInformationGain:
        maxInformationGain = baseEntropy - avgEntropy
        maxInformationGainIndex = i

print(maxInformationGainIndex)

# The root is the column with the highest information gain and the column index is maxInformationGainIndex.
# In the data find the unique values in the column with the index maxInformationGainIndex and their counts in data use pandas unique
uniqueValues = data.iloc[:, maxInformationGainIndex].unique()

# now delete all rows only where in the column with the index maxInformationGainIndex is not uniqueValues[0] and keep the columns except the one with the index maxInformationGainIndex
data00 = data[data.iloc[:, maxInformationGainIndex] == uniqueValues[0]].drop(data.columns[maxInformationGainIndex], axis=1)
print(data00)
maxInformationGain = 0
for i in range(data00.shape[1] - 1):
    avgEntropy = CalculateEntropyForColumn(data00, i)
    baseEntropy = Entropy(data00.iloc[:, -1].value_counts().get('Yes ', 0), data00.iloc[:, -1].value_counts().get('No ', 0))
    if baseEntropy - avgEntropy > maxInformationGain:
        maxInformationGain = baseEntropy - avgEntropy
        maxInformationGainIndex = i

print(maxInformationGainIndex)


uniqueValues = data00.iloc[:, maxInformationGainIndex].unique()

# now delete all rows only where in the column with the index maxInformationGainIndex is not uniqueValues[0] and keep the columns except the one with the index maxInformationGainIndex
data000 = data00[data00.iloc[:, maxInformationGainIndex] == uniqueValues[0]].drop(data00.columns[maxInformationGainIndex], axis=1)
print(data000)
maxInformationGain = 0
for i in range(data000.shape[1] - 1):
    avgEntropy = CalculateEntropyForColumn(data000, i)
    baseEntropy = Entropy(data000.iloc[:, -1].value_counts().get('Yes ', 0), data000.iloc[:, -1].value_counts().get('No ', 0))
    if baseEntropy - avgEntropy > maxInformationGain:
        maxInformationGain = baseEntropy - avgEntropy
        maxInformationGainIndex = i
    

print(maxInformationGainIndex)



uniqueValues = data000.iloc[:, maxInformationGainIndex].unique()

data0000 = data000[data000.iloc[:, maxInformationGainIndex] == uniqueValues[0]].drop(data000.columns[maxInformationGainIndex], axis=1)
print(data0000)
maxInformationGain = 0
for i in range(data0000.shape[1] - 1):
    avgEntropy = CalculateEntropyForColumn(data0000, i)
    baseEntropy = Entropy(data0000.iloc[:, -1].value_counts().get('Yes ', 0), data0000.iloc[:, -1].value_counts().get('No ', 0))
    if baseEntropy - avgEntropy > maxInformationGain:
        maxInformationGain = baseEntropy - avgEntropy
        maxInformationGainIndex = i


print(maxInformationGainIndex)
