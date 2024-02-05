import pandas as pd
import math

def Entropy(yes, no):
    if yes == 0 or no == 0:
        return 0
    else:
        total = yes + no
        return -((yes/total) * math.log2(yes/total) + (no/total) * math.log2(no/total))

def CalculateEntropyForColumn(data, columnIndex):
    uniqueValues = data.iloc[:, columnIndex].unique()
    totalLength = len(data)
    averageEntropy = 0

    for value in uniqueValues:
        filteredData = data[data.iloc[:, columnIndex] == value].iloc[:, -1]
        countYes = filteredData.value_counts().get('Yes ', 0)
        countNo = filteredData.value_counts().get('No ', 0)
        entropyValue = Entropy(countYes, countNo)
        averageEntropy += (len(filteredData) / totalLength) * entropyValue

    return averageEntropy

def BuildDecisionTree(data, ignored_columns=[]):
    if len(data.columns) == len(ignored_columns) + 1:  # Only target column left
        target_entropy = Entropy(data.iloc[:, -1].value_counts().get('Yes ', 0), data.iloc[:, -1].value_counts().get('No ', 0))
        return {'H(S|A)': 0, 'H(S)': target_entropy, 'Result': data.iloc[:, -1].mode()[0]}

    baseEntropy = Entropy(data.iloc[:, -1].value_counts().get('Yes ', 0), data.iloc[:, -1].value_counts().get('No ', 0))
    maxInformationGain = 0
    maxInformationGainIndex = -1

    for i in range(data.shape[1] - 1):
        if i in ignored_columns:
            continue
        avgEntropy = CalculateEntropyForColumn(data, i)
        infoGain = baseEntropy - avgEntropy
        if infoGain > maxInformationGain:
            maxInformationGain = infoGain
            maxInformationGainIndex = i

    if maxInformationGain == 0:
        target_entropy = Entropy(data.iloc[:, -1].value_counts().get('Yes ', 0), data.iloc[:, -1].value_counts().get('No ', 0))
        return {'H(S|A)': 0, 'H(S)': target_entropy, 'Result': data.iloc[:, -1].mode()[0]}

    # Recursively build the tree
    tree = {}
    uniqueValues = data.iloc[:, maxInformationGainIndex].unique()
    ignored_columns.append(maxInformationGainIndex)

    column_names = data.columns

    for value in uniqueValues:
        subset = data[data.iloc[:, maxInformationGainIndex] == value].drop(data.columns[maxInformationGainIndex], axis=1)
        subtree = BuildDecisionTree(subset, ignored_columns.copy())
        tree[(column_names[maxInformationGainIndex], value)] = subtree


    H_S_A = maxInformationGain

    return {'H(S|A)': H_S_A, 'H(S)': baseEntropy, 'Subtree': tree}

# Read the data from the csv file
data = pd.read_csv('id3_data.csv')

# Build the decision tree
decision_tree = BuildDecisionTree(data)
print(decision_tree)

