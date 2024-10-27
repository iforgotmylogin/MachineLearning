import random
from collections import defaultdict


class PreProcessor:
    def __init__(self):
        self.dataPath = None
        self.num_folds = 10  # Default to 10 folds

    def setDatabase(self, path):
        self.dataPath = path

    def importData(self):
        if not self.dataPath:
            raise ValueError("Data path is not set.")
        
        rawData = []
        
        with open(self.dataPath, "r") as f:
            for line in f:
                data = line.strip().split(",")
                proline = []
                for val in data:
                    if val == '?':
                        proline.append(-1)  # Handle missing value
                    elif val == 'n':
                        proline.append(0)   # Handle 'no'
                    elif val == 'y':
                        proline.append(1)   # Handle 'yes'
                    else:
                        try:
                            proline.append(float(val))  # Try to convert to float
                        except ValueError:
                           next
                            
                rawData.append(proline)

        print("Data importation complete.")
        #print(rawData)
        return rawData

    def cleanData(self, rawData):
        cleanedData = [sample for sample in rawData if None not in sample and len(sample) > 1]
        print(f"Cleaned data: {len(cleanedData)} samples remain.")
        #print (cleanedData)
        return cleanedData

    def stratifiedSplit(self, cleanedData, label_index):
        classDict = defaultdict(list)
        for sample in cleanedData:
            # Append samples to the dictionary based on their label
            classDict[sample[label_index]].append(sample)

        # Count occurrences for each label and ensure default values
        posCount = len(classDict.get(1, []))  # Count for positive class
        negCount = len(classDict.get(0, []))  # Count for negative class
        neutralCount = len(classDict.get(10, []))  # Count for neutral class
        otherCount = len(classDict.get(11, []))  # Count for other class

        print(f"Class counts: pos={posCount}, neg={negCount}, neutral={neutralCount}, other={otherCount}")
        
        return classDict, posCount, negCount, neutralCount, otherCount
    
    def regSplit(self, cleanedData, label_index):
        classDict = defaultdict(list)
        
        # Sort cleanedData in ascending order based on the element at label_index
        cleanedData.sort(key=lambda x: x[label_index])
        
        # Create 10 groups by taking every 10th element
        for i in range(len(cleanedData)):
            group_index = i % 10  # Calculate group index (0-9)
            classDict[group_index].append(cleanedData[i])  # Add element to the corresponding group
    
        return classDict
        
    def createFolds(self, classDict, num_folds=10):
        self.num_folds = num_folds
        folds = [[] for _ in range(num_folds)]

        for class_samples in classDict.values():
            random.shuffle(class_samples)
            fold_size = len(class_samples) // num_folds
            print(f"Class samples count: {len(class_samples)}, Fold size: {fold_size}")

            for fold_index in range(num_folds):
                start = fold_index * fold_size
                end = start + fold_size if fold_index < num_folds - 1 else len(class_samples)
                folds[fold_index].extend(class_samples[start:end])

        # Log the contents of each fold
        for i, fold in enumerate(folds):
            print(f"Fold {i} contains {len(fold)} samples")

        print("Folds created with stratified data.")
        return folds

    def generateNoise(self, folds):
        for fold in folds:
            for sample in fold:
                num_features_to_shuffle = max(1, int(0.1 * len(sample)))
                indices_to_shuffle = random.sample(range(len(sample)), num_features_to_shuffle)
                for index in indices_to_shuffle:
                    sublist = [fold[i][index] for i in range(len(fold))]
                    random.shuffle(sublist)
                    for i in range(len(fold)):
                        fold[i][index] = sublist[i]
        print("Noise generation complete.")
    
    def reset(self):
        self.rawData = []
        self.cleanedData = []
        self.classDict = defaultdict(list)
        self.folds = []
