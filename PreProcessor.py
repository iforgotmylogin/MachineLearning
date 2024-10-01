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
                proline = [float(val) if val != '' else None for val in data]
                rawData.append(proline)

        print("Data importation complete.")
        return rawData

    def cleanData(self, rawData):
        cleanedData = [sample for sample in rawData if None not in sample]
        print(f"Cleaned data: {len(cleanedData)} samples remain.")
        return cleanedData

    def stratifiedSplit(self, cleanedData, label_index):
        classDict = defaultdict(list)
        for sample in cleanedData:
            classDict[sample[label_index]].append(sample)

        posCount = len(classDict[2])
        negCount = len(classDict[4])
        neutralCount = len(classDict[6])
        otherCount = len(classDict[8])

        print(f"Class counts: pos={posCount}, neg={negCount}, neutral={neutralCount}, other={otherCount}")
        return classDict, posCount, negCount, neutralCount, otherCount

    def createFolds(self, classDict, num_folds=10):
        self.num_folds = num_folds
        folds = [[] for _ in range(num_folds)]

        for class_samples in classDict.values():
            random.shuffle(class_samples)
            fold_size = len(class_samples) // num_folds

            for fold_index in range(num_folds):
                start = fold_index * fold_size
                end = start + fold_size if fold_index < num_folds - 1 else len(class_samples)
                folds[fold_index].extend(class_samples[start:end])

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
