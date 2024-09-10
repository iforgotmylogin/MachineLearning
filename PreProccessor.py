import random

class PreProccesor:
    def __init__(self):
        self.dataPath = None
        self.num_folds = None 

    def setDatabase(self, path):
        self.dataPath = path

    def importData(self):
        if not self.dataPath:
            raise ValueError("Data path is not set.")

        # Determine line count (number of samples)
        with open(self.dataPath, "r") as f:
            lineCount = sum(1 for _ in f)  # Count the number of lines

        # Initialize raw data
        rawData = [[0] * 11 for _ in range(lineCount)]

        # Read the file and populate rawData
        with open(self.dataPath, "r") as f:
            for i, line in enumerate(f):
                data = line.strip().split(",")  # Strip and split the line by comma
                for j in range(len(data)):
                    rawData[i][j] = int(data[j]) if data[j].isdigit() else 0  # Replace "?" with 0

        print("Data importation complete.")

        # Shuffle the raw data
        random.shuffle(rawData)

        # Initialize positive and negative arrays
        rawPos = []
        rawNeg = []

        # Split data into positive and negative categories
        for sample in rawData:
            if sample[10] == 2:
                rawNeg.append(sample)  # Negative class (2)
            elif sample[10] == 4:
                rawPos.append(sample)  # Positive class (4)

        posCount = len(rawPos)
        negCount = len(rawNeg)

        print("pos / neg splitting complete")
        return rawPos, rawNeg, posCount, negCount

    def createFolds(self, rawPos, rawNeg, posCount, negCount, num_folds=10):
        self.num_folds = num_folds
        # Initialize folds
        folds = [[] for _ in range(num_folds)]

        # Calculate the number of samples per fold
        pos_samples_per_fold = posCount // num_folds
        neg_samples_per_fold = negCount // num_folds

        # Populate the folds
        for fold_index in range(num_folds):
            start_pos = fold_index * pos_samples_per_fold
            end_pos = min(start_pos + pos_samples_per_fold, posCount)
            start_neg = fold_index * neg_samples_per_fold
            end_neg = min(start_neg + neg_samples_per_fold, negCount)

            # Add positive samples to the current fold
            for i in range(start_pos, end_pos):
                folds[fold_index].append(rawPos[i])

            # Add negative samples to the current fold
            for i in range(start_neg, end_neg):
                folds[fold_index].append(rawNeg[i])

            for fold in folds:  # Mix pos and neg elements together in fold
                random.shuffle(fold)

        # Return the folds
        return folds

    def generateNoise(self, folds):
        for fold in folds:
            for sample in fold:
                # Determine the number of features to shuffle (10% of the total features)
                num_features_to_shuffle = max(1, int(0.1 * len(sample)))  # Ensure at least one feature is shuffled
                
                # Select random indices to shuffle
                indices_to_shuffle = random.sample(range(len(sample)), num_features_to_shuffle)
                
                # Shuffle the selected features
                for index in indices_to_shuffle:
                    sublist = [fold[i][index] for i in range(len(fold))]  # Extract the feature column
                    random.shuffle(sublist)  # Shuffle the feature values
                    
                    # Reassign the shuffled values back to the fold
                    for i in range(len(fold)):
                        fold[i][index] = sublist[i]