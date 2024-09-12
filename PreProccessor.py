import random

class PreProcessor:
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
        rawData = []

        # Read the file and populate rawData
        with open(self.dataPath, "r") as f:
            for i, line in enumerate(f):
                data = line.strip().split(",")  # Strip and split the line by comma
                proline = []
               
                for val in data:
                    if val == 'D1':
                        proline.append(2)
                    elif val == 'D2':
                        proline.append(4)
                    elif val == 'D3':
                        proline.append(6)
                    elif val == 'D4':
                        proline.append(8)
                    elif val == '':
                        next
                        #do nothing to remove blanks
                    else:
                        proline.append(float(val))  # For non-label values, append the original value
                rawData.append(proline)

        print("Data importation complete.")

        # Shuffle the raw data
        random.shuffle(rawData)

        # Initialize class arrays
        rawPos = []
        rawNeg = []
        rawNeutral = []  # New class list
        rawOther = []  # New class list
        print(rawData)  
        # Split data into classes
        for sample in rawData:
            print(sample)
            if len(sample) <= 4:
             print("Faulty sample:", sample)
   
            elif sample[35] == 2:
                rawNeg.append(sample)  # Negative class
            elif sample[35] == 4:
                rawPos.append(sample)  # Positive class
            elif sample[35] == 6:
                rawNeutral.append(sample)  # Neutral class
            elif sample[35] == 8:
                rawOther.append(sample)  # Other class

        posCount = len(rawPos)
        negCount = len(rawNeg)
        neutralCount = len(rawNeutral)
        otherCount = len(rawOther)

        print("Class splitting complete")
        return rawPos, rawNeg, rawNeutral, rawOther, posCount, negCount, neutralCount , otherCount

    def createFolds(self, rawPos, rawNeg, rawNeutral, rawOther, posCount, negCount, neutralCount , otherCount, num_folds=10):
        self.num_folds = num_folds
        # Initialize folds
        folds = [[] for _ in range(num_folds)]

        # Calculate the number of samples per fold
        pos_samples_per_fold = posCount // num_folds
        neg_samples_per_fold = negCount // num_folds
        neutral_samples_per_fold = neutralCount // num_folds  # New class fold calculation
        other_samples_per_fold = otherCount // num_folds
        # Populate the folds
        for fold_index in range(num_folds):
            start_pos = fold_index * pos_samples_per_fold
            end_pos = min(start_pos + pos_samples_per_fold, posCount)
            start_neg = fold_index * neg_samples_per_fold
            end_neg = min(start_neg + neg_samples_per_fold, negCount)
            start_neutral = fold_index * neutral_samples_per_fold
            end_neutral = min(start_neutral + neutral_samples_per_fold, neutralCount)

            start_other = fold_index * other_samples_per_fold
            end_other = min(start_other + other_samples_per_fold, otherCount)

            # Add samples to the current fold
            for i in range(start_pos, end_pos):
                folds[fold_index].append(rawPos[i])

            for i in range(start_neg, end_neg):
                folds[fold_index].append(rawNeg[i])

            for i in range(start_neutral, end_neutral):
                folds[fold_index].append(rawNeutral[i])
            
            for i in range(start_other, end_other):
                folds[fold_index].append(rawOther[i])

            for fold in folds:  # Mix samples together in fold
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
                        fold[i][index] = sublist[i]  # Fix typo: 'I' -> 'i'
