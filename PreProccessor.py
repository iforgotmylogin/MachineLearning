import random

# Set the database path globally
def setDatabase(path):
    global dataPath
    dataPath = path

# Import and process the data
def importData():
    # Determine line count (number of samples)
    with open(dataPath, "r") as f:
        lineCount = sum(1 for _ in f)  # Count the number of lines

    # Initialize raw data
    rawData = [[0] * 11 for _ in range(lineCount)]

    # Read the file and populate rawData
    with open(dataPath, "r") as f:
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

# Perform k-fold splitting
def createFolds(rawPos, rawNeg, posCount, negCount, num_folds=10):
    # Initialize folds
    folds = [[] for _ in range(num_folds)]

    # Calculate the number of samples per fold
    pos_samples_per_fold = (posCount // num_folds) + 1
    neg_samples_per_fold = (negCount // num_folds) + 1

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

    # Return the folds instead of printing them
    return folds

# Main function
def main():
    setDatabase("data/breast-cancer-wisconsin.data")
    
    # Import the data
    rawPos, rawNeg, posCount, negCount = importData()

    # Create the folds
    folds = createFolds(rawPos, rawNeg, posCount, negCount, num_folds=10)

    # Now you can print or further process the folds as needed
    for i, fold in enumerate(folds):
        print(f"Fold {i + 1}:")
        for sample in fold:
            print(sample)
        print("-----------------------")

if __name__ == "__main__":
    main()
