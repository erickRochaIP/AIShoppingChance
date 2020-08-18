import csv
import sys
import calendar

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4

abbr_to_num = {name: num for num, name in enumerate(calendar.month_abbr) if num}


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    evidenceLabels = ([], [])
    with open(filename, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            evidenceLabels[1].append(1 if row["Revenue"] == "TRUE" else 0)
            evidenceRow = []
            evidenceRow.append(int(row["Administrative"]))
            evidenceRow.append(float(row["Administrative_Duration"]))
            evidenceRow.append(int(row["Informational"]))
            evidenceRow.append(float(row["Informational_Duration"]))
            evidenceRow.append(int(row["ProductRelated"]))
            evidenceRow.append(float(row["ProductRelated_Duration"]))
            evidenceRow.append(float(row["BounceRates"]))
            evidenceRow.append(float(row["ExitRates"]))
            evidenceRow.append(float(row["PageValues"]))
            evidenceRow.append(float(row["SpecialDay"]))
            evidenceRow.append((abbr_to_num[row["Month"][0:3]])-1)
            evidenceRow.append(int(row["OperatingSystems"]))
            evidenceRow.append(int(row["Browser"]))
            evidenceRow.append(int(row["Region"]))
            evidenceRow.append(int(row["TrafficType"]))
            evidenceRow.append(1 if row["VisitorType"] == "Returning_Visitor" else 0)
            evidenceRow.append(1 if row["Weekend"] == "TRUE" else 0)
            evidenceLabels[0].append(evidenceRow)
    return evidenceLabels


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """

    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model

def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """

    sensitivityPredicted = 0
    sensitivityTotal = 0
    specificityPredicted = 0
    specificityTotal = 0

    for label, predict in zip(labels, predictions):
        if label == 1:
            sensitivityTotal += 1
            if label == predict:
                sensitivityPredicted += 1
        else:
            specificityTotal += 1
            if label == predict:
                specificityPredicted += 1

    return (sensitivityPredicted/sensitivityTotal, specificityPredicted/specificityTotal)


if __name__ == "__main__":
    main()
