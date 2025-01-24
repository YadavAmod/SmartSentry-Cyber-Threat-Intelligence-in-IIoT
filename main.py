def main():
    while True:
        print("\nSelect a Model to Run:")
        print("1: Random Forest")
        print("2: Decision Tree")
        print("3: Extra Tree Classifier")
        print("4: Support Vector Machine")
        print("5: k-Nearest Neighbor")
        print("6: Deep Neural Network")

        choice = input("Enter your choice (1-6): ")

        if choice == '1':
            from models import random_forest as rf
            rf.random_forest_model()
        elif choice == '2':
            from models import decision_tree as dt
            dt.decision_tree_model()
        elif choice == '3':
            from models import extra_tree_classifier as etc
            etc.extra_tree_classifier_model()
        elif choice == '4':
            from models import svm
            svm.svm_model()
        elif choice == '5':
            from models import knn
            knn.knn_model()
        elif choice == '6':
            from models import deep_neural_network as dnn
            dnn.deep_neural_network_model()
        else:
            print("Invalid choice! Please select a number between 1 and 6.")

        cont = input("\nDo you want to run another model? (yes/no): ")
        if cont.lower() != 'yes':
            break

if __name__ == "__main__":
    main()