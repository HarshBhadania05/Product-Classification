from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src import config

def evaluate_models(models, X_test, y_test, le):
    for name, model in models.items():
        print(f"\nEvaluating {name}:")
        y_pred = model.predict(X_test)

        # Decode labels for readability
        y_test_decoded = le.inverse_transform(y_test)
        y_pred_decoded = le.inverse_transform(y_pred)

        print("\nClassification Report:")
        print(classification_report(y_test_decoded, y_pred_decoded))

        # Confusion Matrix Plot
        cm = confusion_matrix(y_test_decoded, y_pred_decoded)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{name} - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(f"{config.FIGURE_DIR}/{name}_confusion_matrix.png")
        plt.close()