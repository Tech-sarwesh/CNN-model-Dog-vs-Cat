import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from data_loader import load_data
from sklearn.metrics import confusion_matrix

test_dir = 'data/test'
# Using ImageDataGenerator for test data
from tensorflow.keras.preprocessing.image import ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary',
    shuffle=False
)

model = load_model('cnn_cats_vs_dogs.h5')

# Evaluate
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")

# Confusion Matrix
probabilities = model.predict(test_generator)
predicted_classes = (probabilities > 0.5).astype(int).squeeze()
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

