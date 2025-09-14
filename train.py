from data_loader import load_data
from model import create_model

base_dir = 'data/train'
train_generator, validation_generator = load_data(base_dir)

model = create_model()
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

model.save('cnn_cats_vs_dogs.h5')
