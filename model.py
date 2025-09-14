from tensorflow.keras import layers, models

def create_model(input_shape=(150,150,3)):
    model = models.Sequential([
        layers.Conv2D(32,(3,3),activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64,(3,3),activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(128,(3,3),activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(32,(3,3),activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        layers.Dense(512,activation='relu'),
        layers.Dense(1,activation='sigmoid')
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model
