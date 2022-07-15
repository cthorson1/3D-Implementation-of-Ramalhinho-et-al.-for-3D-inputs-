from model import TripletLoss, Network

#Config
checkpoint = "./model_checkpoint"
csv_file = "./training.log"
epochs = 15
batch = 24
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1, mode='max'),
    tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor='val_acc', verbose=1, save_best_only=True),
    tf.keras.callbacks.CSVLogger(csv_file, separator=",", append=False)
]
input_size=(67,1,64,128,128)
network = Network(shape = input_size)
network.summary()
