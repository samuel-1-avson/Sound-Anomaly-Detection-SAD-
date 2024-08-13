import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import librosa
import h5py
import zipfile
from sklearn.metrics import mean_squared_error

def autoencoder_model():
    inputs = Input(shape=(2048,))
    
    x = Dense(512, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(2048, activation='linear')(x)
    
    model = Model(inputs, outputs)
    return model

def extract_mel_spectrogram(file_path, frame_size=1024, hop_length=512):
    y, sr = librosa.load(file_path)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=frame_size, hop_length=hop_length)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    target_length = 2048
    if S_dB.size < target_length:
        pad_width = target_length - S_dB.size
        S_dB = np.pad(S_dB.flatten(), (0, pad_width), mode='constant')
    else:
        S_dB = S_dB.flatten()[:target_length]
    
    return S_dB.reshape(1, -1)

def save_preprocessed_data(file_paths, output_file, frame_size=1024, hop_length=512):
    with h5py.File(output_file, 'w') as hf:
        dataset = hf.create_dataset('mel_spectrograms', shape=(len(file_paths), 2048), dtype='float32')
        
        for i, file_path in enumerate(file_paths):
            try:
                mel_spectrogram = extract_mel_spectrogram(file_path, frame_size, hop_length)
                dataset[i] = mel_spectrogram.squeeze()
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

def load_data_info(hdf5_file):
    with h5py.File(hdf5_file, 'r') as hf:
        shape = hf['mel_spectrograms'].shape
        dtype = hf['mel_spectrograms'].dtype
    return shape, dtype

def data_generator(hdf5_file, batch_size):
    while True:
        with h5py.File(hdf5_file, 'r') as hf:
            dataset = hf['mel_spectrograms']
            num_samples = dataset.shape[0]
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_indices = sorted(indices[start:end])
                batch_data = dataset[batch_indices]
                yield batch_data, batch_data

def evaluate_model(model, data_generator, steps):
    reconstruction_errors = []
    for _ in range(steps):
        batch_data, _ = next(data_generator)
        reconstructed = model.predict(batch_data)
        mse = mean_squared_error(batch_data, reconstructed)
        reconstruction_errors.append(mse)
    mean_error = np.mean(reconstruction_errors)
    print(f'Mean Reconstruction Error: {mean_error}')
    return mean_error

def train(training_files, model_dir, frame, lr, batch_size, epochs, gpu_count):
    hdf5_file = 'preprocessed_data.h5'
    save_preprocessed_data(training_files, hdf5_file, frame_size=frame)

    data_shape, data_dtype = load_data_info(hdf5_file)

    if data_shape[0] == 0:
        raise ValueError("Preprocessed train data is empty. Check if the input files are correct and not empty.")

    model = autoencoder_model()

    if gpu_count > 1:
        model = tf.keras.utils.multi_gpu_model(model, gpus=gpu_count)

    model.compile(
        loss='mean_squared_error',
        optimizer=Adam(learning_rate=lr),
        metrics=['mse']
    )

    train_gen = data_generator(hdf5_file, batch_size)
    val_gen = data_generator(hdf5_file, batch_size)

    steps_per_epoch = data_shape[0] // batch_size
    validation_steps = steps_per_epoch // 10

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)
    ]

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        epochs=epochs,
        verbose=2,
        callbacks=callbacks
    )

    os.makedirs(os.path.join(model_dir, 'autoencoder2/model/1'), exist_ok=True)
    model.save(os.path.join(model_dir, 'autoencoder2/model/1/model.h5'))

    print("Evaluating model on training data...")
    evaluate_model(model, train_gen, steps_per_epoch)
    print("Evaluating model on validation data...")
    evaluate_model(model, val_gen, validation_steps)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--frame', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gpu_count', type=int, default=1)
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--training_files', type=str, nargs='+', default=["C:\\Users\\Bridgette Akua Anese\\Downloads\\-6_dB_fan.zip"])
    return parser.parse_args()

if __name__ == '__main__':
    tf.random.set_seed(42)

    args = parse_arguments()
    epochs = args.epochs
    frame = args.frame
    lr = args.learning_rate
    batch_size = args.batch_size
    gpu_count = args.gpu_count
    model_dir = args.model_dir

    zip_path = args.training_files[0]
    extract_path = "C:\\Users\\Bridgette Akua Anese\\Downloads\\extracted_audio\\fan\\id_06\\normal"
    os.makedirs(extract_path, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    extracted_items = os.listdir(extract_path)
    training_files = []
    for item in extracted_items:
        item_path = os.path.join(extract_path, item)
        if os.path.isdir(item_path):
            for root, dirs, files in os.walk(item_path):
                for file in files:
                    if file.endswith('.wav'):
                        training_files.append(os.path.join(root, file))
        elif item_path.endswith('.wav'):
            training_files.append(item_path)

    if not training_files:
        print("No training files found. Please check the extraction path and file extensions.")
    
    train(training_files, model_dir, frame, lr, batch_size, epochs, gpu_count)
