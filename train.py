import os
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # Backend для Docker
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns

# Подавляем предупреждения TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(f"GPU OK: {tf.config.list_physical_devices('GPU')}")
print(f"TF version: {tf.__version__}")

# Создаем выходную директорию
os.makedirs('outputs', exist_ok=True)

dataset_dirs = ['Multi_alph']

def split_data(dataset_dir, classes, validation_split=0.2, test_split=0.1):
    train_dir = os.path.join(dataset_dir, 'train')
    validation_dir = os.path.join(dataset_dir, 'validation')
    test_dir = os.path.join(dataset_dir, 'test')
    
    # Очищаем старые директории если есть
    for dir_path in [train_dir, validation_dir, test_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    print(f"Разделение данных из {dataset_dir}")
    
    total_images = 0
    for class_name in classes:
        class_dir = os.path.join(dataset_dir, class_name)
        
        if not os.path.exists(class_dir):
            print(f"⚠️  Директория {class_dir} не найдена.")
            continue
        
        images = [f for f in os.listdir(class_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if not images:
            print(f"⚠️  Директория {class_dir} пуста.")
            continue
        
        total_images += len(images)
        
        # 70% train, 20% val, 10% test
        train_images, temp_images = train_test_split(images, test_size=0.3, random_state=42)
        val_images, test_images = train_test_split(temp_images, test_size=1/3, random_state=42)
        
        for split_dir, split_images in [(train_dir, train_images), 
                                      (validation_dir, val_images), 
                                      (test_dir, test_images)]:
            split_class_dir = os.path.join(split_dir, class_name)
            os.makedirs(split_class_dir, exist_ok=True)
            
            for img in split_images:
                shutil.copy(os.path.join(class_dir, img), os.path.join(split_class_dir, img))
    
    print(f"✅ Разделение завершено. Всего изображений: {total_images}")

# Основной цикл
for dataset_dir in dataset_dirs:
    print(f"\n{'='*60}")
    print(f"🚀 Обработка: {dataset_dir}")
    print(f"{'='*60}")
    
    classes = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8']
    
    # 1. Разделяем данные
    split_data(dataset_dir, classes)
    
    # Пути
    train_dir = os.path.join(dataset_dir, 'train')
    validation_dir = os.path.join(dataset_dir, 'validation')
    test_dir = os.path.join(dataset_dir, 'test')
    
    # Проверяем что директории созданы
    if not all(os.path.exists(d) for d in [train_dir, validation_dir, test_dir]):
        print("❌ Ошибка: не все директории созданы!")
        continue
    
    # Параметры
    batch_size = 32
    image_size = 128
    
    # Data Generators
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    print("📂 Создание генераторов данных...")
    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(image_size, image_size),
        batch_size=batch_size, class_mode='categorical', shuffle=True
    )
    
    validation_generator = val_datagen.flow_from_directory(
        validation_dir, target_size=(image_size, image_size),
        batch_size=batch_size, class_mode='categorical'
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir, target_size=(image_size, image_size),
        batch_size=batch_size, class_mode='categorical', shuffle=False
    )
    
    print(f"✅ Генераторы созданы:")
    print(f"   Train: {train_generator.samples} изображений")
    print(f"   Val:   {validation_generator.samples} изображений") 
    print(f"   Test:  {test_generator.samples} изображений")
    print(f"   Классы: {list(train_generator.class_indices.keys())}")
    
    # 2. Создаем модель БЕЗ summary()
    print("🔨 Создание модели...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(classes), activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("✅ Модель скомпилирована")
    
    # 3. Обучаем
    print("🎓 Начинаем обучение (10 эпох)...")
    history = model.fit(
        train_generator, 
        validation_data=validation_generator, 
        epochs=10,
        verbose=1
    )
    
    # 4. Сохраняем модель
    model_path = f"outputs/{dataset_dir}_model.h5"
    model.save(model_path)
    print(f"💾 Модель сохранена: {model_path}")
    
    # 5. Графики обучения
    print("📊 Создание графиков...")
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Train Acc', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Val Acc', linewidth=2)
    plt.title(f'{dataset_dir} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    plt.title(f'{dataset_dir} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['accuracy'][-1], label='Final Train Acc', marker='o')
    plt.plot(history.history['val_accuracy'][-1], label='Final Val Acc', marker='s')
    plt.title('Final Results')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"outputs/{dataset_dir}_training_plots.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 6. Тестирование
    print("🧪 Оценка на тестовых данных...")
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
    print(f"✅ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")
    
    # 7. Confusion Matrix
    print("📈 Confusion Matrix...")
    y_pred = model.predict(test_generator, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes
    
    cm = confusion_matrix(y_true, y_pred_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(f'{dataset_dir} - Confusion Matrix (Test Accuracy: {test_accuracy:.3f})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f"outputs/{dataset_dir}_confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 8. Отчет
    report = classification_report(y_true, y_pred_classes, target_names=classes)
    report_path = f"outputs/{dataset_dir}_classification_report.txt"
    with open(report_path, 'w') as f:
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n\n")
        f.write(report)
    
    print(f"📋 Отчет сохранен: {report_path}")
    print(f"🎉 Обработка {dataset_dir} завершена!")
    print(f"{'='*60}\n")

print("🏁 Все датасеты обработаны успешно!")
