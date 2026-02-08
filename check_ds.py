#!/usr/bin/env python3
import os
from PIL import Image
from pathlib import Path
from collections import defaultdict
import argparse

def safe_image_check(img_path):
    """МЯГКАЯ проверка - как в Keras ImageDataGenerator"""
    if img_path.stat().st_size == 0:
        return False, "пустой файл"
    
    try:
        # ✅ ТОЛЬКО Image.open() - как делает Keras!
        with Image.open(img_path) as img:
            # НЕ делаем img.verify() + img.load() - это ломает нормальные файлы
            img = img.convert('RGB')  # Приводим к RGB (Keras ожидает 3 канала)
            w, h = img.size
            if w < 10 or h < 10:  # Слишком маленькие
                return False, f"слишком маленький {w}x{h}"
        return True, "OK"
    except Exception as e:
        return False, str(e)

def analyze_dataset(dataset_dir="Multi_alph", classes=None):
    if classes is None:
        classes = ['class1', 'class2', 'class3', 'class4', 'class5', 
                   'class6', 'class7', 'class8']
    
    stats = defaultdict(lambda: {'total': 0, 'bad': 0, 'ok': 0})
    bad_files = []
    
    print(f"🔍 МЯГКИЙ анализ (как Keras): {dataset_dir}")
    print("=" * 80)
    
    for class_name in classes:
        class_dir = Path(dataset_dir) / class_name
        if not class_dir.exists():
            print(f"⚠️  {class_name} не найден")
            continue
        
        # Все изображения независимо от расширения
        images = list(class_dir.iterdir())
        images = [p for p in images if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}]
        
        stats[class_name]['total'] = len(images)
        print(f"\n📁 {class_name}: {len(images)} файлов")
        
        for img_path in images:
            is_ok, reason = safe_image_check(img_path)
            if is_ok:
                stats[class_name]['ok'] += 1
            else:
                stats[class_name]['bad'] += 1
                bad_files.append(f"{img_path.name}: {reason}")
                print(f"❌ {img_path.name} ({img_path.stat().st_size}б): {reason}")
    
    # СТАТИСТИКА
    total_files = sum(s['total'] for s in stats.values())
    total_bad = sum(s['bad'] for s in stats.values())
    
    print("\n" + "=" * 80)
    print("📊 РЕЗУЛЬТАТЫ")
    print(f"Всего файлов: {total_files}")
    print(f"Проблемных:   {total_bad} ({total_bad/total_files*100:.1f}%)")
    
    if total_bad > 0:
        with open("real_bad_files.txt", "w", encoding='utf-8') as f:
            f.write(f"ПРОБЛЕМНЫЕ ФАЙЛЫ ({total_bad}):\n\n")
            for bad in bad_files:
                f.write(bad + "\n")
        print("💾 Список: real_bad_files.txt")
        
        if total_bad < total_files * 0.05:  # <5%
            print("✅ ОК! Можно удалить проблемные и обучать")
        else:
            print("⚠️  Слишком много проблем (>5%) - проверьте датасет")
    else:
        print("✅ ИДЕАЛЬНО! Все файлы нормальные")
    
    return bad_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--dataset", default="Multi_alph")
    args = parser.parse_args()
    
    bad_files = analyze_dataset(args.dataset)
