"""
Split dataset_final.json into new and original datasets based on 'source' field
"""
import json
from pathlib import Path


def split_dataset(input_file='dataset/data/dataset_final.json',
                  new_file='dataset/data/dataset_new.json',
                  original_file='dataset/data/dataset_original.json'):
    """
    Split dataset into new and original based on source field
    
    Args:
        input_file: Path to input JSON file
        new_file: Path to output file for new advertisements
        original_file: Path to output file for original advertisements
    """
    print(f"📂 Загружаю {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ Загружено {len(data):,} объектов\n")
    
    # Split by source
    new_ads = []
    original_ads = []
    
    for obj in data:
        if obj.get('source') == 'new':
            new_ads.append(obj)
        elif obj.get('source') == 'original':
            original_ads.append(obj)
    
    # Save to files
    print(f"💾 Сохраняю новые объявления ({len(new_ads):,})...")
    with open(new_file, 'w', encoding='utf-8') as f:
        json.dump(new_ads, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Сохраняю старые объявления ({len(original_ads):,})...")
    with open(original_file, 'w', encoding='utf-8') as f:
        json.dump(original_ads, f, ensure_ascii=False, indent=2)
    
    # Statistics
    print(f"\n📊 СТАТИСТИКА РАЗДЕЛЕНИЯ:")
    print(f"  Новые (source=new):       {len(new_ads):,} ({len(new_ads)/len(data)*100:.1f}%)")
    print(f"  Старые (source=original): {len(original_ads):,} ({len(original_ads)/len(data)*100:.1f}%)")
    print(f"  Всего:                    {len(new_ads) + len(original_ads):,}")
    
    print(f"\n✅ ГОТОВО!")
    print(f"  📄 Новые:  {new_file}")
    print(f"  📄 Старые: {original_file}")
    
    return len(new_ads), len(original_ads)


if __name__ == '__main__':
    split_dataset()
