import csv
import glob
import os
import re


def fix_labels_in_file(input_path, output_path):
    """Обрабатывает CSV: если result < 50, устанавливает label=0 и создаёт новый файл."""
    total_rows = 0
    label_one_before = 0
    modifications = 0
    new_label_one = 0

    with open(input_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)

        result_idx = headers.index('result')
        label_idx = headers.index('label')

        rows = list(reader)

    total_rows = len(rows)
    label_one_before = sum(1 for row in rows if row[label_idx] == '1')

    modified_rows = []
    for row in rows:
        result_val = float(row[result_idx])
        if result_val < 20:
            if row[label_idx] != '0':
                modifications += 1
            row[label_idx] = '0'
        modified_rows.append(row)

    new_label_one = sum(1 for row in modified_rows if row[label_idx] == '1')

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(modified_rows)

    return {
        'input_file': os.path.basename(input_path),
        'output_file': os.path.basename(output_path),
        'total_rows': total_rows,
        'label_one_before': label_one_before,
        'label_one_after': new_label_one,
        'modifications': modifications
    }


def main():
    source_dir = r'C:\Users\Aser\AppData\Roaming\MetaQuotes\Terminal\Common\Files'
    target_dir = r'H:\Мой диск\RNN\data'
    os.makedirs(target_dir, exist_ok=True)

    # Собираем файлы, которые НАЧИНАЮТСЯ на "data"
    all_files = os.listdir(source_dir)
    csv_files = [f for f in all_files
                 if f.endswith('.csv') and re.match(r'^data', f, re.IGNORECASE)]

    if not csv_files:
        print('CSV файлы не найдены в папке Common/Files/')
        return

    print('=' * 80)
    print('ОТЧЕТ ПО ИСПРАВЛЕНИЮ LABEL')
    print('=' * 80)

    total_modifications = 0
    total_label_one_before = 0
    total_label_one_after = 0
    total_rows = 0

    for filename in sorted(csv_files):
        input_path = os.path.join(source_dir, filename)
        output_filename = filename.replace('.csv', '_fix.csv')
        output_path = os.path.join(target_dir, output_filename)

        report = fix_labels_in_file(input_path, output_path)

        print(f"\nФайл: {report['input_file']}")
        print(f"  Создан:   {report['output_file']}")
        print(f"  Строк всего:          {report['total_rows']}")
        print(f"  Label=1 (было):      {report['label_one_before']}")
        print(f"  Label=1 (стало):     {report['label_one_after']}")
        print(f"  Правок:              {report['modifications']}")

        total_modifications += report['modifications']
        total_label_one_before += report['label_one_before']
        total_label_one_after += report['label_one_after']
        total_rows += report['total_rows']

    print('\n' + '=' * 80)
    print('ИТОГИ:')
    print(f"  Всего файлов:        {len(csv_files)}")
    print(f"  Всего строк:         {total_rows}")
    print(f"  Label=1 (было):      {total_label_one_before}")
    print(f"  Label=1 (стало):     {total_label_one_after}")
    print(f"  Всего правок:        {total_modifications}")
    print('=' * 80)


if __name__ == '__main__':
    main()
