import shutil

source_file = 'ch1.csv'

for i in range(2, 21):
    target_file = f'ch{i}.csv'
    shutil.copyfile(source_file, target_file)

print("Complete")