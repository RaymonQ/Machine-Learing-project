import os
import subprocess


# add your path to the project here:
path = '/Users/yannickschnider/PycharmProjects/COMP9417-Group-Assignment/'
path_folder = '04_ModelTraining'
# putting all the files to execute into a list
files = [file for file in os.listdir(path + path_folder) if file.endswith(".py")]
# remove this file to avoid a infinity loop
files.remove('run_all.py')
files.remove('SEQ.py')
# files.remove('MNB.py')
# files.remove('RF.py')
# files.remove('GBM.py')
# files.remove('KNN.py')

# run all the scripts one after another
for script in files:
    # os.system(path + path_folder + '/' + script)
    subprocess.check_call(["python3.6", path + path_folder + '/' + script])
    print('Executed ' + script + '\n')


