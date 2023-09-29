import pandas as pd
import os
import subprocess
import sys
import venv
import colorsys

def create_folders_if_not_exist(folder_paths):
    """Function that creates folders if they don't exist
    Parameters
    ----------
    folder_paths : list
        List with the paths of the folders to create
    """
    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


 
def HSVToRGB(h, s, v): 
    (r, g, b) = colorsys.hsv_to_rgb(h, s, v) 
    return (r,g,b)
 
def getDistinctColors(n): 
    huePartition = 1.0 / (n + 1) 
    return (HSVToRGB(huePartition * value, 1.0, 1.0) for value in range(0, n)) 

class CreateVenv:
    def __init__(self, venv_folder = 'venv'):
        self.venv_folder = venv_folder
        self.venv_dir = os.path.join(os.getcwd(), self.venv_folder)
        self.activate_dir = os.path.join(self.venv_dir, "Scripts", "activate.bat")
        self.pip_exe = os.path.join(self.venv_dir, "Scripts", "pip3.exe")
        self.python_exe = os.path.join(self.venv_dir, "Scripts", "python.exe")

        if sys.platform == "win32":
            self.create_venv()
            self.activate_venv()

    def create_venv(self):
        venv.create(self.venv_dir, with_pip=True)
    
    def activate_venv(self):
        subprocess.run([self.activate_dir])

    def install_packages(self,package_list):
        for package in package_list:
            subprocess.run([self.pip_exe, "install", package])
    
    def run_file(self, file_path):
        subprocess.run([self.python_exe,  file_path])

class ReadData:

    def __init__(self):
        pass


    def list_relative_paths_in_folder(self):
        paths = []
        for root, _, files in os.walk(self.DEFAULT_PATH):
            for file in files:
                paths.append(os.path.join(root, file))
        return paths

    def read_file(self,file_path):
        file_extension = file_path.split('.')[-1].lower()
        if file_extension == 'csv':
            df = pd.read_csv(file_path)
        elif file_extension == 'xlsx':
            df = pd.read_excel(file_path, sheet_name=0)
        elif file_extension == 'xls':
            df = pd.read_excel(file_path, sheet_name=0)
        elif file_extension == 'txt':
            df = pd.read_csv(file_path, sep=',')
        elif file_extension == 'json':
            df = pd.read_json(file_path)
        return df

    def get_data_dfs(self, DEFAULT_PATH):
        self.DEFAULT_PATH = DEFAULT_PATH
        self.ACCEPTABLE_FILE_EXTENSIONS = ('.csv', '.xlsx', '.xls', '.txt', '.json')

        file_paths = self.list_relative_paths_in_folder()
        filtered_paths = [path for path in file_paths if path.endswith(self.ACCEPTABLE_FILE_EXTENSIONS)]
        dfs = [self.read_file(path) for path in filtered_paths]
        return dfs
    
    def get_data(self, file_path):
        return self.read_file(file_path)


def test():
    c = CreateVenv("venv")
    c.install_packages(['pandas==2.1.0'])
    c.run_file('main.py')

    reader = ReadData('Data')
    data_dfs = reader.get_data_dfs()
    print(data_dfs[0])


        