import pandas as pd
import os
import subprocess
import sys
import venv
import colorsys


def clean_dataframe(df, columns_econde = None):
    """Function that cleans a dataframe by dropping rows with NaN values and encoding categorical columns
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to clean
    columns_econde : list, optional
        List with the names of the columns to encode, by default None
    Returns
    -------
    pandas.DataFrame
        Cleaned dataframe
    """
    df = df.dropna()
    if columns_econde is not None and len(columns_econde) != 0:
        df = pd.get_dummies(df, columns=columns_econde, dtype=float)
    return df

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
    """Function that converts a color from HSV to RGB
    Parameters
    ----------
    h : float
        Hue
    s : float
        Saturation
    v : float
        Value
    Returns
    -------
    tuple
        RGB color
    """
    (r, g, b) = colorsys.hsv_to_rgb(h, s, v) 
    return (r,g,b)
 
def getDistinctColors(n):
    """Function that returns a list of n distinct rgb colors
    Parameters
    ----------
    n : int
        Number of colors to return
    Returns
    -------
    list
        List of n distinct rgb colors
    """
    huePartition = 1.0 / (n + 1) 
    return (HSVToRGB(huePartition * value, 1.0, 1.0) for value in range(0, n)) 

class CreateVenv:
    """Class that represents a virtual environment it can create and activate the environment
    as well as install required packages and run scripts inside the environment
    Parameters
    ----------
    venv_folder : str, optional
        Name of the folder where the virtual environment will be created, by default 'venv'
    
    """
    def __init__(self, venv_folder = 'venv'):
        self.venv_folder = venv_folder
        self.venv_dir = os.path.join(os.getcwd(), self.venv_folder)
        print(self.venv_dir)
        self.activate_dir = os.path.join(self.venv_dir, "Scripts", "activate.bat")
        self.pip_exe = os.path.join(self.venv_dir, "Scripts", "pip3.exe")
        self.python_exe = os.path.join(self.venv_dir, "Scripts", "python.exe")
        print(self.python_exe)
        self.create_venv()
        self.activate_venv()

    def create_venv(self):
        """Creates the virtual environment"""
        venv.create(self.venv_dir, with_pip=True)
    
    def activate_venv(self):
        """Activates the virtual environment"""
        subprocess.run([self.activate_dir])

    def install_packages(self,package_list):
        """Installs the given packages in the virtual environment
        Parameters
        ----------
        package_list : list
            List of packages to install
            """
        for package in package_list:
            subprocess.run([self.pip_exe, "install", package])

    def install_requirements(self, requirements_file):
        """Installs the packages in the given requirements file
        Parameters
        ----------
        requirements_file : str
            Path of the requirements file
        """
        subprocess.run([self.pip_exe, "install", "-r", requirements_file])
    
    def run_file(self, file_path, args = None):
        """Runs the given file inside the virtual environment
        Parameters
        ----------
        file_path : str
            Path of the file to run
        args : list, optional
            string of arguments to pass to the file, by default None
        """
        if args is None:
            subprocess.run([self.python_exe,  file_path])
        else:
            print(args)
            subprocess.run([self.python_exe,  file_path, *args])

class ReadData:
    """Class that represents a reader of data.

    It can read one file given its path and return a pandas dataframe or
    it can scan an entire folder and read all the files and return a list of dataframes"""
    def __init__(self):
        pass

    def read_file(self,file_path):
        """Reads the file in the given path and returns a pandas dataframe.
        Accepted file extensions are: csv, xlsx, xls, txt, json
        Parameters
        ----------
        file_path : str
            Path of the file to read
        Returns
        -------
        pandas.DataFrame
            Dataframe with the data in the file
        """
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
        """Returns a list of dataframes with the data in the files in the given folder
        Parameters
        ----------
        DEFAULT_PATH : str
            Path of the folder to scan
        Returns
        -------
        list
            List of dataframes with the data in the files in the given folder
        """
        self.DEFAULT_PATH = DEFAULT_PATH
        self.ACCEPTABLE_FILE_EXTENSIONS = ('.csv', '.xlsx', '.xls', '.txt', '.json')

        file_paths = self.list_paths_in_folder()
        filtered_paths = [path for path in file_paths if path.endswith(self.ACCEPTABLE_FILE_EXTENSIONS)]
        dfs = [self.read_file(path) for path in filtered_paths]
        return dfs

    def list_paths_in_folder(self):
        """Returns a list of the relative paths of all the files in the folder
        Returns
        -------
        list
            List of the paths of all the files in the folder
        """
        paths = []
        for root, _, files in os.walk(self.DEFAULT_PATH):
            for file in files:
                paths.append(os.path.join(root, file))
        return paths

   
def test():
    c = CreateVenv("venv")
    c.install_packages(['pandas==2.1.0'])
    c.run_file('main.py')

    reader = ReadData('Data')
    data_dfs = reader.get_data_dfs()
    print(data_dfs[0])


        