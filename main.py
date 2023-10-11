from src.utils import CreateVenv

def main(file_path, args = None):
    c = CreateVenv() #create virtual environment
    c.install_packages(['scikit-learn','pandas', 'xlrd', 'openpyxl', 'matplotlib' , 'scikit-learn']) #install packages
    c.run_file(file_path, args) #run file

if __name__ == "__main__":
    main('src/run_all.py', ['Data/iris.csv' ,'Iris'])