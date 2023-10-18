from src.utils import CreateVenv

def main(file_path, args = None):
    c = CreateVenv() #create virtual environment
    c.install_packages(['scikit-learn', 'tensorflow', 'pandas', 'xlrd', 'openpyxl', 'matplotlib']) #install packages
    c.run_file(file_path, args) #run file with list of arguments

if __name__ == "__main__":
    main('src/run_all.py', ['Data/iris.csv' ,'not', '2,3,4', ''])
    main('src/run_all.py', ['Data/iris.csv' ,'Iris Low Dimension', '2,3,4' , 'low'])
    main('src/run_all.py', ['Data/iris.csv' ,'Iris High Dimension', '2,3,4' , 'high'])