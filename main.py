from src.utils import CreateVenv

def main(file_path, args = None):
    c = CreateVenv() #create virtual environment
    c.install_packages(['scikit-learn', 'tensorflow', 'umap-learn' , 'umap-learn[plot]','pandas', 'xlrd', 'openpyxl', 'matplotlib']) #install packages
    c.run_file(file_path, args) #run file with list of arguments

if __name__ == "__main__":
    main('src/run_all.py', ['Data/iris.csv' ,'Iris UMAP', '2,3,4', 'umap' , "{'n_neighbors': 15, 'min_dist': 0.3, 'n_components': 2, 'metric':'euclidean'}"])


    # main('src/run_all.py', ['Data/iris.csv' ,'Iris', '2,3,4', ''])
    # main('src/run_all.py', ['Data/iris.csv' ,'Iris Low Dimension', '2,3' , 'low'])
    # main('src/run_all.py', ['Data/iris.csv' ,'Iris High Dimension', '2,3' , 'high'])

    # main('src/run_all.py', ['Data/iris.csv' ,'Iris', '2,3,4', 'umap'])