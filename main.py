from src.utils import CreateVenv

def main(file_path, args = None):
    """Runs the given file inside a virtual environment
    Parameters
    ----------
    file_path : str
        Path of the file to run
    args : list, optional
        arguments to pass to the file, by default None
    """
    c = CreateVenv() #create virtual environment
    c.install_requirements('requirements.txt') #install requirements
    c.run_file(file_path, args) #run file with list of arguments

if __name__ == "__main__":

    # Run autoencoder to get low and high dimensional data
    
    main('src/autoencoder.py', ['Data/Mall_Customers.csv','Data/Mall_Customers_high.csv', 'high'])
    main('src/autoencoder.py', ['Data/Mall_Customers.csv',f'Data/Mall_Customers_low.csv', 'low'])
    main('src/autoencoder.py', ['Data/iris.csv' ,'Data/iris_high.csv' , 'high'])
    main('src/autoencoder.py', ['Data/iris.csv' ,'Data/iris_low.csv', 'low'])

    # Run naive, exploratory, centroid based and gravity based algorithms

    main('src/run_all.py', ['Data/iris.csv' ,'Iris', '2,3', ''])
    main('src/run_all.py', ['Data/iris_low_2.csv' ,'Iris Low Dimension', '2,3' ,''])
    main('src/run_all.py', ['Data/iris_high.csv' ,'Iris High Dimension', '2,3' ,''])
    main('src/run_all.py', ['Data/iris.csv' ,'Iris UMAP', '2,3', 'umap' , "{'n_neighbors': 15, 'min_dist': 0.3, 'n_components': 2, 'metric':'euclidean'}"])

    main('src/run_all.py', ['Data/Mall_Customers.csv' ,'Customers', '2,3', ''])
    main('src/run_all.py', ['Data/Mall_Customers_low3.csv' ,'Customers Low Dimension', '2,3', ''])
    main('src/run_all.py', ['Data/Mall_Customers_high.csv' ,'Customers High Dimension', '2,3', ''])
    main('src/run_all.py', ['Data/Mall_Customers.csv','Customers UMAP', '2,3', 'umap' , "{'n_neighbors': 15, 'min_dist': 0.3, 'n_components': 2, 'metric':'euclidean'}"])
