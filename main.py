from src.utils import CreateVenv

def main(file_path, args = None):
    c = CreateVenv() #create virtual environment
    c.install_packages(['scikit-learn', 'tensorflow', 'umap-learn' , 'umap-learn[plot]','pandas', 'xlrd', 'openpyxl', 'matplotlib']) #install packages
    c.run_file(file_path, args) #run file with list of arguments

if __name__ == "__main__":
    

    # Run naive, exploratory and centroid based algorithms


    # main('src/run_all.py', ['Data/Mall_Customers.csv' ,'Customers', '2,3', ''])
    # main('src/run_all.py', ['Data/Mall_Customers.csv' ,'Customers Low Dimension', '2,3', 'low'])

    main('src/run_all.py', ['Data/Mall_Customers.csv','Customers High Dimension', '2,3', 'high'])
    # main('src/run_all.py', ['Data/Mall_Customers.csv','Customers UMAP', '2,3', 'umap' , "{'n_neighbors': 15, 'min_dist': 0.3, 'n_components': 2, 'metric':'euclidean'}"])


    # # Run the best algorithms in each case, get embedded data through UMAP and plot the results in a 2d scatter plot (with embedded data)


    # Run the best algorithms in each case, get embedded data through UMAP and plot the results in a 2d scatter plot (with embedded data)
    # main('src/run_best.py', ['Data/Mall_Customers.csv' ,'Customers', '','Best params/best_customers.json', "{'n_neighbors': 15, 'min_dist': 0.3, 'n_components': 2, 'metric':'euclidean'}"])
    # main('src/run_best.py', ['Data/Mall_Customers.csv' ,'Customers Low Dimension', 'low','Best params/best_customers_low.json', "{'n_neighbors': 15, 'min_dist': 0.3, 'n_components': 2, 'metric':'euclidean'}"])
    # main('src/run_best.py', ['Data/Mall_Customers.csv' ,'Customers High Dimension', 'high','Best params/best_customers_high.json', "{'n_neighbors': 15, 'min_dist': 0.3, 'n_components': 2, 'metric':'euclidean'}"])
    # main('src/run_best.py', ['Data/Mall_Customers.csv' ,'Customers UMAP', 'umap','Best params/best_customers_umap.json', "{'n_neighbors': 15, 'min_dist': 0.3, 'n_components': 2, 'metric':'euclidean'}"])


    # main('src/run_all.py', ['Data/iris.csv' ,'Iris', '2,3', ''])
    # main('src/run_all.py', ['Data/iris.csv' ,'Iris Low Dimension', '2,3' , 'low'])
    # main('src/run_all.py', ['Data/iris.csv' ,'Iris High Dimension', '2,3' , 'high'])
    # main('src/run_all.py', ['Data/iris.csv' ,'Iris UMAP', '2,3', 'umap' , "{'n_neighbors': 15, 'min_dist': 0.3, 'n_components': 2, 'metric':'euclidean'}"])

    # # Run the best algorithms in each case, get embedded data through UMAP and plot the results in a 2d scatter plot (with embedded data)
    # main('src/run_best.py', ['Data/iris.csv' ,'Iris', '','Best params/best_iris.json', "{'n_neighbors': 15, 'min_dist': 0.3, 'n_components': 2, 'metric':'euclidean'}"])
    # main('src/run_best.py', ['Data/iris.csv' ,'Iris Low Dimension', 'low','Best params/best_iris_low.json', "{'n_neighbors': 15, 'min_dist': 0.3, 'n_components': 2, 'metric':'euclidean'}"])
    # main('src/run_best.py', ['Data/iris.csv' ,'Iris High Dimension', 'high','Best params/best_iris_high.json', "{'n_neighbors': 15, 'min_dist': 0.3, 'n_components': 2, 'metric':'euclidean'}"])
    # main('src/run_best.py', ['Data/iris.csv' ,'Iris UMAP', 'umap','Best params/best_iris_umap.json', "{'n_neighbors': 15, 'min_dist': 0.3, 'n_components': 2, 'metric':'euclidean'}"])
