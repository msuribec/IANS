from src.utils import CreateVenv


if __name__ == "__main__":
    c = CreateVenv()
    c.install_packages(['pandas', 'xlrd', 'openpyxl', 'matplotlib'])
    c.run_file('src/run_clustering.py')
