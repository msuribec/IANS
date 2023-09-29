from src.utils import ReadData

reader = ReadData('Data')
data_dfs = reader.get_data_dfs()
print(data_dfs[0])
