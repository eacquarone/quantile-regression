import pandas as pd
from table_generator import TableGenerator

def main():
	census80 = pd.read_stata("Data/census80.dta")
	census90 = pd.read_stata("Data/census90.dta")
	census00 = pd.read_stata("Data/census00.dta")

	table = TableGenerator(census80)
	census80t2g, list_tab_80 = table.process()
	census90t2g, list_tab_90 = table.process(census90)
	census00t2g, list_tab_00 = table.process(census00)
	
	census80t2g.to_csv("Data/census80t2g.csv")
	census90t2g.to_csv("Data/census90t2g.csv")
	census00t2g.to_csv("Data/census00t2g.csv")

if __name__ == '__main__':
	main()