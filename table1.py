import pandas as pd
from table_generator import TableGenerator

def main():
	census80 = pd.read_stata("Data/census80.dta")

	table = TableGenerator(census80)

	result = table.process()
	
	result.to_csv("Data/census80t2g.csv")

if __name__ == '__main__':
	main()