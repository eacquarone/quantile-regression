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

	list_tables = []

	for i in range(3):
		df = pd.DataFrame({'Census' : ["1980", "1990", "2000"],
						   'Obs' : [list_tab_80[i]["perwt"], list_tab_90[i]["perwt"], list_tab_00[i]["perwt"]],
						   'CQ9010' : [list_tab_80[i]["d9010"], list_tab_90[i]["d9010"], list_tab_00[i]["d9010"]],
						   'QR9010' : [list_tab_80[i]["ad9010"], list_tab_90[i]["ad9010"], list_tab_00[i]["ad9010"]],
						   'CQ9050' : [list_tab_80[i]["d9050"], list_tab_90[i]["d9050"], list_tab_00[i]["d9050"]],
						   'QR9050' : [list_tab_80[i]["ad9050"], list_tab_90[i]["ad9050"], list_tab_00[i]["ad9050"]],
						   'CQ5010' : [list_tab_80[i]["d5010"], list_tab_90[i]["d5010"], list_tab_00[i]["d5010"]],
						   'QR5010' : [list_tab_80[i]["ad5010"], list_tab_90[i]["ad5010"], list_tab_00[i]["ad5010"]]})
		list_tables += [df.loc[:,["Census","Obs","CQ9010","QR9010","CQ9050","QR9050","CQ5010","QR5010"]]]


	list_tables[0].to_csv("Data/overall.csv", index = False)
	list_tables[1].to_csv("Data/highschool.csv", index = False)
	list_tables[2].to_csv("Data/college.csv", index = False)

if __name__ == '__main__':
	main()