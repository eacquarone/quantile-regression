import pandas as pd
from quantile_regression import QRGenerator

def main():
	census80 = pd.read_stata("Data/census80.dta")
	census80cq = pd.read_stata("Data/census80cq.dta")

	qr_generator = QRGenerator(census80, census80cq)

	result = qr_generator.process()
	
	result[0].to_csv("Data/census80qr.csv")
	result[1].to_csv("Data/census80g.csv")

if __name__ == '__main__':
	main()