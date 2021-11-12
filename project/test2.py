import os, os.path
import csv

def main():
	# csv file name
	filename = "Data/all_the_news/articles1.csv"
	print('AHHHHHHHHHH')
	print('â€”')
	
	file = open(filename, errors="ignore")
	  
	csvreader = csv.reader(file)
	
	header = []
	header = next(csvreader)[0:10]
	print(header[4])
	
	rows = []
	for row in csvreader:
		#print("hi")
		#row = row.decode("iso-8859-1")
		rows.append(row[0:10]) #0:10
	print(rows[0])
	
	code = 'â€”'
	
	print(code in rows[0])
	#print(len(rows))
	
	#listSet = set(rows)
	#print(len(listSet))

if __name__ == "__main__":
    main()