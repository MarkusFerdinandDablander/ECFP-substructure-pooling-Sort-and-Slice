import os


for old_filename in os.listdir("."):
	if old_filename[-4:] == ".csv":
		new_filename = old_filename.replace("mpro", "postera_sars_cov_2_mpro")
		new_filename = new_filename.replace("chi2", "filtered")
		new_filename = new_filename.replace("sorted", "sort_and_slice")

		os.replace(old_filename, new_filename)
