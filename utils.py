import os

def create_directory(dir_name):
	if not os.path.exists(dir_name):
		os.makedirs(dir_name)

def clear_directory(dir_name, recursive=False):
	for f in os.listdir(dir_name):
		fpath = os.path.join(dir_name, f)
		try:
			if os.path.isfile(fpath):
				os.unlink(fpath)
			elif recursive and os.path.isdir(fpath):
				delete_directory(fpath, recursive)
				os.unlink(fpath)
		except Exception as e:
			print(e)