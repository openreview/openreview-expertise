import glob, shutil

def clean_tmp_files():
    """Removes files from the tmp directory that are created by SPECTER"""
    dirs = glob.glob("/tmp/tmp*/")
    for dir in dirs:
        print(f"removing {dir}")
        shutil.rmtree(dir)

if __name__ == '__main__':
    clean_tmp_files()