import glob
import argparse
import numpy as np
from tqdm import tqdm
import concurrent.futures
from library.constants import data_folder
from library.helper import completely_process_one_folder,merge_all_processed_trails_and_dump

parser=argparse.ArgumentParser(description='Execute Data Processing Pipeline')
parser.add_argument('--workers', type=int, help='max cpu worker to be used', default=2)
args = parser.parse_args()


folders=glob.glob(data_folder+"/*/*")
folder_numbers=np.arange(len(folders))
params=list(zip(folder_numbers,folders))


with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
    results=list(tqdm(executor.map(lambda e:completely_process_one_folder(*e), params), total=len(params)))

merge_all_processed_trails_and_dump()
print("Completed")