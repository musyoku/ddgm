from scipy.io import wavfile
import numpy as np
import os, sys, time
sys.path.append(os.path.split(os.getcwd())[0])
from model import params, ddgm
from args import args
import dataset

def main():
	np.random.seed(args.seed)

if __name__ == '__main__':
	main()
