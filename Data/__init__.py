
from .DataReader_TMNIST import DataReader_TMNIST
from .DataReader_GivmeCred import DataReader_GivmeCred
from .getDataReader import getData


dataReaders = {"TMNIST":DataReader_TMNIST,"GIVECREDIT":DataReader_GivmeCred}