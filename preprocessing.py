from util import *
import ConfigParser

config=ConfigParser.ConfigParser()
config.read('./config.txt')


species = 'Arabidopsis'
chromosomes = config.get(species, 'chromosomes').split(';')
kfold = int(config.get(species, 'kfold'))
mul_train = config.get(species, 'mul_train')
gap = int(config.get(species, 'gap'))
data_file = config.get(species, 'data_file')
output_file = config.get(species, 'output_file')


# print "#####################################################"
# print species + " data normalization start"
# data_normalization(species, chromosomes, data_file, data_file, output_file)
# print species + " data normalization finish"
# print "#####################################################"
# print species + " data split start"
# kfold_split(species, kfold, output_file, output_file, output_file)
# print species + " data split finish"
print "#####################################################"
print species + " mul to gap start"
mul2gap(species, kfold, gap, chromosomes, data_file, output_file, output_file, mul_train)
print species + " mul to gap finish"
print "#####################################################"

