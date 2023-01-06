import matplotlib.colors
import numpy
from Bio.PDB import PDBParser
from immuneML.data_model.encoded_data.EncodedData import EncodedData

from immuneML.data_model.dataset.PDBDataset import PDBDataset
from immuneML.encodings.DatasetEncoder import DatasetEncoder
from immuneML.encodings.EncoderParams import EncoderParams
import numpy as np
from immuneML.data_model.receptor.RegionType import RegionType


class PDBEncoder(DatasetEncoder):



    @staticmethod
    def build_object(dataset=None, **params):
        return PDBEncoder(**params)

    def __init__(self, name: str = None, region_type: RegionType = None):

        self.name = name
        self.region_type = region_type

    def encode(self, dataset, params: EncoderParams):

        structures =[]

        pdbParser = PDBParser(
            PERMISSIVE=True
        )

        CDRcounter = 0



        for files in dataset.get_data():
            parserObject = pdbParser.get_structure("pdbStructure", files)


            collection = []
            list = []

            region = self.region_type
            counter = 0

            for model in parserObject:
                for chain in model:
                    residueCounter = 0
                    for residue in chain:
                        if residueCounter == 0:
                            list.append(residue.get_parent())
                            residueCounter = 1

                        for atom in residue:
                            if (atom.get_name() == "CA"):

                                if "FULL_SEQUENCE" in region:
                                    list.append(atom.get_coord())

                                else:
                                    if "IMGT_CDR3" in region and (isInCDR3(atom.full_id[3][1]) or len(collection) >= 2):
                                        list.append(atom.get_coord())

                                    if "IMGT_CDR2" in region and (isInCDR2(atom.full_id[3][1]) or len(collection) >= 2):
                                        list.append(atom.get_coord())

                                    if "IMGT_CDR1" in region and (isInCDR1(atom.full_id[3][1]) or len(collection) >= 2):
                                        list.append(atom.get_coord())


                    saveList = list.copy()
                    collection.append(saveList)
                    counter = counter + 1
                    list.clear()

            lightID = -1
            heavyID = -1
            antigenID = -1

            for i in range(0,len(collection)):
                chainID = collection[i][0].id

                if 'L' in chainID or 'A' in chainID:
                    lightID = i

                elif 'H' in chainID or 'B' in chainID:
                    heavyID = i

                elif 'P' in chainID or 'C' in chainID:
                    antigenID = i

                collection[i].pop(0)


            if lightID >= 0 and heavyID >= 0 and antigenID == -1:
                if lightID + heavyID == 1:
                    antigenID = 2

                elif lightID + heavyID == 3:
                    antigenID = 0

            elif lightID == -1  or heavyID == -1 or antigenID == -1:
                lightID = 0
                heavyID = 1
                antigenID = 2


            lightChainDistanceToAntigen = np.zeros(shape=(len(collection[antigenID]), len(collection[lightID])))

            caluculateDistance(collection, lightID, antigenID, lightChainDistanceToAntigen)

            heavyChainDistanceToAntigen = np.zeros(shape=(len(collection[antigenID]), len(collection[heavyID])))
            caluculateDistance(collection, heavyID, antigenID, heavyChainDistanceToAntigen)

            paddedArrays = paddingOfArray(lightChainDistanceToAntigen,heavyChainDistanceToAntigen)

            structures.append([paddedArrays[0],paddedArrays[1]])
            CDRcounter = CDRcounter +1


        longestAntigen = 0
        longestChain = 0

        for x in structures:
            if len(x[0]) > longestAntigen:
                longestAntigen = len(x[0])
            if len(x[0][0]) > longestChain:
                longestChain = len(x[0][0])

        paddedStructures = []
        for x in structures:
            paddedStructures.append(expandAntibodyChainToMaxLength(x, longestAntigen, longestChain))



        numpArray = np.array(paddedStructures)

        encoded_dataset = PDBDataset(self, dataset.pdbFilePaths,dataset.labels,dataset.metadata_file,EncodedData(numpArray,None,dataset.get_example_ids()))

        #print("Shape of encoded_dataset.encoded_data.examples: " , numpy.shape(encoded_dataset.encoded_data.examples))
        return encoded_dataset


def caluculateDistance(collection, chain, antigen, list):
    for x in range(len(collection[antigen])):
        for i in range(len(collection[chain])):
            a = np.array((collection[chain][i][0], collection[chain][i][1], collection[chain][i][2]))
            b = np.array((collection[antigen][x][0], collection[antigen][x][1], collection[antigen][x][2]))

            rounded = round(np.linalg.norm(a - b), 2)
            list[x][i] = rounded


"""
This method takes in two 2D arrays as input. The format should be the light chain first and heavy second
Then pads the shortest one to the length of the longest one
It returns a tuple like (lightChain,HeavyChain)

The arrays should be numpy arrays.
The arrays should have the same amount of rows, but not columns
It is the individual rows that gets padded values to match the longer one

"""
def paddingOfArray(lightChain, heavyChain):
    diff = len(lightChain[0]) - len(heavyChain[0])

    if diff > 0:
        longest = lightChain
        shortest = heavyChain
        paddedShortest = []
        for x in shortest:
            paddedArray = np.pad(x, (0, diff), 'constant', constant_values=(0, np.inf))
            paddedShortest.append(paddedArray)

        paddedShortestToNumpyArray = np.array(paddedShortest)
        return (lightChain, paddedShortestToNumpyArray)

    elif diff < 0:
        longest= heavyChain
        shortest= lightChain
        diff = abs(diff)

        paddedShortest = []
        for x in shortest:
            paddedArray = np.pad(x, (0, diff), 'constant', constant_values=(0, np.inf))
            paddedShortest.append(paddedArray)

        paddedShortestToNumpyArray = np.array(paddedShortest)
        return (paddedShortestToNumpyArray, heavyChain)

    else:
        return (lightChain, heavyChain)


##
# This method pads the chains to the max size of the antigen
#
#
# #
def expandAntibodyChainToMaxLength(chains, antigenLength, chainLength):

    returnArray = chains.copy()
    if chainLength > len(chains[0][0]):
        for x in range(0,len(chains)):
            paddedChain = []
            chainDiff = chainLength-len(chains[x][0])

            if chainDiff > 0:
                paddedArrayColumns = []
                for y in chains[x]:
                    paddedArray = np.pad(y, (0, chainDiff), 'constant', constant_values=(0, np.inf))
                    paddedArrayColumns.append(paddedArray)


                paddedChain = np.array(paddedArrayColumns)
            returnArray[x] = paddedChain

    if antigenLength > len(chains[0]):
        for i in range(0,len(chains)):
            antigenDiff = antigenLength - len(chains[i])

            if antigenDiff > 0:
                for j in range(0,antigenDiff):
                    emptyArr = np.full((1, chainLength), np.inf)
                    returnArray[i] = numpy.append(returnArray[i], emptyArr, 0)

    return returnArray


def isInCDR3(num):
    if 117 >= num >= 105:
        return True
    else:
        return False


def isInCDR2(num):
    if 65 >= num >= 56:
        return True
    else:
        return False


def isInCDR1(num):
    if 38 >= num >= 27:
        return True
    else:
        return False