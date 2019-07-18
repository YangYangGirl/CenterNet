import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--msg_p', action='store', dest="msg_path",
                        help="path")
parser.add_argument('-g', action='store', dest="graph",
                        help="input graph")
parser.add_argument('-p', action='store', dest="prizes",
                        help="input prizes")
parser.add_argument('-t', action='store', dest="terminals", default=None,
                        help="input terminals")
parser.add_argument('-o', action='store', dest="output", default='pcsf.graphml',
                        help="graphML format to cytoscape")
args = parser.parse_args()
