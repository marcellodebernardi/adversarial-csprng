#!/usr/bin/env python

import argparse
import sys
import randtest

tests={1:'monobitfrequencytest',\
       2:'blockfrequencytest',\
       3:'runstest',\
       4:'longestrunones10000',
       5:'binarymatrixranktest',\
       6:'spectraltest',\
       7:'nonoverlappingtemplatematchingtest',\
       8:'overlappingtemplatematchingtest',\
       9:'maurersuniversalstatistictest',\
       10:'linearcomplexitytest',\
       11:'serialtest',\
       12:'aproximateentropytest',\
       13:'cumultativesumstest',\
       14:'randomexcursionstest',\
       15:'randomexcursionsvarianttest',\
       16:'cumultativesumstestreverse',\
       17:'lempelzivcompressiontest'\
      }

randthrs=0.01 # the defined threshold for the tests

parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=('''\
        Evaluate the input for randomness 
        NIST Test Suite
        see: http://csrc.nist.gov/rng/'''))

parser.add_argument('-i','--infile', type=argparse.FileType('r'), default=sys.stdin,
        help='the file from where the random number should be read'
             '(default: read from stdin)')
parser.add_argument('-o','--outfile', type=argparse.FileType('w'), default=sys.stdout,
        help='the file where the result should be written '
             '(default: write the result to stdout)')
parser.add_argument('-t','--tests',nargs='*',type=int,default=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],help='which tests to run, default: all')
parser.add_argument('-b','--binary', action='store_true', help='read in binary file')
parser.add_argument('-c','--comma', action='store_true', help='seperate result with comma')
parser.add_argument('-m','--mathematica', action='store_true', help='prepare for Mathematica input')
parser.add_argument('-j','--join', action='store_true', help='read from a text file and join the lines')
parser.add_argument('-e','--evaluate', action='store_true', help='evaluate if Random or not (True/False)')
parser.add_argument('-s','--summarize', action='store_true', help='evaluate & summarize if Random or not (True/False)')
parser.add_argument('-x','--explanations', action='store_true', help='show a header of these runs')
parser.add_argument('-v','--version', action='version', version='%(prog)s 1.0')

args = parser.parse_args()

##############################################################################

out=""
res=""
header=""
index=0

# set default value, which tests should be performed
if len(args.tests)>=1: testlist=args.tests
else: testlist=range(1,16)
ltst=len(testlist)

# define the separator for the following outputs
if args.comma or args.mathematica: sep=","
else: sep="\t"

def performrndtest(inputbits):
    outh=""
    testhere=0
    for i in testlist:
         testhere+=1
         thistest=str(eval("randtest."+tests.get(i))(inputbits))
         if args.evaluate:
             if thistest<randthrs: outh+="False"
             else: outh+="True"
         else:
             outh+=thistest
         if testhere<ltst: outh+=sep
    return outh

# gie out the header for the following tests
testhere=0
for i in testlist:
    testhere+=1
    header+=str(tests.get(i))
    if testhere<len(testlist): header+=sep

if args.binary: # interpret the file as a binary file, 
    bytes_read=args.infile.read()
    randin=""
    for b in bytes_read:
        n=ord(b)
        bstr=""
        for x in xrange(8):
            bstr=str(n%2) + bstr
            n = n >> 1
        randin+=bstr
    out+=performrndtest(randin)

elif args.join:  # joine all lines in a text file
    randin=""
    for bits in args.infile.readlines():
        randin+=bits.strip()
    out+=performrndtest(randin)

else: # otherwise cycle through all lines in the file
    for bits in args.infile.readlines():
        index+=1
        if index>1: out+="\n"
        if ltst>1: out+="["
        out+=performrndtest(bits.strip())

if args.mathematica: # format nicely for mathematica output
    out=out.replace("[","{")
    out=out.replace("]\n","},")
    out=out.replace("]","}")
    out=out.replace("}{","},{")
    if not args.evaluate: out=out.replace("e","*10^")
    if ltst>1: res+="randomtestsname={"+header+"}\nrandomtests={"+out+"}"
    else: res+="randomtestsname="+header+"\nrandomtests="+out+""
elif args.explanations: 
    res+="# "+header+"\n"+out
else:
    res+=out

args.outfile.write('%s\n' % res) # finalize the output and close the file
args.outfile.close()

