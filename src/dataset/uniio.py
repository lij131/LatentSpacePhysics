#******************************************************************************
#
# Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
# Copyright 2018 Steffen Wiewel, Moritz Becher, Nils Thuerey
#
# read/write uni files
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#******************************************************************************

import gzip
import struct
from datetime import date
from collections import namedtuple
import numpy as np

# this is for reading the field
def _read_content(bytestream, head):
    #dt = np.dtype(np.float64).newbyteorder('>')
    assert (head['bytesPerElement'] == 12 and head['elementType'] == 2) or head['bytesPerElement'] == 4
    if (head['elementType'] == 0):
        data = np.frombuffer(bytestream.read(), dtype="int32")
    else:
        data = np.frombuffer(bytestream.read(), dtype="float32")
    if (head['elementType'] == 2):
        return data.reshape(head['dimX'], head['dimY'], head['dimZ'], 3, order='C')
    else:
        return data.reshape(head['dimX'], head['dimY'], head['dimZ'], order='C')

# read important information in the header such as dimensions and grid type
def _read_head(bytestream):
    ID = bytestream.read(4)
    #dimX, dimY, dimZ, gridType, elementType, bytesPerElement, info, timestamp = struct.unpack('iiiiii256sQ', bytestream.read(288))

    # unpack header struct object
    head = namedtuple('UniHeader', 'dimX, dimY, dimZ, gridType, elementType, bytesPerElement, info, timestamp')
    # convert to namedtuple and then directly to a dict
    head = head._asdict(head._make(struct.unpack('iiiiii256sQ', bytestream.read(288))))
    #head['ID'] = ID
    
    return head

# use this to read the .uni file. It will return the header as dictionary and the content as np-array
def readuni(filename):
    with gzip.open(filename, 'rb') as bytestream:
        head = _read_head(bytestream)
        content = _read_content(bytestream, head)

        return head, content

# use this to write a .uni file. The head has to be supplied in the same dictionary format as the output of readuni
def writeuni(filename, head, content):
    with gzip.open(filename, 'wb') as bytestream:
        # write the head of the uni file
        bytestream.write(b'MNT2')
        head_tuple = namedtuple('GenericDict', head.keys())(**head)
        head_buffer = struct.pack('iiiiii256sQ', *head_tuple)
        bytestream.write(head_buffer)
        if (head['elementType'] == 2):
            content = content.reshape(head['dimX']*head['dimY']*head['dimZ']*3, order='C')
        else:
            content = content.reshape(head['dimX']*head['dimY']*head['dimZ'], order='C')

        bytestream.write(content.tobytes())