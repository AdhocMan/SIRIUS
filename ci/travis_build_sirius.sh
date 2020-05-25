#!/bin/bash

mkdir build
cd build
cmake ../ -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_PREFIX_PATH=/home/travis/local -DCMAKE_INSTALL_PREFIX=/home/travis/local -DBUILD_TESTS=1
make -j 2 install

