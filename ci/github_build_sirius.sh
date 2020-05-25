#!/bin/bash

mkdir build
cd build
cmake ../ -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_PREFIX_PATH=$HOME/local -DCMAKE_INSTALL_PREFIX=$HOME/local -DBUILD_TESTS=1
make -j 2 install

