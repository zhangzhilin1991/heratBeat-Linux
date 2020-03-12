#!/bin/sh
export PKG_CONFIG_PATH=~/heatbeatproject/numcpp/NumCpp/install/lib/pkg-config:~/heatbeatproject/opencv/opencv-4.2.0/linux-build/install/lib/pkgconfig:~/heatbeatproject/openblas/OpenBLAS/mybuild/linux-build/install/lib/pkgconfig:~/heatbeatproject/numcpp/NumCpp/boost/boost_1_70_0/mybuild/linux-build/install/lib/pkgconfig

export LD_LIBRARY_PATH=~/heatbeatproject/opencv/opencv-4.2.0/linux-build/install/lib:~/heatbeatproject/openblas/OpenBLAS/mybuild/linux-build/install/lib:~/heatbeatproject/numcpp/NumCpp/boost/boost_1_70_0/mybuild/linux-build/install/lib

export LIBRARY_PATH=~/heatbeatproject/opencv/opencv-4.2.0/linux-build/install/lib/:~/heatbeatproject/openblas/OpenBLAS/mybuild/linux-build/install/lib/:~/heatbeatproject/numcpp/NumCpp/boost/boost_1_70_0/mybuild/linux-build/install/lib/:${LIBRARY_PATH}
