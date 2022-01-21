CXX = g++
CXXFLAGS = -O2 -Wall --std=c++17 -I ~/Downloads/libtorch_2/include/torch/csrc/api/include/ -I ~/Downloads/libtorch_2/include/ -D_GLIBCXX_USE_CXX11_ABI=0

all: npt-mtk

npt-mtk: npt-mtk.o
	$(CXX) -o npt-mtk npt-mtk.o

clean:
	rm -f *.o npt-mtk
