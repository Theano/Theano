aa.x : aa.cc
	g++ -O3 -ffast-math -ftree-vectorize aa.cc -o aa.x -L${PUB_PREFIX}/lib -lgsl -lmkl
	#g++ aa.cc -o aa.x -L${PUB_PREFIX}/lib -lgsl -lmkl

clean : 
	rm aa.x
