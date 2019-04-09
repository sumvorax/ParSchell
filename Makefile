include Makefile.in

all: clean compile run convert rm_ppm

rm_ppm:
	rm -rf *.ppm

rm_png:
	rm -rf *.png

clean: rm_ppm rm_png
	rm -rf *.out *.gif

compile:
	mpicxx -O3 schelling.cpp -o schelling.out

run:
	mpirun -n $(PROC) ./schelling.out $(ITER) $(SIZE) $(THRESH) $(PROB) $(EMPTY)

convert:
	./convert.sh $(ITER)

