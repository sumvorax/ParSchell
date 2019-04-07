include Makefile.in

all: clean compile run convert rm_ppm

rm_ppm:
	rm -rf *.ppm

rm_png:
	rm -rf *.png

clean: rm_png
	rm -rf *.out *.gif

compile:
	mpicxx -O3 schelling.cpp -o schelling.out

run:
	mpirun -n $(NUM_PROC) ./schelling.out $(SIZE) $(THRESH) $(ITER)

convert:
	./convert.sh $(ITER)

