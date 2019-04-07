// schelling.cpp

/*******************************************************************************
 
    PARALLEL SCHELLING MODEL CELLULAR AUTOMATON

********************************************************************************

    Parallel run:
  * Compile with 'mpicxx schelling.cpp -o schelling.out'
  * Run with 'mpirun -n <p> ./schelling.out <size> <threshold> <iterations>'

*******************************************************************************/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <random>
#include <inttypes.h>

typedef unsigned int uint_t;

typedef std::default_random_engine randengine_t;
typedef std::uniform_int_distribution<int> udist_t;
typedef std::bernoulli_distribution bdist_t;

void Shuffle(randengine_t & generator, const uint_t size, uint8_t * arr);

////////////////////////////////////////////////////////////////////////////////
//  City class
////////////////////////////////////////////////////////////////////////////////
class City
{
        //====================================================================//
        //  Fields
        //====================================================================//
        // MPI specifiers
        int _prank;
        int _psize; 

        // city map 
        uint_t _size[2]; // height, width
        uint_t _border[2]; // upper, lower
        uint8_t * _houses; 

        // threshold
        double _thresh;

        // unhappy houses
        uint_t _locstate[2];
        uint_t _totstate[2];
        uint_t * _unhappy;

        // for root only
        uint_t * _partstate;
        uint_t * _partrank;

        // random generator
        randengine_t _generator;
        
        //====================================================================//
        //  Get/Set methods
        //====================================================================//
        uint_t GetFullHeight(void) const;
        uint_t GetFullIndex(const int row, const int col) const;
        uint_t GetVicinitySize(const int row, const int col) const;
        
        uint8_t * GetFirstRow(void);
        uint8_t * GetLastRow(void);
        uint8_t * GetUpperGhost(void);
        uint8_t * GetLowerGhost(void);

        uint8_t & GetHouse(const int ind) const;
        uint8_t & GetHouse(const int row, const int col) const;
        void SetHouse(const int ind, const uint_t house);
        void SetHouse(const int row, const int col, const uint_t house);

        //====================================================================//
        //  Redistribution methods
        //====================================================================//
        friend void Swap(uint8_t & first, uint8_t & second);
        friend void Swap(uint_t & first, uint_t & second);
        friend void Shuffle(
            randengine_t & generator, const uint_t size, uint_t * arr
        );
        friend void Shuffle(
            randengine_t & generator, const uint_t size, uint8_t * arr
        );
        friend void Shuffle(
            randengine_t & generator, const uint_t size, const uint_t * inds,
            uint8_t * arr
        );

        void LocalDistribute(void);
        void DefaultDistribute(void);

        void ExchangeGhosts(void);
        void FindUnhappy(void);
        void RedistributeUnhappy(void);

        //====================================================================//
        //  Result dump methods
        //====================================================================//
        void FileDump(const uint_t step);
        void ParallelFileDump(const uint_t step);

    public:

        //====================================================================//
        //  Structors
        //====================================================================//
        City(void);
        City(const uint_t size, const double thresh);
        ~City(void);

        //====================================================================//
        //  Iterate
        //====================================================================//
        void Iterate(const uint_t iters);
};

////////////////////////////////////////////////////////////////////////////////
//  Get/Set methods
////////////////////////////////////////////////////////////////////////////////
inline uint_t City::GetFullHeight(void) const
{
    return _size[0] + _border[0] + _border[1];
}

inline uint_t City::GetFullIndex(const int row, const int col) const
{
    return (row + _border[0]) * _size[1] + col;
}

inline uint_t City::GetVicinitySize(const int row, const int col) const
{ 
    uint_t r = (row > 0 || _border[0]);  
    uint_t c = (col > 0);
    uint_t res = r + c + (r && c);

    c = (col < _size[1] - 1);
    res += c + (r && c);

    r = (row < _size[0] - 1 || _border[1]);
    res += r + (r && c) + (r && (col > 0));  

    //printf("p = %d, [%d, %d]: res = %d\n", _prank, row, col, res);
    return res;
}

inline uint8_t * City::GetFirstRow(void)
{
    return _houses + (_border[0]? _size[1]: 0);
}

inline uint8_t * City::GetLastRow(void)
{
    return _houses + (_size[0] - !(_border[0])) * _size[1];
}

inline uint8_t * City::GetUpperGhost(void)
{
    return _houses;
}

inline uint8_t * City::GetLowerGhost(void)
{
    return _houses + (_size[0] + _border[0]) * _size[1];
}

inline uint8_t & City::GetHouse(const int ind) const
{
    return _houses[(_border[0]? _size[1]: 0) + ind];
}

inline uint8_t & City::GetHouse(const int row, const int col) const
{
    return _houses[(row + _border[0]) * _size[1] + col];
}

inline void City::SetHouse(const int ind, const uint_t house)
{
    _houses[(_border[0]? _size[1]: 0) + ind] = uint8_t(house);
}

inline void City::SetHouse(const int row, const int col, const uint_t house)
{
    _houses[(row + _border[0]) * _size[1] + col] = uint8_t(house);
}

////////////////////////////////////////////////////////////////////////////////
//  Structors
////////////////////////////////////////////////////////////////////////////////
City::City(void):
    _prank(0),
    _psize(0),
    _size{0, 0},
    _border{0, 0},
    _houses(NULL),
    _thresh(0),
    //_unsize(0),
    //_unweight(0),
    _locstate{0, 0},
    _totstate{0, 0},
    _unhappy(NULL),
    _partstate(NULL),
    _partrank(NULL)
{}

City::City(const uint_t size, const double thresh):
    _size{0, size},
    _border{0, 0},
    _thresh(thresh),
    //_unsize(0),
    //_unweight(0),
    _locstate{0, 0},
    _totstate{0, 0},
    _partstate(NULL),
    _partrank(NULL)
{
    MPI_Comm_rank(MPI_COMM_WORLD, &_prank);
    MPI_Comm_size(MPI_COMM_WORLD, &_psize);

    _size[0] = _size[1] / _psize + (_prank < _size[1] % _psize);

    if (_size[0])
    {
        _border[0] = (_prank > 0);
        if (_prank < _size[1] - 1) { _border[1] = (_prank < _psize - 1); }
    }

    _unhappy = (uint_t *)malloc(_size[0] * _size[1] * sizeof(uint_t));
    _houses = (uint8_t *)malloc(GetFullHeight() * _size[1]);

    _generator.seed(int(MPI_Wtime() * 10000) ^ _prank);
    udist_t distribution(0, 3);

    uint8_t tmp;

    // set houses randomly
    for (int s = 0; s < _size[0] * _size[1]; ++s)
    {
        tmp = distribution(_generator);
        SetHouse(s, (tmp > 0));
    }
    
    // shuffle houses
    Shuffle(_generator, _size[0] * _size[1], _houses); 
    
    // only on root
    if (!_prank)
    {
        _partrank = (uint_t *)malloc(_psize * sizeof(uint_t));
        _partstate = (uint_t *)malloc(2 * _psize * sizeof(uint_t));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    return;
}

City::~City(void)
{
    if (_houses)
    {
        free(_houses);
    }

    if (_unhappy)
    {
        free(_unhappy);
    }

    if (_partrank)
    {
        free(_partrank);
    }

    if (_partstate)
    {
        free(_partstate);
    }

    return;
}

////////////////////////////////////////////////////////////////////////////////
//  Happy tree friends :)
////////////////////////////////////////////////////////////////////////////////
// min
inline const uint_t & min(const uint_t & first, const uint_t & second)
{
    return (first < second)? first: second;
}

// swap
inline void Swap(uint8_t & first, uint8_t & second)
{
    uint8_t tmp = second;
    second = first;
    first = tmp;

    return;
}

inline void Swap(uint_t & first, uint_t & second)
{
    uint_t tmp = second;
    second = first;
    first = tmp;

    return;
}

// Fisher-Yates shuffle
void Shuffle(
    randengine_t & generator, const uint_t size, uint_t * arr
)
{
    udist_t distribution(0, size - 1);

    for (int q = 0; q < 3; ++q)
    {
        for (int s = 0; s < size; ++s)
        {
            Swap(arr[s], arr[distribution(generator)]);
        }
    }

    return;
}

void Shuffle(
    randengine_t & generator, const uint_t size, uint8_t * arr
)
{
    udist_t distribution(0, size - 1);

    for (int q = 0; q < 3; ++q)
    {
        for (int s = 0; s < size; ++s)
        {
            Swap(arr[s], arr[distribution(generator)]);
        }
    }

    return;
}

// Fisher-Yates shuffle, double addressation
void Shuffle(
    randengine_t & generator, const uint_t size, const uint_t * inds,
    uint8_t * arr
)
{
    udist_t distribution(0, size - 1);

    for (int q = 0; q < 3; ++q)
    {
        for (int s = 0; s < size; ++s)
        {
            Swap(arr[inds[s]], arr[inds[distribution(generator)]]);
        }
    }

    return;
}

////////////////////////////////////////////////////////////////////////////////
//  Redistribution methods
////////////////////////////////////////////////////////////////////////////////
void City::ExchangeGhosts(void)
{
    MPI_Request request;

    if (_border[0])
    {
        MPI_Isend(
            GetFirstRow(), _size[1], MPI_UINT8_T, _prank - 1, _prank,
            MPI_COMM_WORLD, &request
        );

        MPI_Irecv(
            GetUpperGhost(), _size[1], MPI_UINT8_T, _prank - 1, _prank - 1,
            MPI_COMM_WORLD, &request
        );
    }

    if (_border[1])
    {
        MPI_Isend(
            GetLastRow(), _size[1], MPI_UINT8_T, _prank + 1, _prank,
            MPI_COMM_WORLD, &request
        );

        MPI_Irecv(
            GetLowerGhost(), _size[1], MPI_UINT8_T, _prank + 1, _prank + 1,
            MPI_COMM_WORLD, &request
        );
    }

    MPI_Barrier(MPI_COMM_WORLD);

    return;
}

void City::FindUnhappy(void)
{
    memset(_locstate, 0, 2 * sizeof(int));

    uint_t vicweight;

    for (int row = 0; row < _size[0]; ++row)
    {
        for (int col = 0; col < _size[1]; ++col)
        {
            vicweight = 0;

            if (row || _border[0])
            {
                vicweight += GetHouse(row - 1, col);
                if (col) { vicweight += GetHouse(row - 1, col - 1); }

                if (col < _size[1] - 1)
                {
                    vicweight += GetHouse(row - 1, col + 1);
                }
            }

            if (row < _size[0] - 1 || _border[1])
            {
                vicweight += GetHouse(row + 1, col);
                if (col) { vicweight += GetHouse(row + 1, col - 1); }

                if (col < _size[1] - 1)
                {
                    vicweight += GetHouse(row + 1, col + 1);
                }
            }

            if (col) { vicweight += GetHouse(row, col - 1); }
            if (col < _size[1] - 1) { vicweight += GetHouse(row, col + 1); }

            if (
                GetHouse(row, col)?
                double(vicweight) < (1. - _thresh) * GetVicinitySize(row, col):
                double(vicweight) > _thresh * GetVicinitySize(row, col)
            )
            {
                _locstate[0] += GetHouse(row, col);
                _unhappy[(_locstate[1])++] = GetFullIndex(row, col);
            }
        }
    }

    return;
}

// put random weight with precalculated probability 
void City::LocalDistribute(void)
{
    bdist_t distribution(double(_totstate[0]) / _totstate[1]);

    _locstate[0] = 0;

    for (int u = 0; u < _locstate[1]; ++u)
    {
        if (distribution(_generator)) { ++(_locstate[0]); }
    }

    return;
}

void City::RedistributeUnhappy(void)
{
    if (_psize > 1)
    {
        uint_t total;

        // reduce process states
        MPI_Allreduce(
            _locstate, _totstate, 2, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD
        );

        LocalDistribute();

        // gather process states
        MPI_Gather(
            _locstate, 2, MPI_UNSIGNED, _partstate, 2, MPI_UNSIGNED, 0,
            MPI_COMM_WORLD
        );

        // redistribute to processes
        if (!_prank)
        {
            uint_t tmp;

            _partrank[0] = 0;
            total = _partstate[0];

            // initialize ranks, sum-reduce sizes
            for (int p = 1; p < _psize; ++p)
            {
                _partrank[p] = p;
                total += _partstate[p << 1];
            }

            // shuffle ranks
            Shuffle(_generator, _psize, _partrank);

            //printf("exsess: %d %d\n", total, _totstate[0]);
            //fflush(stdout);

            uint_t excess = total < _totstate[0];
            total = excess? _totstate[0] - total: total - _totstate[0];

            udist_t distribution(1, 3);

            for (int p = 0; total; ++p)
            {
                if (p == _psize) { p = 0; }

                tmp = min(
                    distribution(_generator),
                    _partstate[(_partrank[p] << 1) + 1]
                    - _partstate[_partrank[p] << 1]
                );

                if (tmp)
                {
                    tmp = 1 + (tmp - 1) % total;

                    if (excess)
                    {
                        _partstate[_partrank[p] << 1] += tmp;
                    }
                    else
                    {
                        _partstate[_partrank[p] << 1] -= tmp;
                    }

                    total -= tmp;
                }
                //printf("%d: tmp=%d tot=%d\n", p, tmp, total);
                //fflush(stdout);
            }
        }

        // scatter redistributed weights
        MPI_Scatter(
            _partstate, 2, MPI_UNSIGNED, _locstate, 2, MPI_UNSIGNED, 0,
            MPI_COMM_WORLD
        );
    }
    ///printf("I am %d\n", _prank);
    ///fflush(stdout);

    // set ones to be redistributed to the beginning
    // zeros, zeros everywhere (else)!
    for (int u = 0; u < _locstate[1]; ++u)
    {
        _houses[_unhappy[u]] = (u < _locstate[0]);
    }

    // shuffle ranks
    Shuffle(_generator, _locstate[1], _unhappy, _houses);

    return;
}

////////////////////////////////////////////////////////////////////////////////
//  Result dump methods
////////////////////////////////////////////////////////////////////////////////
void City::FileDump(const uint_t iteration)
{
    uint8_t tmp;
    char filename[32];
    snprintf(filename, 32 * sizeof(char), "dump_%d_%d.ppm", _prank, iteration);

    FILE * file = fopen(filename, "wb");

    fprintf(file, "P2\n#\n");
    fprintf(file, "%05d %05d\n%05d", _size[1], _size[0], 1);

    for (int row = 0; row < _size[0]; ++row)
    {
        fprintf(file, "\n");
        for (int col = 0; col < _size[1]; ++col)
        {
            tmp = GetHouse(row, col);

            fprintf(file, "%d ", tmp);
        }
    }

    fclose(file);

    return;
}

void City::ParallelFileDump(const uint_t iteration)
{
    MPI_Barrier(MPI_COMM_WORLD);

    uint8_t tmp;
    char buffer[32];
    snprintf(buffer, 32 * sizeof(char), "dump_%d.ppm", iteration);

    MPI_Status status;
    MPI_Offset offset
        = ((_prank? 22: 0) + _prank * _size[0] * (1 + 2 *_size[1]))
        * sizeof(char);

    MPI_File file;
    MPI_File_open(
        MPI_COMM_WORLD, buffer, MPI_MODE_WRONLY | MPI_MODE_CREATE,
        MPI_INFO_NULL, &file
    );
    MPI_File_set_view(
        file, offset, MPI_CHAR, MPI_CHAR, "native", MPI_INFO_NULL
    );

    if (!_prank)
    {
        snprintf(
            buffer, 32 * sizeof(char),"P2\n#\n%05d %05d\n%05d", _size[1],
            _size[1], 1
        );
        MPI_File_write(file, buffer, 22, MPI_CHAR, &status);
    }

    for (int row = 0; row < _size[0]; ++row)
    {
        buffer[0] = '\n';
        MPI_File_write(file, buffer, 1, MPI_CHAR, &status);

        for (int col = 0; col < _size[1]; ++col)
        {
            tmp = GetHouse(row, col);
            snprintf(buffer, 6 * sizeof(char), "%d ", tmp);
            MPI_File_write(file, buffer, 2, MPI_CHAR, &status);
        }
    }

    MPI_File_close(&file);
    MPI_Barrier(MPI_COMM_WORLD);

    return;
}

////////////////////////////////////////////////////////////////////////////////
//  Iterate
////////////////////////////////////////////////////////////////////////////////
void City::Iterate(const uint_t iterations)
{
    ParallelFileDump(0);

    for (int i = 1; i <= iterations; ++i)
    {
        ExchangeGhosts();

        FindUnhappy();
        RedistributeUnhappy();

        ParallelFileDump(i);
    }

    return;
}

////////////////////////////////////////////////////////////////////////////////
//  Main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char ** argv)
{
    MPI_Init(&argc, &argv);

    if (argc < 4)
    {
        int prank;
        MPI_Comm_rank(MPI_COMM_WORLD, &prank);

        if (!prank)
        {
            fprintf(
                stderr, "Not enough arguments: size, threshold, iterations\n"
            );
        }

        MPI_Abort(MPI_COMM_WORLD, 42);
    }

    uint_t size = 100;
    double thresh = .51;
    uint_t iter = 1;

    sscanf(argv[1], "%u", &size); 
    sscanf(argv[2], "%lf", &thresh); 
    sscanf(argv[3], "%u", &iter); 

    City city(size, thresh);
    city.Iterate(iter);

    MPI_Finalize();

    return 0;
}

// schelling.cpp
