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

void Shuffle(
    std::default_random_engine & generator, const uint_t size, uint8_t * arr
);

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
        uint_t _unweight;
        uint_t _unsize;
        uint_t * _unhappy;

        // for root only
        uint_t * _partweight;
        uint_t * _partsize;
        uint8_t * _partrank;

        // random generator
        std::default_random_engine _generator;
        
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
        friend void Shuffle(
            std::default_random_engine & generator, const uint_t size,
            uint8_t * arr
        );
        friend void Shuffle(
            std::default_random_engine & generator, const uint_t size,
            uint_t * inds, uint8_t * arr
        );

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
    _unsize(0),
    _unweight(0),
    _unhappy(NULL),
    _partsize(NULL),
    _partrank(NULL)
{}

City::City(const uint_t size, const double thresh):
    _thresh(thresh),
    _size{0, size},
    _unsize(0),
    _unweight(0),
    _partsize(NULL),
    _partrank(NULL),
    _partweight(NULL)
{
    MPI_Comm_rank(MPI_COMM_WORLD, &_prank);
    MPI_Comm_size(MPI_COMM_WORLD, &_psize);

    _border[0] = (_prank > 0);
    _border[1] = (_prank < _psize - 1);
    _size[0] = _size[1] / _psize + (_prank < _size[1] % _psize);
    _unhappy = (uint_t *)malloc(_size[1] * _size[0] * sizeof(uint_t));
    _houses = (uint8_t *)malloc(_size[1] * GetFullHeight());

    _generator.seed(int(MPI_Wtime() * 10000) ^ _prank);
    std::uniform_int_distribution<int> distribution(0, 3);

    uint8_t tmp;

    // set houses randomly
    for (int s = 0; s < _size[0] * _size[1]; ++s)
    {
        tmp = distribution(_generator);
        SetHouse(s, (tmp > 0));
    }
    
    Shuffle(_generator, _size[0] * _size[1], _houses); 
    
    // only on root
    if (!_prank)
    {
        _partrank = (uint8_t *)malloc(_psize);
        _partsize = (uint_t *)malloc(_psize * sizeof(uint_t));
        _partweight = (uint_t *)malloc(_psize * sizeof(uint_t));
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

    if (_partsize)
    {
        free(_partsize);
    }

    if (_partweight)
    {
        free(_partweight);
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

// Fisher-Yates shuffle
void Shuffle(
    std::default_random_engine & generator, const uint_t size, uint8_t * arr
)
{
    std::uniform_int_distribution<int> distribution(0, size - 1);

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
    std::default_random_engine & generator, const uint_t size, uint_t * inds,
    uint8_t * arr
)
{
    std::uniform_int_distribution<int> distribution(0, size - 1);

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
    _unsize = 0;
    _unweight = 0;

    uint_t vweight;

    for (int row = 0; row < _size[0]; ++row)
    {
        for (int col = 0; col < _size[1]; ++col)
        {
            vweight = 0;

            if (row || _border[0])
            {
                vweight += GetHouse(row - 1, col);
                if (col) { vweight += GetHouse(row - 1, col - 1); }

                if (col < _size[1] - 1)
                {
                    vweight += GetHouse(row - 1, col + 1);
                }
            }

            if (row < _size[0] - 1 || _border[1])
            {
                vweight += GetHouse(row + 1, col);
                if (col) { vweight += GetHouse(row + 1, col - 1); }

                if (col < _size[1] - 1)
                {
                    vweight += GetHouse(row + 1, col + 1);
                }
            }

            if (col) { vweight += GetHouse(row, col - 1); }
            if (col < _size[1] - 1) { vweight += GetHouse(row, col + 1); }

            if (
                GetHouse(row, col)?
                double(vweight) < (1. - _thresh) * GetVicinitySize(row, col):
                double(vweight) > _thresh * GetVicinitySize(row, col)
            )
            {
                _unweight += GetHouse(row, col);
                _unhappy[_unsize++] = GetFullIndex(row, col);
            }
        }
    }

    return;
}

void City::RedistributeUnhappy(void)
{
    if (_psize > 1)
    {
        // reduce process weights
        MPI_Reduce(
            _prank? &_unweight: MPI_IN_PLACE, &_unweight, 1, MPI_INT, MPI_SUM,
            0, MPI_COMM_WORLD
        );

        // gather process sizes
        MPI_Gather(
            &_unsize, 1, MPI_INT, _partsize, 1, MPI_INT, 0, MPI_COMM_WORLD
        );

        // redistribute to processes
        if (!_prank)
        {
            _partrank[0] = 0;
            _partweight[0] = 0;

            // initialize ranks, sum-reduce sizes
            for (int p = 1; p < _psize; ++p)
            {
                _partrank[p] = p;
                _unsize += _partsize[p];
                _partweight[p] = 0;
            }

            // shuffle ranks
            Shuffle(_generator, _psize, _partrank);

            std::uniform_int_distribution<int> distribution(0, _unsize - 1);

            /// // randomly redistribute weights to all processes except last
            /// for (int p = 0; p < _psize - 1 && _unweight; ++p)
            /// {
            ///     if (tmp = min(_partsize[_partrank[p]], _unweight))
            ///     {
            ///         _partweight[_partrank[p]] = distribution(_generator) % tmp;
            ///         _unweight -= _partweight[_partrank[p]];
            ///     }
            /// }

            /// // put as most as possible to last process
            /// _partweight[_partrank[_psize - 1]]
            ///     = min(_partsize[_partrank[_psize - 1]], _unweight);
            /// _unweight -= _partweight[_partrank[_psize - 1]];
              
            /// // distribute the reminder by one to the successive processes
            /// for (int p = 0; _unweight; --_unweight, ++p)
            /// {
            ///     if (p == _psize) { p -= _psize; }

            ///     if (_partweight[_partrank[p]] < _partsize[_partrank[p]])
            ///     {
            ///         ++(_partweight[_partrank[p]]);
            ///     }
            /// }
            
            int tmpsize;
            int tmpweight;
            
            // randomly redistribute weights to all processes except last
            for (int p = 0; _unweight; ++p)
            {
                printf("ps[0] = %d, pw[0] = %d\n", _partsize[_partrank[0]], _partweight[_partrank[0]]);
                printf("ps[1] = %d, pw[1] = %d\n", _partsize[_partrank[1]], _partweight[_partrank[1]]);
                if (p == _psize) { p = 0; }

                if (_unweight < 3)
                {
                    if ( _partweight[_partrank[p]] <= _partsize[_partrank[p]] - _unweight)
                    {
                        _partweight[_partrank[p]] += _unweight;
                        break;
                    }
                    else
                    {
                    printf("w = %d\n", _unweight);
                        continue;
                    }
                }

                if (tmpsize = min(_partsize[_partrank[p]], _unweight))
                {
                    printf("_partsize[_partrank[%d]] = %d, tmpsize = %d, _unweight = %d\n", _partrank[p], _partsize[_partrank[p]], tmpsize, _unweight);
                    tmpweight = ((distribution(_generator) % tmpsize) >> 1);
                    _partweight[_partrank[p]] += tmpweight;
                    _partsize[_partrank[p]] -= tmpweight;
                    _unweight -= tmpweight;
                }
            }
        }

        // scatter redistributed weights
        MPI_Scatter(
            _partweight, 1, MPI_INT, &_unweight, 1, MPI_INT, 0, MPI_COMM_WORLD 
        );
    }

    // set ones to be redistributed to the beginning
    // zeros, zeros everywhere (else)!
    for (int u = 0; u < _unsize; ++u)
    {
        _houses[_unhappy[u]] = (u < _unweight);
    }

    // shuffle ranks
    Shuffle(_generator, _unsize, _unhappy, _houses);

    return;
}

////////////////////////////////////////////////////////////////////////////////
//  Result dump methods
////////////////////////////////////////////////////////////////////////////////
void City::FileDump(const uint_t iteration)
{
    uint8_t tmp;
    char filename[32];

    snprintf(
        filename, sizeof(char) * 32, "dump_%d_%d.ppm", _prank, iteration
    );
    FILE * file = fopen(filename, "wb");

    fprintf(file, "P2\n#\n");
    fprintf(file, "%05d %05d\n%02d", _size[1], _size[0], 1);

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
        snprintf(buffer, 32 * sizeof(char), "\n");
        MPI_File_write(file, buffer, 1, MPI_CHAR, &status);

        for (int col = 0; col < _size[1]; ++col)
        {
            tmp = GetHouse(row, col);
            snprintf(buffer, 32 * sizeof(char), "%d ", tmp);
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
        //printf("%d: %d %d\n", i, _unweight, _unsize);
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
