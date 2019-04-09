// schelling.cpp

/*******************************************************************************
 
    PARALLEL SCHELLING MODEL CELLULAR AUTOMATON

********************************************************************************

  * Compile with: 'mpicxx schelling.cpp -o schelling.out'
  * Run with: 'mpirun -n <proc> ./schelling.out \
        <iter> <size> <thresh> <prob> <empty>'

*******************************************************************************/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <random>
#include <unistd.h>
#include <inttypes.h>

typedef unsigned int uint_t;

typedef std::default_random_engine randgen_t;
typedef std::uniform_int_distribution<int> udist_t;
typedef std::bernoulli_distribution bdist_t;
typedef std::discrete_distribution<int> ddist_t;

typedef enum 
{
    DISCR_ZZZ = 0, // 000
    DISCR_PNZ = 1, // +-0
    DISCR_PPN = 2, // ++-
    DISCR_NNP = 3  // --+
} discrepancy_t; 

static const char * ZZZ = "000";
static const char * PNZ = "+-0";
static const char * PPN = "++-";
static const char * NNP = "--+";

static const char * PrintDiscrepancy(discrepancy_t discrepancy)
{
    if (discrepancy == DISCR_ZZZ)
    {
        return ZZZ;
    }
    else if (discrepancy == DISCR_PNZ)
    {
        return PNZ;
    }
    else if (discrepancy == DISCR_PPN)
    {
        return PPN;
    }
    else 
    {
        return NNP;
    }
}

////////////////////////////////////////////////////////////////////////////////
//  Happy tree friends :)
////////////////////////////////////////////////////////////////////////////////
// min
inline const uint_t & min(const uint_t & first, const uint_t & second)
{
    return (first < second)? first: second;
}

inline const int & absmin(const int sign, const int & first, const int & second)
{
    if (sign > 0)
    {
        return (first < second)? first: second;
    }
    else
    {
        return (first < second)? second: first;
    }
}

// swap
inline void Swap(uint_t & first, uint_t & second)
{
    uint_t tmp = second;
    second = first;
    first = tmp;

    return;
}

inline void Swap(uint8_t & first, uint8_t & second)
{
    uint8_t tmp = second;
    second = first;
    first = tmp;

    return;
}

// Fisher-Yates shuffle
void Shuffle(randgen_t & generator, const uint_t size, uint_t * arr)
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

void Shuffle(randgen_t & generator, const uint_t size, uint8_t * arr)
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
    randgen_t & generator, const uint_t size, const uint_t * inds, uint8_t * arr
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
        uint_t _weights[3]; // night watch, wasteland, white walkers, total
        uint8_t * _houses; 

        // threshold of intolerance
        double _thresh;
        // probability of night watch
        double _prob;
        // probability of wasteland
        double _empty;

        // moving houses
        uint_t _locstate[4]; // night watch, wasteland, white walkers, total
        uint_t _totstate[4]; // night watch, wasteland, white walkers, total
        uint_t * _moving;

        // for root only
        uint_t * _partrank;
        uint_t * _partstate;

        // random generator
        randgen_t _generator;
        
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

        uint8_t & GetHouse(const uint_t ind) const;
        uint8_t & GetHouse(const int row, const int col) const;
        void SetHouse(const int ind, const uint_t house);
        void SetHouse(const int row, const int col, const uint_t house);

        void AddNeighbour(const int row, const int col, uint_t * weights);

        //====================================================================//
        //  Redistribution methods
        //====================================================================//
        friend void Swap(uint_t & first, uint_t & second);
        friend void Swap(uint8_t & first, uint8_t & second);

        friend void Shuffle(
            randgen_t & generator, const uint_t size, uint_t * arr
        );
        friend void Shuffle(
            randgen_t & generator, const uint_t size, uint8_t * arr
        );
        //friend void Shuffle(
        //    randgen_t & generator, const uint_t size, uint_t * arr
        //);
        //friend void Shuffle(
        //    randgen_t & generator, const uint_t size, uint8_t * arr
        //);

        friend void Shuffle(
            randgen_t & generator, const uint_t size, const uint_t * inds,
            uint8_t * arr
        );

        void GuessState(void);
        discrepancy_t DetectDiscrepancy(const int * weights, uint8_t * inds);
        void Equilibrate(void);
        void ExchangeGhosts(void);
        void FindMoving(void);
        void RedistributeMoving(void);

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
        City(
            const uint_t size, const double thresh, const double prob,
            const double empty
        );
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

inline uint8_t & City::GetHouse(const uint_t ind) const
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

    return;
}

inline void City::SetHouse(const int row, const int col, const uint_t house)
{
    _houses[(row + _border[0]) * _size[1] + col] = uint8_t(house);

    return;
}

inline void City::AddNeighbour(const int row, const int col, uint_t * weights)
{
    ++(weights[GetHouse(row, col)]);

    return;
}

////////////////////////////////////////////////////////////////////////////////
//  Structors
////////////////////////////////////////////////////////////////////////////////
City::City(void):
    _prank(0),
    _psize(0),
    _size{0, 0},
    _border{0, 0},
    _weights{0, 0, 0},
    _houses(NULL),
    _thresh(0),
    _prob(0),
    _empty(0),
    _locstate{0, 0},
    _totstate{0, 0},
    _moving(NULL),
    _partstate(NULL),
    _partrank(NULL)
{}

City::City(
    const uint_t size, const double thresh, const double prob,
    const double empty
):
    _size{0, size},
    _thresh(thresh),
    _prob(prob),
    _empty(empty),
    _border{0, 0},
    _weights{0, 0, 0},
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
        // in case this is the last process with data
        if (_prank < _size[1] - 1) { _border[1] = (_prank < _psize - 1); }
    }

    _moving = (uint_t *)malloc(_size[0] * _size[1] * sizeof(uint_t));
    _houses = (uint8_t *)malloc(GetFullHeight() * _size[1]);

    _generator.seed(int(MPI_Wtime() * 10000) ^ _prank);
    ddist_t distribution{_prob, _empty, 1. - _prob - _empty};

    uint8_t house;

    // set houses randomly
    for (int s = 0; s < _size[0] * _size[1]; ++s)
    {
        house = distribution(_generator);
        ++(_weights[house]);

        SetHouse(s, house);
    }
    
    // shuffle houses
    Shuffle(_generator, _size[0] * _size[1], GetFirstRow()); 

    uint_t newweights[3];

    MPI_Reduce(
        _weights, newweights, 3, MPI_UNSIGNED, MPI_SUM, 0, MPI_COMM_WORLD
    );

    if (!_prank)
    {
        printf(
            "WEIGHTS: %d %d %d\n", newweights[0], newweights[1], newweights[2]
        );
    }

    // only on root
    if (_psize > 1 && !_prank)
    {
        _partrank = (uint_t *)malloc(_psize * sizeof(uint_t));
        _partstate = (uint_t *)malloc(4 * _psize * sizeof(uint_t));
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

    if (_moving)
    {
        free(_moving);
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
//  Redistribution methods
////////////////////////////////////////////////////////////////////////////////
void City::ExchangeGhosts(void)
{
    MPI_Request request;

    MPI_Barrier(MPI_COMM_WORLD);

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

// determine local state
void City::FindMoving(void)
{
    uint_t vicstate[3];

    memset(_locstate, 0, 4 * sizeof(uint_t));

    for (int row = 0; row < _size[0]; ++row)
    {
        for (int col = 0; col < _size[1]; ++col)
        {
            memset(vicstate, 0, 3 * sizeof(uint_t));

            if (row || _border[0])
            {
                AddNeighbour(row - 1, col, vicstate);
                if (col) { AddNeighbour(row - 1, col - 1, vicstate); }

                if (col < _size[1] - 1)
                {
                    AddNeighbour(row - 1, col + 1, vicstate);
                }
            }

            if (row < _size[0] - 1 || _border[1])
            {
                AddNeighbour(row + 1, col, vicstate);
                if (col) { AddNeighbour(row + 1, col - 1, vicstate); }

                if (col < _size[1] - 1)
                {
                    AddNeighbour(row + 1, col + 1, vicstate);
                }
            }

            if (col) { AddNeighbour(row, col - 1, vicstate); }
            if (col < _size[1] - 1) { AddNeighbour(row, col + 1, vicstate); }

            if (
                GetHouse(row, col) == 1
                || vicstate[2 - GetHouse(row, col)]
                > _thresh * GetVicinitySize(row, col)
            )
            {
                ///if (_prank)
                ///printf("[%d] row = %d, col = %d, (%d) %d %d %d sum = %d %d\n", 
                ///        _prank, row, col, GetHouse(row, col), _locstate[0], _locstate[1], _locstate[2],
                ///        _locstate[0] + _locstate[1] + _locstate[2],_locstate[3]);
                AddNeighbour(row, col, _locstate);
                _moving[(_locstate[3])++] = GetFullIndex(row, col);
            }
        }
    }

    ///printf("[%d] FINDMOV %d %d %d sum = %d %d\n", 
    ///        _prank, _locstate[0], _locstate[1], _locstate[2],
    ///        _locstate[0] + _locstate[1] + _locstate[2],_locstate[3]);

    MPI_Barrier(MPI_COMM_WORLD);

    return;
}

// generate random weight with precalculated probability 
void City::GuessState(void)
{
    ddist_t distribution{
        double(_totstate[0]), double(_totstate[1]), double(_totstate[2])
    };

    memset(_locstate, 0, 3 * sizeof(uint_t));

    for (int u = 0; u < _locstate[3]; ++u)
    {
        ++(_locstate[distribution(_generator)]);
    }

    ///printf("GUESS %d %d %d sum = %d %d\n", _locstate[0], _locstate[1], _locstate[2], _locstate[0] + _locstate[1] + _locstate[2],_locstate[3]);

    MPI_Barrier(MPI_COMM_WORLD);

    return;
}

// 
discrepancy_t City::DetectDiscrepancy(const int * weights, uint8_t * inds)
{
    discrepancy_t discrepancy = DISCR_ZZZ;

    inds[0] = 0;
    inds[1] = 1;
    inds[2] = 2;

    // (0)??
    if (!(weights[inds[0]]))
    {
        Swap(inds[0], inds[2]);

        // (-)?0
        if (weights[inds[0]] < 0)
        {
            Swap(inds[0], inds[1]);
            discrepancy = DISCR_PNZ;
        }
        // (+)?0
        else if (weights[inds[0]] > 0) { discrepancy = DISCR_PNZ; }
    }
    // (-)??
    else if (weights[inds[0]] < 0)
    {
        Swap(inds[0], inds[1]);

        // (0)-?
        if (!(weights[inds[0]]))
        {
            Swap(inds[0], inds[2]);
            discrepancy = DISCR_PNZ;
        }
        // (-)-?
        else if (weights[inds[0]] < 0) { discrepancy = DISCR_NNP; }
        // (+)-?
        else
        {
            // +-(0)
            if (!(weights[inds[2]])) { discrepancy = DISCR_PNZ; }
            // +-(-)
            else if (weights[inds[2]] < 0)
            {
                Swap(inds[0], inds[2]);
                discrepancy = DISCR_NNP;
            }
            // +-(+)
            else {
                Swap(inds[1], inds[2]);
                discrepancy = DISCR_PPN;
            }
        }
    }
    // (+)??
    else
    {
        // +(0)?
        if (!(weights[inds[1]]))
        {
            Swap(inds[1], inds[2]);
            discrepancy = DISCR_PNZ;
        }
        // +(-)?
        else if (weights[inds[1]] < 0)
        {
            // +-(0)
            if (!(weights[inds[2]])) { discrepancy = DISCR_PNZ; }
            // +-(-)
            else if (weights[inds[2]] < 0)
            {
                Swap(inds[0], inds[2]);
                discrepancy = DISCR_NNP;
            }
            // +-(+)
            else {
                Swap(inds[1], inds[2]);
                discrepancy = DISCR_PPN;
            }
        }
        // +(+)?
        else { discrepancy = DISCR_PPN; }
    }

    return discrepancy;
}

// for root only!
void City::Equilibrate(void)
{
    int total[3];

    _partrank[0] = 0;
    // total = _partsize
    total[0] = _partstate[0];
    total[1] = _partstate[1];
    total[2] = _partstate[2];

    // initialize ranks, sum-reduce sizes
    for (int p = 1; p < _psize; ++p)
    {
        _partrank[p] = p;
        total[0] += _partstate[p << 2];
        total[1] += _partstate[(p << 2) + 1];
        total[2] += _partstate[(p << 2) + 2];
    }
    ///printf("TOTSTATE AFTER %d %d %d\n", total[0], total[1], total[2]);
    ///fflush(stdout);

    total[0] -= _totstate[0];
    total[1] -= _totstate[1];
    total[2] -= _totstate[2];

    //printf("%d %d %d\n", total[0], total[1], total[2]);
    //fflush(stdout);

    ///for (int pp = 0; pp < _psize; ++pp)
    ///{
    ///    printf("PARTSTATE = %d %d %d\n", _partstate[pp << 2], _partstate[(pp << 2) + 1], _partstate[(pp << 2) + 2]);
    ///    fflush(stdout);
    ///}

    // shuffle ranks
    Shuffle(_generator, _psize, _partrank);
    ///for (int pp = 0; pp < _psize; ++pp)
    ///{
    ///    printf("%d ", _partrank[pp]);
    ///    printf("SHUFFLE partstate = %d %d %d\n", _partstate[pp << 2], _partstate[(pp << 2) + 1], _partstate[(pp << 2) + 2]);
    ///    fflush(stdout);
    ///}

    ddist_t where{1, 1};
    udist_t what(1, 3);
    uint_t tmp;
    uint8_t inds[3];
    discrepancy_t discrepancy;

    // 000 -- nothing to do
    if ((discrepancy = DetectDiscrepancy(total, inds)) == DISCR_ZZZ)
    {
        return;
    }
    // +-0 / ++- / --+
    else
    {
        int pos;
        int end = 2;
        int sign = (discrepancy != DISCR_NNP)? 1: -1;

        if (discrepancy == DISCR_PNZ)
        {
            where = ddist_t{1};
            end = 1;
        };

        if (discrepancy == DISCR_NNP) { what = udist_t(-3, -1); };
        ///printf("NEXT %d\n", _prank);
        ///fflush(stdout);

        // repeat while nonzero discrepancy
        for (int p = 0; total[inds[0]] || total[inds[1]]; ++p)
        {
            //printf("INSIDE %d\n", _prank);
            //fflush(stdout);
            
            if (p == _psize) { p = 0; }

            // (0)??
            if (!(total[inds[0]]))
            {
                discrepancy = DISCR_PNZ;
                what = udist_t(1, 3);
                where = ddist_t{1};
                end = 1;
                sign = 1; 

                Swap(inds[0], inds[2]);

                // (-)?0
                if (total[inds[0]] < 0) { Swap(inds[0], inds[1]); }
            }
            
            // ?(0)?
            if (!(total[inds[1]]))
            {
                discrepancy = DISCR_PNZ;
                what = udist_t(1, 3);
                where = ddist_t{1};
                end = 1;
                sign = 1;

                Swap(inds[1], inds[2]);

                // (-)?0
                if (total[inds[0]] < 0) { Swap(inds[0], inds[1]); }
            }
            //printf("NEXT %d\n", _prank);
            //fflush(stdout);

            pos = where(_generator);
            //printf("TEXN %d\n", _prank);
            //fflush(stdout);

            ///printf("==================== %d %d %d\n", 
            ///    what(_generator), total[inds[pos]],
            ///    sign * int(_partstate[(_partrank[p] << 2) + inds[pos]])
            ///);

            while (sign > 0 && !(_partstate[(_partrank[p] << 2) + inds[pos]]))
            {
                //printf("A pos = %d, discr = %s\n", pos, PrintDiscrepancy(discrepancy));
                //fflush(stdout);
                //printf("AX: %d %d\n", inds[pos], inds[end]);
                //printf("AIN: %d %d %d\n", total[inds[0]], total[inds[1]], total[inds[2]]);
                //for (int pp = 0; pp < _psize; ++pp)
                //{
                //    printf("A<> partstate = %d %d %d\n", _partstate[pp << 2], _partstate[(pp << 2) + 1], _partstate[(pp << 2) + 2]);
                //    fflush(stdout);
                //}
                //for (int pp = 0; pp < _psize; ++pp)
                //{
                //    printf("AIN: partstate = %d %d\n", _partstate[(pp << 2) + inds[pos]], _partstate[(pp << 2) + inds[end]]);
                //    fflush(stdout);
                //}
                //sleep(1);
                ++p; 
                if (p == _psize) { p = 0; }
                //printf("%d ", p);
            }

            //printf("ZZZZZ %d\n", _prank);
            //fflush(stdout);
            tmp = (sign > 0)?
                min(
                    min(what(_generator), total[inds[pos]]),
                    _partstate[(_partrank[p] << 2) + inds[pos]]
                ):
                absmin(sign, what(_generator), total[inds[pos]]);
            //printf("RRRRR %d\n", _prank);
            //fflush(stdout);

            //printf("BEFORE %d %d %d\n", total[inds[0]], total[inds[1]], total[inds[2]]);
            //for (int pp = 0; pp < _psize; ++pp)
            //{
            //    printf("BEFORE partstate = %d %d %d\n", _partstate[pp << 2], _partstate[(pp << 2) + 1], _partstate[(pp << 2) + 2]);
            //    fflush(stdout);
            //}
            //fflush(stdout);
            ///////printf("INDS %d %d %d\n", inds[0], inds[1], inds[2]);
            //printf("pos = %d, discr = %s, tmp = %d\n", pos, PrintDiscrepancy(discrepancy), tmp);
            //printf("CX: %d %d\n", inds[pos], inds[end]);
            //printf("CIN: %d %d %d\n", total[inds[0]], total[inds[1]], total[inds[2]]);
            //fflush(stdout);

            if (tmp)
            {
                ///for (int pp = 0; pp < _psize; ++pp)
                ///{
                ///    printf("A partstate = %d %d %d\n", _partstate[pp << 2], _partstate[(pp << 2) + 1], _partstate[(pp << 2) + 2]);
                ///    fflush(stdout);
                ///}
                total[inds[pos]] -= tmp;
                //printf("BPOS: %u\n", _partstate[(_partrank[p] << 2) + inds[pos]]);
                _partstate[(_partrank[p] << 2) + inds[pos]] -= tmp;
                //printf("APOS: %u\n", _partstate[(_partrank[p] << 2) + inds[pos]]);

                ///for (int pp = 0; pp < _psize; ++pp)
                ///{
                ///    printf("B partstate = %d %d %d\n", _partstate[pp << 2], _partstate[(pp << 2) + 1], _partstate[(pp << 2) + 2]);
                ///    fflush(stdout);
                ///}

                total[inds[end]] += tmp;
                ///printf("BEND: %u\n", _partstate[(_partrank[p] << 2) + inds[end]]);

                ///for (int pp = 0; pp < _psize; ++pp)
                ///{
                ///    printf("C partstate = %d %d %d\n", _partstate[pp << 2], _partstate[(pp << 2) + 1], _partstate[(pp << 2) + 2]);
                ///    fflush(stdout);
                ///}

                while (sign < 0 && !(_partstate[(_partrank[p] << 2) + inds[end]]))
                {
                    //printf("B pos = %d, discr = %s\n", pos, PrintDiscrepancy(discrepancy));
                    //fflush(stdout);
                    //printf("BX: %d %d\n", inds[pos], inds[end]);
                    //printf("BIN: %d %d %d\n", total[inds[0]], total[inds[1]], total[inds[2]]);
                    //for (int pp = 0; pp < _psize; ++pp)
                    //{
                    //    printf("B<> partstate = %d %d %d\n", _partstate[pp << 2], _partstate[(pp << 2) + 1], _partstate[(pp << 2) + 2]);
                    //    fflush(stdout);
                    //}
                    //for (int pp = 0; pp < _psize; ++pp)
                    //{
                    //    printf("BIN: partstate = %d %d\n", _partstate[(pp << 2) + inds[pos]], _partstate[(pp << 2) + inds[end]]);
                    //    fflush(stdout);
                    //}
                    ++p;
                    if (p == _psize) { p = 0; }
                    //printf("%d ", p);
                }

                ///printf("SIGN = %d\n", sign);
                _partstate[(_partrank[p] << 2) + inds[end]] += tmp;

                ///printf("AEND: %u\n", _partstate[(_partrank[p] << 2) + inds[end]]);
            }
            //printf("AFTER %d %d %d\n", total[inds[0]], total[inds[1]], total[inds[2]]);
            //fflush(stdout);

            ///for (int pp = 0; pp < _psize; ++pp)
            ///{
            ///    printf("D partstate = %d %d %d\n", _partstate[pp << 2], _partstate[(pp << 2) + 1], _partstate[(pp << 2) + 2]);
            ///    fflush(stdout);
            ///}
        }
    }
    //printf("OUTSIDE %d\n", _prank);
    //fflush(stdout);

    return;
}

void City::RedistributeMoving(void)
{
    if (_psize > 1)
    {
        // reduce process states
        MPI_Allreduce(
            _locstate, _totstate, 4, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD
        );

        ///printf("[%d] LOCSTATE BEFORE GUESS %d %d %d sum = %d %d\n",
        ///    _prank, _locstate[0], _locstate[1], _locstate[2],
        ///    _locstate[0] + _locstate[1] + _locstate[2], _locstate[3]
        ///);
        ///fflush(stdout);

        ///if (!_prank)
        ///printf("[%d] TOTSTATE %d %d %d sum = %d %d\n",
        ///    _prank, _totstate[0], _totstate[1], _totstate[2],
        ///    _totstate[0] + _totstate[1] + _totstate[2], _totstate[3]
        ///);
        ///fflush(stdout);

        GuessState();

        ///uint_t _tt[4];
        ///MPI_Allreduce(
        ///    _locstate, _tt, 4, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD
        ///);
        ///if (!_prank)
        ///printf("[%d] TOTSTATE BEFORE %d %d %d sum = %d %d\n",
        ///    _prank, _tt[0], _tt[1], _tt[2], _tt[0] + _tt[1] + _tt[2], _tt[3]
        ///);
        ///fflush(stdout);

        ///printf("[%d] LOCSTATE %d %d %d sum = %d %d\n",
        ///    _prank, _locstate[0], _locstate[1], _locstate[2],
        ///    _locstate[0] + _locstate[1] + _locstate[2], _locstate[3]
        ///);
        ///fflush(stdout);

        // gather process states
        MPI_Gather(
            _locstate, 4, MPI_UNSIGNED, _partstate, 4, MPI_UNSIGNED, 0,
            MPI_COMM_WORLD
        );

        // redistribute to processes
        if (!_prank) { Equilibrate(); }

        // scatter redistributed weights
        MPI_Scatter(
            _partstate, 4, MPI_UNSIGNED, _locstate, 4, MPI_UNSIGNED, 0,
            MPI_COMM_WORLD
        );
    }

    // set ones to be redistributed to the beginning
    // zeros, zeros everywhere (else)!
    int u = 0;

    //printf("%d %d %d\n", _locstate[0], _locstate[1], _locstate[2]);
    // night watch
    for ( ; u < _locstate[0]; ++u)
    {
        _houses[_moving[u]] = 0;
    }

    // wasteland
    for ( ; u < _locstate[0] + _locstate[1]; ++u)
    {
        _houses[_moving[u]] = 1;
    }

    // white walkers
    for ( ; u < _locstate[0] + _locstate[1] + _locstate[2]; ++u)
    {
        _houses[_moving[u]] = 2;
    }

    // shuffle moving houses
    Shuffle(_generator, _locstate[0] + _locstate[1] + _locstate[2], _moving, _houses);

    /// do Error check
    memset(_weights, 0, 3 * sizeof(uint_t));

    for (int s = 0; s < _size[0] * _size[1]; ++s)
    {
        ++(_weights[GetHouse(s)]);

        ///if (_prank)
        ///printf("%d ", GetHouse(s));
        ///fflush(stdout);
    }
    ///printf("\n[%d]: %d %d %d\n", _prank, _weights[0], _weights[1], _weights[2]);
    ///fflush(stdout);

    uint_t newweights[3];

    MPI_Reduce(
        _weights, newweights, 3, MPI_UNSIGNED, MPI_SUM, 0, MPI_COMM_WORLD
    );

    if (!_prank)
    {
        printf(
            "WEIGHTS: %d %d %d\n", newweights[0], newweights[1], newweights[2]
        );
    }
    /// end Error check

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
    fprintf(file, "%05d %05d\n%05d", _size[1], _size[0], 2);

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
            _size[1], 2
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

    MPI_Barrier(MPI_COMM_WORLD);
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
        FindMoving();
        RedistributeMoving();

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

    if (argc < 5)
    {
        int prank;
        MPI_Comm_rank(MPI_COMM_WORLD, &prank);

        if (!prank)
        {
            fprintf(
                stderr, "Not enough arguments: "
                "iterations, size, threshold, "
                "first fraction probability, empty probability\n"
            );
        }

        MPI_Abort(MPI_COMM_WORLD, 42);
    }

    uint_t iter;
    uint_t size;
    double thresh;
    double prob;
    double empty;

    sscanf(argv[1], "%u", &iter); 
    sscanf(argv[2], "%u", &size); 
    sscanf(argv[3], "%lf", &thresh); 
    sscanf(argv[4], "%lf", &prob); 
    sscanf(argv[5], "%lf", &empty); 

    City city(size, thresh, prob, empty);
    city.Iterate(iter);

    MPI_Finalize();

    return 0;
}

// schelling.cpp
