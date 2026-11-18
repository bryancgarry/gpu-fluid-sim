/*
 * Module 9 Assignment: CUDA Advanced Libraries
 *
 * Generates a 2D turbulent velocity field, meant to be used to
 * initialize fluid simulations, then checks that the result has 
 * the energy spectrum it was asked for.
 *
 *   cuRAND: fills a stream function's spectrum with Gaussian noise
 *   cuFFT:  transforms the shaped spectrum into a real velocity field,
 *           then back again for analysis
 *   Thrust: works directly on cuFFT's output to measure the energy
 *           spectrum and verify the physics
 *
 * Three checks are run on every field:
 *   Spectrum slope: the fitted slope of E(k) should match the target
 *   Parseval:       energy summed in real space and in frequency space
 *                   should agree
 *   Divergence:     the field is built from a stream function, so it
 *                   should be have no divergence
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cufft.h>
#include <curand.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>

// Default parameters
#define DEFAULT_GRID_SIZE 512
#define DEFAULT_SLOPE     (-5.0f / 3.0f) // Kolmogorov spectrum
#define DEFAULT_SEED      42ULL
#define MIN_GRID_SIZE     32    // leaves room for the slope fit range

// Threads per block for custom kernels
#define BLOCK_SIZE        256

// u and v are transformed together in one batched cuFFT call
#define VELOCITY_COMPONENTS 2

// Wavenumber range used to fit the measured spectrum slope. Very low
// k has too few modes per ring to average out the randomness.
#define FIT_K_MIN         4
#define FIT_K_MAX(n)      ((n) / 4)

// Pass/fail tolerances for the three checks
#define SLOPE_TOLERANCE      0.1
#define ENERGY_TOLERANCE     1e-3
#define DIVERGENCE_TOLERANCE 1e-4

// Output image params
#define OUTPUT_FORMAT     "speed_%d_%.2f_%llu.pgm"
#define FILENAME_LENGTH   64
#define GRAY_MAX          255.0f

struct FieldBuffers {
    cufftComplex *d_psiHat; // random stream function spectrum
    cufftComplex *d_velHat; // u and v spectra, back to back
    float *d_vel;           // real u and v fields, back to back
    float *d_speed;         // speed for the output image
};

struct RunResults {
    float measuredSlope;    // fitted log-log slope of E(k)
    double energyRatio;     // frequency / real space energy (Parseval)
    double divergence;      // relative divergence, should be ~0
    float generateMs;       // cuRAND + shaping + inverse cuFFT
    float analyzeMs;        // forward cuFFT + Thrust analysis
};

/* ------------------------------------------------------------------ */
/* Device helpers                                                     */
/* ------------------------------------------------------------------ */

// Integer wavenumbers of a mode in cuFFT's R2C layout. Columns hold
// kx >= 0, rows past the middle hold negative ky.
__host__ __device__ void modeWavenumbers(int idx, int gridSize,
                                         float &kx, float &ky)
{
    int halfWidth = gridSize / 2 + 1;
    int row = idx / halfWidth;
    kx = idx % halfWidth;
    ky = (row <= gridSize / 2) ? row : row - gridSize;
}

// R2C only stores kx >= 0, so kx > 0 modes also stand in for their
// mirrored -kx partners and count twice in any sum over modes
__host__ __device__ float modeWeight(float kx)
{
    return (kx > 0.0f) ? 2.0f : 1.0f;
}

/* ------------------------------------------------------------------ */
/* Kernels                                                            */
/* ------------------------------------------------------------------ */

// Shape the Gaussian noise into a turbulence spectrum and build the
// velocity spectra from it. psiHat is a stream function, so
// uHat = i*ky*psiHat and vHat = -i*kx*psiHat, which makes the
// velocity field divergence-free by construction.
__global__ void shapeSpectrum(const cufftComplex *psiHat,
                              cufftComplex *velHat, int gridSize,
                              float slope)
{
    int modeCount = gridSize * (gridSize / 2 + 1);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= modeCount)
        return;

    float kx, ky;
    modeWavenumbers(idx, gridSize, kx, ky);
    float k = sqrtf(kx * kx + ky * ky);

    // E(k) ~ k^slope needs |psiHat| ~ k^((slope - 3) / 2).
    // The mean (k = 0) and Nyquist modes are left at zero.
    float amp = 0.0f;
    if (k > 0.0f && kx < gridSize / 2 && ky != gridSize / 2)
        amp = powf(k, (slope - 3.0f) * 0.5f);

    float a = psiHat[idx].x * amp;
    float b = psiHat[idx].y * amp;

    // Multiplying (a + bi) by i gives (-b + ai)
    velHat[idx]             = make_cuComplex(-ky * b,  ky * a);
    velHat[idx + modeCount] = make_cuComplex( kx * b, -kx * a);
}

// Speed sqrt(u^2 + v^2) at each cell. Speed follows the energy
// spectrum, so the large eddies stand out in the output image.
__global__ void computeSpeed(const float *vel, float *speed,
                             int gridSize)
{
    int cellCount = gridSize * gridSize;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cellCount)
        return;

    float u = vel[idx];
    float v = vel[idx + cellCount];
    speed[idx] = sqrtf(u * u + v * v);
}

/* ------------------------------------------------------------------ */
/* Thrust functors                                                    */
/* ------------------------------------------------------------------ */

// Wavenumber ring a spectral mode falls in (rounded to the nearest k)
struct ModeBin {
    int gridSize;

    __host__ __device__ int operator()(int idx) const
    {
        float kx, ky;
        modeWavenumbers(idx, gridSize, kx, ky);
        return (int)roundf(sqrtf(kx * kx + ky * ky));
    }
};

// Kinetic energy |uHat|^2 + |vHat|^2 of one mode, read straight from
// cuFFT's output buffer
struct ModeEnergy {
    const cufftComplex *velHat;
    int gridSize;

    __host__ __device__ float operator()(int idx) const
    {
        int modeCount = gridSize * (gridSize / 2 + 1);
        cufftComplex u = velHat[idx];
        cufftComplex v = velHat[idx + modeCount];
        float kx, ky;
        modeWavenumbers(idx, gridSize, kx, ky);
        return modeWeight(kx)
             * (u.x * u.x + u.y * u.y + v.x * v.x + v.y * v.y);
    }
};

// |kx*uHat + ky*vHat|^2, the spectral divergence of one mode
struct ModeDivergence {
    const cufftComplex *velHat;
    int gridSize;

    __host__ __device__ float operator()(int idx) const
    {
        int modeCount = gridSize * (gridSize / 2 + 1);
        cufftComplex u = velHat[idx];
        cufftComplex v = velHat[idx + modeCount];
        float kx, ky;
        modeWavenumbers(idx, gridSize, kx, ky);
        float re = kx * u.x + ky * v.x;
        float im = kx * u.y + ky * v.y;
        return modeWeight(kx) * (re * re + im * im);
    }
};

// k^2 times the mode's energy, the scale the divergence is compared to
struct ModeGradient {
    const cufftComplex *velHat;
    int gridSize;

    __host__ __device__ float operator()(int idx) const
    {
        float kx, ky;
        modeWavenumbers(idx, gridSize, kx, ky);
        return (kx * kx + ky * ky) * ModeEnergy{velHat, gridSize}(idx);
    }
};

struct Square {
    __host__ __device__ float operator()(float x) const
    {
        return x * x;
    }
};

// Map speed in [0, maxSpeed] to an 8-bit gray level
struct ToGray {
    float maxSpeed;

    __host__ __device__ unsigned char operator()(float s) const
    {
        return (unsigned char)(GRAY_MAX * s / maxSpeed);
    }
};

/* ------------------------------------------------------------------ */
/* Library wrappers                                                   */
/* ------------------------------------------------------------------ */

// Fill the stream function spectrum with complex Gaussian noise.
// A cufftComplex is two floats, so cuRAND writes directly into the
// buffer cuFFT will use later
void generateRandomSpectrum(cufftComplex *d_psiHat, int modeCount,
                            unsigned long long seed)
{
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateNormal(gen, (float *)d_psiHat, 2 * modeCount,
                         0.0f, 1.0f);
    curandDestroyGenerator(gen);
}

// Batched 2D plan covering both velocity components. The distance
// between u and v depends on which side of the transform is complex
cufftHandle makeVelocityPlan(int gridSize, cufftType type)
{
    int dims[2] = {gridSize, gridSize};
    int modeCount = gridSize * (gridSize / 2 + 1);
    int cellCount = gridSize * gridSize;
    int inDist  = (type == CUFFT_R2C) ? cellCount : modeCount;
    int outDist = (type == CUFFT_R2C) ? modeCount : cellCount;

    cufftHandle plan;
    cufftPlanMany(&plan, 2, dims, NULL, 1, inDist, NULL, 1, outDist,
                  type, VELOCITY_COMPONENTS);
    return plan;
}

// Inverse transform both velocity spectra to real space in one call.
// C2R overwrites its input, so d_velHat is scratch afterward
void spectrumToVelocity(cufftComplex *d_velHat, float *d_vel,
                        int gridSize)
{
    cufftHandle plan = makeVelocityPlan(gridSize, CUFFT_C2R);
    cufftExecC2R(plan, d_velHat, d_vel);
    cufftDestroy(plan);
}

// Forward transform the real velocity field back to spectra so the
// result can be checked
void velocityToSpectrum(float *d_vel, cufftComplex *d_velHat,
                        int gridSize)
{
    cufftHandle plan = makeVelocityPlan(gridSize, CUFFT_R2C);
    cufftExecR2C(plan, d_vel, d_velHat);
    cufftDestroy(plan);
}

// Radially binned energy spectrum E(k), computed by Thrust directly
// on cuFFT's output. Every integer ring from 0 up appears, so
// h_spectrum[k] ends up holding the energy in ring k. Returns the
// total energy over all modes.
double measureSpectrum(const cufftComplex *d_velHat, int gridSize,
                       float *h_spectrum)
{
    int modeCount = gridSize * (gridSize / 2 + 1);
    thrust::counting_iterator<int> first(0);
    thrust::device_vector<int> bins(modeCount);
    thrust::device_vector<float> energy(modeCount);
    thrust::device_vector<float> ringEnergy(gridSize);

    thrust::transform(first, first + modeCount, bins.begin(),
                      ModeBin{gridSize});
    thrust::transform(first, first + modeCount, energy.begin(),
                      ModeEnergy{d_velHat, gridSize});

    // Group modes by ring, then sum each ring's energy
    thrust::sort_by_key(bins.begin(), bins.end(), energy.begin());
    auto ends = thrust::reduce_by_key(bins.begin(), bins.end(),
                                      energy.begin(),
                                      thrust::make_discard_iterator(),
                                      ringEnergy.begin());
    thrust::copy(ringEnergy.begin(), ends.second, h_spectrum);
    return thrust::reduce(ringEnergy.begin(), ends.second, 0.0,
                          thrust::plus<double>());
}

// Sum of u^2 + v^2 over the real space grid
double realEnergy(const float *d_vel, int cellCount)
{
    thrust::device_ptr<const float> vel(d_vel);
    return thrust::transform_reduce(vel,
                                    vel + VELOCITY_COMPONENTS * cellCount,
                                    Square(), 0.0, thrust::plus<double>());
}

// RMS spectral divergence relative to the RMS velocity gradient.
// Exact zero in theory, so anything ~0 will pass
double relativeDivergence(const cufftComplex *d_velHat, int gridSize)
{
    int modeCount = gridSize * (gridSize / 2 + 1);
    thrust::counting_iterator<int> first(0);

    double divergence = thrust::transform_reduce(
        thrust::device, first, first + modeCount,
        ModeDivergence{d_velHat, gridSize}, 0.0, thrust::plus<double>());
    double gradient = thrust::transform_reduce(
        thrust::device, first, first + modeCount,
        ModeGradient{d_velHat, gridSize}, 0.0, thrust::plus<double>());
    return sqrt(divergence / gradient);
}

/* ------------------------------------------------------------------ */
/* Helpers                                                            */
/* ------------------------------------------------------------------ */

void allocateBuffers(FieldBuffers &buf, int gridSize)
{
    size_t modeCount = (size_t)gridSize * (gridSize / 2 + 1);
    size_t cellCount = (size_t)gridSize * gridSize;

    cudaMalloc(&buf.d_psiHat, modeCount * sizeof(cufftComplex));
    cudaMalloc(&buf.d_velHat,
               VELOCITY_COMPONENTS * modeCount * sizeof(cufftComplex));
    cudaMalloc(&buf.d_vel, VELOCITY_COMPONENTS * cellCount * sizeof(float));
    cudaMalloc(&buf.d_speed, cellCount * sizeof(float));
}

void freeBuffers(FieldBuffers &buf)
{
    cudaFree(buf.d_psiHat);
    cudaFree(buf.d_velHat);
    cudaFree(buf.d_vel);
    cudaFree(buf.d_speed);
}

// Least squares slope of log E(k) against log k over [kMin, kMax]
float fitSlope(const float *spectrum, int kMin, int kMax)
{
    double sumX = 0.0, sumY = 0.0, sumXX = 0.0, sumXY = 0.0;
    int count = kMax - kMin + 1;

    for (int k = kMin; k <= kMax; k++) {
        double x = log((double)k);
        double y = log((double)spectrum[k]);
        sumX += x;
        sumY += y;
        sumXX += x * x;
        sumXY += x * y;
    }
    return (count * sumXY - sumX * sumY) / (count * sumXX - sumX * sumX);
}

// cuRAND noise into spectral shaping into inverse cuFFT
void generateField(FieldBuffers &buf, int gridSize, float slope,
                   unsigned long long seed)
{
    int modeCount = gridSize * (gridSize / 2 + 1);
    int blocks = (modeCount + BLOCK_SIZE - 1) / BLOCK_SIZE;

    generateRandomSpectrum(buf.d_psiHat, modeCount, seed);
    shapeSpectrum<<<blocks, BLOCK_SIZE>>>(buf.d_psiHat, buf.d_velHat,
                                          gridSize, slope);
    spectrumToVelocity(buf.d_velHat, buf.d_vel, gridSize);
}

// Forward cuFFT, then Thrust runs the three checks on its output
void analyzeField(FieldBuffers &buf, int gridSize, RunResults &results)
{
    int cellCount = gridSize * gridSize;
    float *h_spectrum = (float *)malloc(gridSize * sizeof(float));

    velocityToSpectrum(buf.d_vel, buf.d_velHat, gridSize);
    double spectralEnergy = measureSpectrum(buf.d_velHat, gridSize,
                                            h_spectrum);

    // cuFFT is unnormalized, so Parseval's theorem picks up a
    // factor of cellCount between the two sums
    results.energyRatio = spectralEnergy
                        / (cellCount * realEnergy(buf.d_vel, cellCount));
    results.divergence = relativeDivergence(buf.d_velHat, gridSize);
    results.measuredSlope = fitSlope(h_spectrum, FIT_K_MIN,
                                     FIT_K_MAX(gridSize));
    free(h_spectrum);
}

// Speed kernel, then Thrust finds the peak speed and converts the
// field to gray levels on the GPU before writing a binary PGM image
void writeSpeedImage(FieldBuffers &buf, int gridSize,
                     const char *filename)
{
    int cellCount = gridSize * gridSize;
    int blocks = (cellCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    computeSpeed<<<blocks, BLOCK_SIZE>>>(buf.d_vel, buf.d_speed,
                                         gridSize);

    // Zero speed is black, the fastest cell is white
    thrust::device_ptr<float> speed(buf.d_speed);
    float maxSpeed = *thrust::max_element(speed, speed + cellCount);

    thrust::device_vector<unsigned char> pixels(cellCount);
    thrust::transform(speed, speed + cellCount, pixels.begin(),
                      ToGray{maxSpeed});
    thrust::host_vector<unsigned char> h_pixels = pixels;

    FILE *file = fopen(filename, "wb");
    fprintf(file, "P5\n%d %d\n255\n", gridSize, gridSize);
    fwrite(h_pixels.data(), 1, cellCount, file);
    fclose(file);
}

const char *passFail(bool passed)
{
    return passed ? "PASS" : "FAIL";
}

// Report results
void report(int gridSize, float slope, unsigned long long seed,
            const RunResults &results, const char *filename)
{
    printf("Grid %d x %d | Target slope %.3f | Seed %llu\n",
           gridSize, gridSize, slope, seed);
    printf("  Generate (cuRAND + cuFFT) : %9.3f ms\n", results.generateMs);
    printf("  Analyze  (cuFFT + Thrust) : %9.3f ms\n", results.analyzeMs);
    printf("  Measured slope            : %9.3f   %s\n",
           results.measuredSlope,
           passFail(fabs(results.measuredSlope - slope) < SLOPE_TOLERANCE));
    printf("  Parseval energy ratio     : %9.6f   %s\n",
           results.energyRatio,
           passFail(fabs(results.energyRatio - 1.0) < ENERGY_TOLERANCE));
    printf("  Relative divergence       : %9.2e   %s\n",
           results.divergence,
           passFail(results.divergence < DIVERGENCE_TOLERANCE));
    printf("  Speed image               : %s\n\n", filename);
}

// Warm up, time the generate and analyze stages, then report results
void runPipeline(int gridSize, float slope, unsigned long long seed)
{
    FieldBuffers buf;
    RunResults results;
    allocateBuffers(buf, gridSize);

    // Cut library startup cost before anything is timed
    generateField(buf, gridSize, slope, seed);
    analyzeField(buf, gridSize, results);

    cudaEvent_t start, generated, analyzed;
    cudaEventCreate(&start);
    cudaEventCreate(&generated);
    cudaEventCreate(&analyzed);

    cudaEventRecord(start);
    generateField(buf, gridSize, slope, seed);
    cudaEventRecord(generated);
    analyzeField(buf, gridSize, results);
    cudaEventRecord(analyzed);
    cudaEventSynchronize(analyzed);
    cudaEventElapsedTime(&results.generateMs, start, generated);
    cudaEventElapsedTime(&results.analyzeMs, generated, analyzed);

    char filename[FILENAME_LENGTH];
    snprintf(filename, FILENAME_LENGTH, OUTPUT_FORMAT,
             gridSize, slope, seed);
    writeSpeedImage(buf, gridSize, filename);
    report(gridSize, slope, seed, results, filename);

    cudaEventDestroy(start);
    cudaEventDestroy(generated);
    cudaEventDestroy(analyzed);
    freeBuffers(buf);
}

int main(int argc, char **argv)
{
    int gridSize = (argc > 1) ? atoi(argv[1]) : DEFAULT_GRID_SIZE;
    float slope = (argc > 2) ? atof(argv[2]) : DEFAULT_SLOPE;

    // Read seed from command line (or fall back to default)
    unsigned long long seed =
        (argc > 3) ? strtoull(argv[3], NULL, 10) : DEFAULT_SEED;

    if (gridSize < MIN_GRID_SIZE || slope >= 0.0f) {
        fprintf(stderr, "Usage: %s [gridSize (>= %d)] [slope (< 0)] "
                "[seed]\n", argv[0], MIN_GRID_SIZE);
        return EXIT_FAILURE;
    }

    runPipeline(gridSize, slope, seed);
    return 0;
}