#include <complex>
#include <iostream>
#include <valarray>
#include <math.h>


const double PI = 3.141592653589793238460;

typedef std::complex<double> Complex;
typedef std::valarray<Complex> CArray;

// Cooley–Tukey FFT (in-place, divide-and-conquer)
// Higher memory requirements and redundancy although more intuitive


/*
void fft(CArray& x)
{
	const size_t N = x.size();
	if (N <= 1) return;

	// divide
	CArray even = x[std::slice(0, N / 2, 2)];
	CArray  odd = x[std::slice(1, N / 2, 2)];

	// conquer
	fft(even);
	fft(odd);

	// combine
	for (size_t k = 0; k < N / 2; ++k)
	{
		Complex t = std::polar(1.0, -2 * PI * k / N) * odd[k];
		x[k] = even[k] + t;
		x[k + N / 2] = even[k] - t;
	}
}
*/


//Void fft(CArray& x)
CArray fft(CArray& x)
{
	// DFT
	unsigned int N = x.size(), k = N, n;
	//unsigned int N = sizeof(x);
	//unsigned int k = N, n;

	double thetaT = 3.14159265358979323846264338328L / N;
	Complex phiT = Complex(cos(thetaT), -sin(thetaT)), T;
//	CArray Temp;
	while (k > 1)
	{
		n = k;
		k >>= 1;
		phiT = phiT * phiT;
		T = 1.0L;
		for (unsigned int l = 0; l < k; l++)
		{
			for (unsigned int a = l; a < N; a += n)
			{
				unsigned int b = a + k;
				Complex t = x[a] - x[b];
				x[a] += x[b];
				x[b] = t * T;
			}
			T *= phiT;
		}
	}
	// Decimate
	unsigned int m = (unsigned int)log2(N);
	for (unsigned int a = 0; a < N; a++)
	{
		unsigned int b = a;
		// Reverse bits
		b = (((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1));
		b = (((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2));
		b = (((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4));
		b = (((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8));
		b = ((b >> 16) | (b << 16)) >> (32 - m);
		if (b > a)
		{
			Complex t = x[a];
			x[a] = x[b];
			x[b] = t;
		}
	}
	return x;
}



// inverse fft (in-place)
//Void ifft(CArray& x)
CArray ifft(CArray& x)
{
	Complex Temp;
	// conjugate the complex numbers
	x = x.apply(std::conj);

	// forward fft
//	fft(x);
	x=fft(x);

	// conjugate the complex numbers again
	x = x.apply(std::conj);

	// scale the numbers
	x /= x.size();
	return x;
}

int main()
{
	const int Nx = 128;
	int Nx_half;
	int Nt;
	Complex xx[Nx] = { };
	Complex u[Nx] = { };
	Complex k[Nx] = { };
	Complex kernel[Nx] = {};
	CArray Temp1;
	CArray Kernel;
	double A, B;
	double hx;
	double T_f;
	double dt;

	//End points of domain (1-dimensional)
	A = -12.0;
	B = 12.0;
	hx = (B - A) / (Nx - 1);

	Nt = 10;
	T_f = 0.5;
	dt = T_f / double(Nt);

	//Set up the reciprocal lattice:
	Nx_half = round(Nx / 2);
	for (int i = 0; i < Nx_half + 1; i++) {
		k[i] = 2.0 * PI / (B - A + hx) * i;
	}
	for (int i = 0; i < Nx_half - 1; i++) {
		k[Nx - i - 1] = -2.0 * PI / (B - A + hx) * (i + 1);
	}

	//Set up the domain and initial condition:
	for (int i = 0; i < Nx; i++) {
		xx[i] = A + (B - A) / (Nx - 1) * i;
		u[i] = exp(-1.0 * pow(xx[i], 2));
	}
	CArray data(u, Nx);

	//Creating the FFT of the Green's function for heat equation
	for (int i = 0; i < Nx; i++)
	{
		kernel[i] = exp(-dt / 2.0 * pow(k[i], 2));
	}
	Kernel = CArray(kernel, Nx);

	//Applying the convolution theorem to propagate the solution through time
	for (int i = 0; i < Nt - 1; i++)
	{
		Temp1 = Kernel * fft(data);
		data = ifft(Temp1);
	}

	for (int i = 0; i < Nx; i++)
	{
		std::cout << data[i] << std::endl;
	}


	return 0;
}



