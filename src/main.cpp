#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <vector>
#include <complex>
#include <cmath>
#include <matplot/matplot.h>

namespace py = pybind11;
namespace mpl = matplot;

std::vector<std::complex<double>> dft(const std::vector<double>& signal) {
    int N = signal.size();
    std::vector<std::complex<double>> spectrum(N);

    for (int k=0; k<N;++k) {
        std::complex<double> sum(0.0, 0.0);
        for (int n=0; n < N; ++n) {
            double angle =-2 *mpl::pi *k*n/N;
            sum+=signal[n] * std::complex<double>(cos(angle), sin(angle));
        }
        spectrum[k]=sum;
    }
    return spectrum;
}

std::vector<double> idft(const std::vector<std::complex<double>>& spectrum) {
    int N=spectrum.size();
    std::vector<double> signal(N);

    for (int n = 0; n < N; ++n) {
        std::complex<double> sum(0.0, 0.0);
        for (int k=0; k<N; ++k) {
            double angle=2*mpl::pi * k* n/N;
            sum+=spectrum[k] *std::complex<double>(cos(angle), sin(angle));
        }
        signal[n] = sum.real() / N;
    }
    return signal;
}


std::vector<double> filter1d(const std::vector<double>& signal){
    int N=signal.size();
    std::vector<double> filtered(N);
    for (int i=1; i < N-1; ++i) {
        filtered[i] = (signal[i-1] + signal[i] + signal[i+1]) / 3.0;
    }
    filtered[0] = signal[0];
    filtered[N-1] = signal[N-1];
    return filtered;
}


std::vector<std::vector<double>> filter2d(const std::vector<std::vector<double>>& image) {
    int rows=image.size();
    int cols=image[0].size();
    std::vector<std::vector<double>> output(rows, std::vector<double>(cols));

    for (int i=1; i<rows-1; ++i) {
        for (int j=1; j<cols-1; ++j) {
            output[i][j]=
                (image[i - 1][j - 1] + image[i - 1][j] + image[i - 1][j + 1] +
                 image[i][j - 1]     + image[i][j]     + image[i][j + 1] +
                 image[i + 1][j - 1] + image[i + 1][j] + image[i + 1][j + 1]) / 9.0;
        }
    }
    return output;
}


std::vector<double> generate_sin(double freq, double sample_rate, double duration) {
    int N=static_cast<int>(sample_rate*duration);
    std::vector<double> signal(N);
    for (int n=0; n<N; ++n)
        signal[n] = sin(2*mpl::pi*freq*n/sample_rate);
    return signal;
}

std::vector<double> generate_cos(double freq, double sample_rate, double duration) {
    int N = static_cast<int>(sample_rate*duration);
    std::vector<double> signal(N);
    for (int n = 0; n < N; ++n)
        signal[n] = cos(2*mpl::pi*freq*n/sample_rate);
    return signal;
}

std::vector<double> generate_square(double freq, double sample_rate, double duration) {
    int N = static_cast<int>(sample_rate * duration);
    std::vector<double> signal(N);
    for (int n = 0;n<N; ++n)
        signal[n] =sin(2 * mpl::pi*freq *n / sample_rate) >= 0 ? 1.0 : -1.0;
    return signal;
}

std::vector<double> generate_sawtooth(double freq, double sample_rate, double duration) {
    int N = static_cast<int>(sample_rate*duration);
    std::vector<double> signal(N);
    for (int n = 0; n < N; ++n) {
        double t = static_cast<double>(n) / sample_rate;
        signal[n] = 2.0 *(t*freq -floor(t *freq + 0.5));
    }
    return signal;
}

void show_signals(
    const std::vector<double>& original,
    const std::vector<double>& noisy,
    const std::vector<double>& filtered,
    const std::vector<double>& reconstructed
) {
    std::vector<double> t(original.size());
    for (size_t i = 0; i < t.size(); ++i)
        t[i] = i;

    mpl::figure(true);
    mpl::subplot(2, 2, 1);
    mpl::plot(t, original);
    mpl::title("Oryginalny");

    mpl::subplot(2, 2, 2);
    mpl::plot(t, noisy);
    mpl::title("Zakłócony");

    mpl::subplot(2, 2, 3);
    mpl::plot(t, filtered);
    mpl::title("Po filtracji");

    mpl::subplot(2, 2, 4);
    mpl::plot(t, reconstructed);
    mpl::title("Zrekonstruowany");

    mpl::show();
}

void show_image(const std::vector<std::vector<double>>& original,
                const std::vector<std::vector<double>>& filtered) {
    mpl::figure(true);
    mpl::subplot(1, 2, 1);
    mpl::imagesc(original);
    mpl::title("Obraz oryginalny");
    mpl::subplot(1, 2, 2);
    mpl::imagesc(filtered);
    mpl::title("Po filtracji 2D");

    mpl::show();
}


#ifdef VERSION_INFO
    #define STRINGIFY(x) #x
    #define MACRO_STRINGIFY(x) STRINGIFY(x)
#endif

PYBIND11_MODULE(_core, m) {
    py::class_<std::complex<double>>(m, "complex")
        .def_property_readonly("real", [](const std::complex<double>& c) {return c.real(); })
        .def_property_readonly("imag", [](const std::complex<double>& c) {return c.imag(); });

    m.def("dft", &dft);
    m.def("idft", &idft);
    m.def("filter1d", &filter1d);
    m.def("filter2d", &filter2d);

    m.def("generate_sin", &generate_sin);
    m.def("generate_cos", &generate_cos);
    m.def("generate_square", &generate_square);
    m.def("generate_sawtooth", &generate_sawtooth);

    m.def("show_signals", &show_signals);
    m.def("show_image", &show_image);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
