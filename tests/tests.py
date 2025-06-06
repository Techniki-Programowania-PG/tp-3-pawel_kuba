import scikit_build_example as sc
import random

fs =1000
duration =0.05
freq = 50
N = int(fs *duration)

sine =sc.generate_sin(freq, fs, duration)
noisy = [s+0.1 * (2*random.random()-1) for s in sine]
filtered = sc.filter1d(noisy)
reconstructed = sc.idft(sc.dft(sine))

image = [
    [10]*15,
    [10,80,80,80,80,80,80,80,80,80,80,80,80,80,10],
    [10,80,30,30,30,30,30,30,30,30,30,30,30,80,10],
    [10,80,30,100,100,100,100,100,100,100,100,100,30,80,10],
    [10,80,30,100,50,50,50,50,50,50,50,100,30,80,10],
    [10,80,30,100,50,80,80,80,80,80,50,100,30,80,10],
    [10,80,30,100,50,80,100,100,100,80,50,100,30,80,10],
    [10,80,30,100,50,80,100,200,100,80,50,100,30,80,10],
    [10,80,30,100,50,80,100,100,100,80,50,100,30,80,10],
    [10,80,30,100,50,80,80,80,80,80,50,100,30,80,10],
    [10,80,30,100,50,50,50,50,50,50,50,100,30,80,10],
    [10,80,30,100,100,100,100,100,100,100,100,100,30,80,10],
    [10,80,30,30,30,30,30,30,30,30,30,30,30,80,10],
    [10,80,80,80,80,80,80,80,80,80,80,80,80,80,10],
    [10]*15
]

filtered_image = sc.filter2d(image)

sc.show_signals(sine, noisy, filtered, reconstructed)
sc.show_image(image, filtered_image)
