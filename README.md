## Nolina

Naive NumPy (and eventually TensorFlow) implementations of randomised linear algorithms, including:

| Method             | NumPy              | TensorFlow         |
| ------------------ |:------------------:| ------------------:|
| power              | :heavy_check_mark: |                    |
| inverse iteration  | :heavy_check_mark: |                    |
| Rayleigh iteration | :heavy_check_mark: |                    |
| Jacobi             |                    |                    |
| Steepest descent   | :heavy_check_mark: |                    |
| Conjugate gradient | :heavy_check_mark: |                    |
| Arnoldi            | :heavy_check_mark: |                    |


### References

- Wendland, Holger. Numerical Linear Algebra: An Introduction. Cambridge Texts in Applied Mathematics. Cambridge University Press, 2017.
- Mahoney, Michael W. ‘Lecture Notes on Randomized Linear Algebra’. ArXiv Preprint ArXiv:1608.04481, 2016. http://arxiv.org/abs/1608.04481.




### Installation

```bash
python setup.py install
```


### Development

```bash
conda env create -f environment.yml
```
