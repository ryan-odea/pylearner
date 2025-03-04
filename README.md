# learner-py

The `learner-py` package implements transfer learning methods for low-rank
matrix estimation. These methods leverage similarity in the latent row
and column spaces between the source and target populations to improve
estimation in the target population. The methods include the LatEnt
spAce-based tRaNsfer lEaRning (LEARNER) method and the direct projection
LEARNER (D-LEARNER) method described by [McGrath et
al. (2024)](https://doi.org/10.48550/arXiv.2412.20605).

## Installation

You can install the released version of `learner-py` from
[PyPi](https://pypi.org/project/learner-py/) with:

```python
pip install learner-py
```
#### LEARNER Method

We can apply the LEARNER method via the `learner` function.

This method allows for flexible patterns of heterogeneity across the
source and target populations. It consequently requires specifying the
tuning parameters `lambda_1` and `lambda_2` which control the degree of
transfer learning between the populations. These values can be selected
based on cross-validation via the `cv.learner` function. For example, we
can specify candidate values of 1, 10, and 100 for `lambda_1` and
`lambda_2` and select the optimal values based on cross-validation as
follows:

```python
res_cv = pylearner.cv_learner(
            Y_source=dat_highsim["Y_source"],
            Y_target=dat_highsim["Y_target"],
            lambda1_all=[1, 10, 100],
            lambda2_all=[1, 10, 100],
            step_size=0.003
        )

print(res_cv['lambda_1_min']) #100
print(res_cv["lambda_2_min"]) #1
```
## Example

We illustrate an example of how `learner` can be used. We first load the
package.

```python
import learner-py as pylearner
```

In this illustration, we will use one of the toy data sets in the
package (`dat_highsim`) that has a high degree of similarity between the
latent spaces of the source and target populations. The object
`dat_highsim` is a list which contains the observed source population
data matrix `Y_source` and the target population data matrix `Y_target`.
Since the data was simulated, the true values of the matrices are
included in `dat_highsim` as `Theta_source` and `Theta_target`.

#### LEARNER Method

We can apply the LEARNER method via the `learner` function.

This method allows for flexible patterns of heterogeneity across the
source and target populations. It consequently requires specifying the
tuning parameters `lambda_1` and `lambda_2` which control the degree of
transfer learning between the populations. These values can be selected
based on cross-validation via the `cv.learner` function. For example, we
can specify candidate values of 1, 10, and 100 for `lambda_1` and
`lambda_2` and select the optimal values based on cross-validation as
follows:

```python
res_cv = pylearner.cv_learner(
            Y_source=dat_highsim["Y_source"],
            Y_target=dat_highsim["Y_target"],
            lambda1_all=[1, 10, 100],
            lambda2_all=[1, 10, 100],
            step_size=0.003
        )

print(res_cv['lambda_1_min']) #100
print(res_cv["lambda_2_min"]) #1
```

Next, we apply the `learner` function with these values of `lambda1`
and `lambda2`:

```python
res_learner <- learner(Y_source = dat_highsim$Y_source, 
                       Y_target = dat_highsim$Y_target,
                       lambda1 = 100, lambda2 = 1, 
                       step_size = 0.003)
```

The LEARNER estimate is given by the `learner_estimate` component in the
output of the `learner` function, e.g.,

```python
print(res_learner['learner_estimate'][0:4, 0:4])
[[ 0.14688887 -1.63414688  1.18573076  0.00809396]
 [ 0.10877659 -0.2439277  -1.0061737  -0.05811972]
 [-0.62130277  1.04206753 -0.69077759  0.90570933]
 [ 1.58115883 -3.0000149   0.39780804 -2.12428512]]
```

#### D-LEARNER Method

We can apply the D-LEARNER method via the `dlearner` function. This
method makes stronger assumptions on the heterogeneity across the source
and target populations. It consequently does not rely on choosing tuning
parameters. The `dlearner` function can be applied as follows:

```python
res_dlearner = pylearner.dlearner(Y_source=dat_highsim["Y_source"],
                                  Y_target=dat_highsim["Y_target"])

print(res_dlearner['dlearner_estimate'][0:4, 0:4])

[[ 0.0959171  -1.72143637  1.15500149  0.00547827]
 [ 0.10771367 -0.3224348  -1.03557177 -0.10819564]
 [-0.72372751  0.86634922 -0.43528206  1.10580219]
 [ 1.63476957 -2.91985925  0.04876288 -2.35750409]]
```
